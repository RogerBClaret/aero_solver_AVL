
"""
Construct AVL geometry from the aircraft input files.

The YAML defines the right semi-wing.
AVL mirrors the surface across Y=0 using YDUPLICATE.
"""

from pathlib import Path

from avlwrapper import (
    Surface, Section, Point, Control, Spacing,
    NacaAirfoil, FileAirfoil,
)

try:
    from avlwrapper import Geometry
except ImportError:
    from avlwrapper import Aircraft as Geometry

from .input_loader import ROOT
from .wing_geometry import (
    validate_sections,
    interpolate_section,
    section_coordinates,
    calculate_reference,
)


def build_airfoil(config):
    """Create an AVL airfoil from the YAML specification."""

    kind = config["type"].lower()

    # Option A: External airfoil coordinates
    if kind == "file":
        path = Path(config["path"])

        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            raise FileNotFoundError(f"Airfoil not found: {path}")

        return FileAirfoil(str(path.resolve()))

    # Option B: NACA 4-digit airfoil
    if kind == "naca":
        code = str(config["code"])

        if len(code) != 4 or not code.isdigit():
            raise ValueError("Expected a NACA 4-digit code.")

        return NacaAirfoil(naca=code)

    # Option C: Flat plate (AVL default)
    if kind == "flat_plate":
        return None

    raise ValueError(f"Unknown airfoil type: {kind}")


def validate_controls(controls, surface_name):
    """Validate control names, span limits and hinge locations."""

    names = [c["name"] for c in controls]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate control names.")

    for control in controls:
        name = control["name"]

        if control["surface"] != surface_name:
            raise ValueError(f"Invalid surface for {name}.")

        start = float(control["span"]["start"])
        end = float(control["span"]["end"])

        if not 0 <= start < end <= 1:
            raise ValueError(f"Invalid span for {name}.")

        hinge = control["hinge"]

        for key in ("root_chord_fraction", "tip_chord_fraction"):
            if not 0 < float(hinge[key]) < 1:
                raise ValueError(f"Invalid hinge for {name}.")

        if float(control.get("duplicate_sign", 1)) not in (-1, 1):
            raise ValueError(f"Invalid duplicate_sign for {name}.")


def controls_at_section(controls, eta):
    """
    Construct controls active at a spanwise section.

    Hinge location is interpolated between the inboard
    and outboard boundaries of each control surface.
    """

    result = []

    for control in controls:
        start = float(control["span"]["start"])
        end = float(control["span"]["end"])

        if not start <= eta <= end:
            continue

        t = (eta - start) / (end - start)
        hinge = control["hinge"]

        x_hinge = (
            float(hinge["root_chord_fraction"])
            + t * (
                float(hinge["tip_chord_fraction"])
                - float(hinge["root_chord_fraction"])
            )
        )

        result.append(
            Control(
                name=control["name"],
                gain=float(control.get("gain", 1.0)),
                x_hinge=x_hinge,
                duplicate_sign=float(
                    control.get("duplicate_sign", 1.0)
                ),
            )
        )

    return result


def build_geometry(geometry_data, mesh_data, controls_data):
    """Build the AVL geometry for one symmetric wing."""

    cfg = geometry_data["geometry"]
    surfaces = geometry_data["surfaces"]

    if len(surfaces) != 1:
        raise ValueError("This model currently supports one wing.")

    wing_cfg = surfaces[0]
    planform = wing_cfg["planform"]

    mesh = mesh_data["mesh"]
    controls = controls_data.get("control_surfaces", [])

    # 1. Validate the user-defined sections and controls.
    original = validate_sections(wing_cfg["sections"])
    validate_controls(controls, wing_cfg["name"])

    # 2. Calculate full-wing reference quantities.
    ref = calculate_reference(planform, original)

    # 3. Insert stations at all control boundaries.
    stations = {float(s["y"]) for s in original}

    for control in controls:
        stations.add(float(control["span"]["start"]))
        stations.add(float(control["span"]["end"]))

    stations = sorted(stations)

    # 4. Define the airfoil and discretization.
    airfoil = build_airfoil(wing_cfg["airfoil"])

    spacing = {
        "uniform": Spacing.equal,
        "cosine": Spacing.cosine,
    }

    chord_spacing = spacing[mesh["chord_spacing"]]
    span_spacing = spacing[mesh["span_spacing"]]

    nc = int(mesh["chordwise_panels"])
    ns = int(mesh["spanwise_panels"])

    if nc <= 0 or ns <= 0:
        raise ValueError("Panel counts must be positive.")

    if ns < len(stations) - 1:
        raise ValueError("Insufficient spanwise panels.")

    # 5. Build dimensional AVL sections.
    avl_sections = []

    for eta in stations:
        normalized = interpolate_section(original, eta)
        dimensional = section_coordinates(planform, normalized)

        avl_sections.append(
            Section(
                leading_edge_point=Point(
                    dimensional["x_le"],
                    dimensional["y_le"],
                    dimensional["z_le"],
                ),
                chord=dimensional["chord"],
                angle=dimensional["twist"],
                airfoil=airfoil,
                controls=controls_at_section(controls, eta),
            )
        )

    # 6. Build the wing and mirror it about Y=0.
    wing = Surface(
        name=wing_cfg["name"],
        n_chordwise=nc,
        chord_spacing=chord_spacing,
        n_spanwise=ns,
        span_spacing=span_spacing,
        y_duplicate=0.0,
        sections=avl_sections,
    )

    # 7. Assemble the AVL aircraft geometry.
    return Geometry(
        name=cfg["name"],
        reference_area=ref["Sref"],
        reference_span=ref["Bref"],
        reference_chord=ref["Cref"],
        reference_point=Point(*ref["reference_point"]),
        surfaces=[wing],
    )


