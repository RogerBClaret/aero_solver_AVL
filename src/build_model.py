
"""Build a single half-wing AVL model and one operating case."""

from pathlib import Path

import yaml

from avlwrapper import (
    Geometry, Surface, Section, Point,
    NacaAirfoil, Control, Spacing,
    Case, Parameter
)

ROOT = Path(__file__).resolve().parents[1]


def load_inputs(folder=None):
    folder = Path(folder) if folder else ROOT / "input"

    def read(name):
        with (folder / name).open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    return (
        read("geometry.yaml"),
        read("mesh.yaml"),
        read("control_surfaces.yaml"),
        read("flight_condition.yaml"),
    )


def interpolate(sections, y):
    """Linearly interpolate wing geometry at position y."""

    for a, b in zip(sections[:-1], sections[1:]):
        if a["y"] <= y <= b["y"]:
            t = (y - a["y"]) / (b["y"] - a["y"])

            return {
                key: a[key] + t * (b[key] - a[key])
                for key in (
                    "x_le", "z_le", "chord", "twist"
                )
            }

    raise ValueError(f"Invalid span position: {y}")


def build_geometry(geometry_data, mesh_data, controls_data):
    """Build ONE half-wing Surface, without mirroring."""

    cfg = geometry_data["geometry"]
    wing_cfg = geometry_data["surfaces"][0]
    mesh = mesh_data["mesh"]
    controls_cfg = controls_data["control_surfaces"]

    # Enforce half-wing-only calculations.
    if cfg.get("symmetric", False):
        raise ValueError(
            "Set symmetric: false in geometry.yaml"
        )

    original_sections = sorted(
        wing_cfg["sections"],
        key=lambda s: s["y"]
    )

    semispan = original_sections[-1]["y"]

    if original_sections[0]["y"] != 0:
        raise ValueError("Wing must begin at y=0")

    # Add sections at control-surface boundaries.
    positions = {
        float(s["y"]) for s in original_sections
    }

    names = [c["name"] for c in controls_cfg]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate control names")

    for ctrl in controls_cfg:
        if ctrl["surface"] != wing_cfg["name"]:
            raise ValueError(
                f"Unknown wing for {ctrl['name']}"
            )

        start = float(ctrl["span"]["start"])
        end = float(ctrl["span"]["end"])

        if not 0 <= start < end <= 1:
            raise ValueError(
                f"Invalid span limits: {ctrl['name']}"
            )

        positions.add(start * semispan)
        positions.add(end * semispan)

    positions = sorted(positions)

    # Airfoil: NACA or flat plate.
    airfoil_cfg = wing_cfg.get("airfoil", {})
    airfoil = None

    if airfoil_cfg.get("type") == "naca":
        airfoil = NacaAirfoil(
            naca=str(airfoil_cfg["code"])
        )

    sections = []

    for y in positions:
        data = interpolate(original_sections, y)
        section_controls = []

        for ctrl in controls_cfg:
            y_start = ctrl["span"]["start"] * semispan
            y_end = ctrl["span"]["end"] * semispan

            if y_start <= y <= y_end:
                t = (y - y_start) / (y_end - y_start)
                hinge = ctrl["hinge"]

                x_hinge = (
                    hinge["root_chord_fraction"]
                    + t * (
                        hinge["tip_chord_fraction"]
                        - hinge["root_chord_fraction"]
                    )
                )

                section_controls.append(
                    Control(
                        name=ctrl["name"],
                        gain=ctrl.get("gain", 1.0),
                        x_hinge=x_hinge,
                        duplicate_sign=1.0
                    )
                )

        sections.append(
            Section(
                leading_edge_point=Point(
                    data["x_le"], y, data["z_le"]
                ),
                chord=data["chord"],
                angle=data["twist"],
                airfoil=airfoil,
                controls=section_controls
            )
        )

    spacing = {
        "uniform": Spacing.equal,
        "cosine": Spacing.cosine
    }

    wing = Surface(
        name="Wing",
        n_chordwise=int(mesh["chordwise_panels"]),
        chord_spacing=spacing[mesh["chord_spacing"]],
        n_spanwise=int(mesh["spanwise_panels"]),
        span_spacing=spacing[mesh["span_spacing"]],
        y_duplicate=None,  # ONLY the modeled half-wing
        sections=sections
    )

    # Compute reference quantities automatically.
    half_area = 0.0
    mac_integral = 0.0

    for a, b in zip(
        original_sections[:-1],
        original_sections[1:]
    ):
        dy = b["y"] - a["y"]
        ca, cb = a["chord"], b["chord"]

        if dy <= 0 or min(ca, cb) <= 0:
            raise ValueError("Invalid wing geometry")

        half_area += 0.5 * (ca + cb) * dy
        mac_integral += (
            dy * (ca**2 + ca * cb + cb**2) / 3
        )

    ref = cfg["reference"]

    geometry = Geometry(
        name=cfg["name"],
        reference_area=half_area,      # Modeled half-wing
        reference_span=2 * semispan,   # Equivalent full span
        reference_chord=mac_integral / half_area,
        reference_point=Point(*ref["point"]),
        surfaces=[wing]
    )

    return geometry


def build_case(flight_data, controls_data):
    """Build ONE AVL case, optionally trimming one variable.

    A fixed variable uses its numerical value.
    A trimmed variable uses AVLWrapper Parameter.

    Trimming is performed by AVL within this single case.
    """

    flight = flight_data["flight_condition"]

    known_controls = [
        c["name"]
        for c in controls_data["control_surfaces"]
    ]

    deflections = flight.get("controls", {})

    unknown = set(deflections) - set(known_controls)

    if unknown:
        raise ValueError(
            f"Unknown control deflections: {unknown}"
        )

    kwargs = {
        "alpha": float(flight.get("alpha", 0.0)),
        "beta": float(flight.get("beta", 0.0)),
        "velocity": float(flight["velocity"]),
        "density": float(flight["density"]),
    }

    for name in known_controls:
        kwargs[name] = float(deflections.get(name, 0.0))

    trim = flight.get("trim", {})

    if trim.get("enabled", False):
        variable = trim["variable"]
        constraint = trim["constraint"]
        target = float(trim["target"])

        allowed = {"alpha"} | set(known_controls)

        if variable not in allowed:
            raise ValueError(
                f"Invalid trim variable: {variable}"
            )

        kwargs[variable] = Parameter(
            name=variable,
            constraint=constraint,
            value=target
        )

    return Case(
        name="operating_point",
        **kwargs
    )
