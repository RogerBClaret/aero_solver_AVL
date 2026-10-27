
"""
Plot one semi-wing and its AVL spanwise distributions.

Outputs:
    outputs/01_wing_mesh.png
    outputs/02_spanwise_distributions.png

The mesh is reconstructed from the YAML geometry and
discretization settings. Aerodynamic data come from AVL.

No section markers or section labels are plotted.
"""

from pathlib import Path
from math import cos, pi

import numpy as np
import matplotlib.pyplot as plt

from .input_loader import ROOT
from .wing_geometry import (
    validate_sections,
    interpolate_section,
    section_coordinates,
)


# Default output directory.
OUTPUT_DIR = ROOT / "outputs"


# ============================================================
# 1. GEOMETRY HELPERS
# ============================================================

def _wing_point(planform, sections, eta, xsi):
    """
    Return (X,Y) for a point on the wing planform.

    eta : normalized semi-span coordinate
    xsi : normalized local chord coordinate
          0 = leading edge, 1 = trailing edge
    """

    section = interpolate_section(sections, float(eta))
    p = section_coordinates(planform, section)

    return (
        p["x_le"] + xsi * p["chord"],
        p["y_le"],
    )


def _spacing(n, kind):
    """Generate uniform or cosine-distributed coordinates."""

    t = np.linspace(0.0, 1.0, n + 1)

    if kind == "cosine":
        return 0.5 * (1.0 - np.cos(pi * t))

    if kind == "uniform":
        return t

    raise ValueError(f"Unknown panel spacing: {kind}")


def _panel_counts(stations, n_total):
    """
    Allocate spanwise panels across section intervals.

    Each interval receives at least one panel.
    Remaining panels are allocated approximately in
    proportion to interval length.

    This is a visualization allocation, not AVL's
    exact internal panel allocation.
    """

    lengths = np.diff(stations)
    intervals = len(lengths)

    if n_total < intervals:
        raise ValueError("Too few spanwise panels.")

    counts = np.ones(intervals, dtype=int)
    remaining = n_total - intervals

    weights = remaining * lengths / lengths.sum()
    counts += np.floor(weights).astype(int)

    # Distribute any leftover panels.
    leftover = n_total - counts.sum()
    fractions = weights - np.floor(weights)

    for idx in np.argsort(-fractions)[:leftover]:
        counts[idx] += 1

    return counts


# ============================================================
# 2. PLOT WING MESH AND CONTROL SURFACES
# ============================================================

def plot_wing_mesh(
    geometry_data,
    mesh_data,
    controls_data,
    output_dir=OUTPUT_DIR,
):
    """
    Plot the right semi-wing in top view.

    Includes:
    - Leading and trailing edges
    - Aerodynamic panel grid
    - Control-surface regions and hinge lines

    Excludes:
    - Wing section markers
    - Wing section labels
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wing = geometry_data["surfaces"][0]
    planform = wing["planform"]
    sections = validate_sections(wing["sections"])
    mesh = mesh_data["mesh"]
    controls = controls_data.get("control_surfaces", [])

    nc = int(mesh["chordwise_panels"])
    ns = int(mesh["spanwise_panels"])

    # Mesh boundaries include control-surface endpoints.
    stations = {float(s["y"]) for s in sections}

    for control in controls:
        stations.add(float(control["span"]["start"]))
        stations.add(float(control["span"]["end"]))

    stations = np.array(sorted(stations))
    counts = _panel_counts(stations, ns)

    # Assemble normalized spanwise mesh boundaries.
    eta_grid = []

    for i, count in enumerate(counts):
        local = _spacing(count, mesh["span_spacing"])

        eta = stations[i] + (
            stations[i + 1] - stations[i]
        ) * local

        if i > 0:
            eta = eta[1:]

        eta_grid.extend(eta)

    eta_grid = np.asarray(eta_grid)
    xsi_grid = _spacing(nc, mesh["chord_spacing"])

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Wing outline.
    eta_smooth = np.linspace(0, 1, 250)

    leading = np.array([
        _wing_point(planform, sections, eta, 0.0)
        for eta in eta_smooth
    ])

    trailing = np.array([
        _wing_point(planform, sections, eta, 1.0)
        for eta in eta_smooth
    ])

    polygon = np.vstack((leading, trailing[::-1]))

    ax.fill(
        polygon[:, 1], polygon[:, 0],
        color="0.94", zorder=0
    )

    # Chordwise panel boundaries.
    for xsi in xsi_grid:
        points = np.array([
            _wing_point(planform, sections, eta, xsi)
            for eta in eta_grid
        ])

        ax.plot(
            points[:, 1], points[:, 0],
            color="0.72", lw=0.65, zorder=1
        )

    # Spanwise panel boundaries.
    for eta in eta_grid:
        points = np.array([
            _wing_point(planform, sections, eta, xsi)
            for xsi in xsi_grid
        ])

        ax.plot(
            points[:, 1], points[:, 0],
            color="0.72", lw=0.65, zorder=1
        )

    # Control-surface overlays.
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i, control in enumerate(controls):
        start = float(control["span"]["start"])
        end = float(control["span"]["end"])

        h0 = float(
            control["hinge"]["root_chord_fraction"]
        )
        h1 = float(
            control["hinge"]["tip_chord_fraction"]
        )

        eta = np.linspace(start, end, 80)
        hinge_fraction = np.linspace(h0, h1, 80)

        hinge = np.array([
            _wing_point(planform, sections, e, h)
            for e, h in zip(eta, hinge_fraction)
        ])

        trailing_control = np.array([
            _wing_point(planform, sections, e, 1.0)
            for e in eta
        ])

        region = np.vstack((
            hinge,
            trailing_control[::-1]
        ))

        color = colors[i % len(colors)]

        ax.fill(
            region[:, 1], region[:, 0],
            color=color, alpha=0.35,
            label=control["name"],
            zorder=2,
        )

        ax.plot(
            hinge[:, 1], hinge[:, 0],
            color=color, lw=2.0, zorder=3
        )

    # Draw outer wing boundary above mesh and controls.
    ax.plot(
        leading[:, 1], leading[:, 0],
        color="black", lw=1.6
    )

    ax.plot(
        trailing[:, 1], trailing[:, 0],
        color="black", lw=1.6
    )

    for eta in (0.0, 1.0):
        points = np.array([
            _wing_point(planform, sections, eta, xsi)
            for xsi in (0.0, 1.0)
        ])

        ax.plot(
            points[:, 1], points[:, 0],
            color="black", lw=1.6
        )

    ax.set_title("Right semi-wing | mesh and controls")
    ax.set_xlabel("Spanwise position Y [m]")
    ax.set_ylabel("Chordwise position X [m]")

    # Equal scale is important for planform inspection.
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    ax.grid(alpha=0.15)
    ax.legend(loc="best")

    fig.tight_layout()

    path = output_dir / "01_wing_mesh.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {path}")
    return path


# ============================================================
# 3. EXTRACT AVL STRIP DATA
# ============================================================

def _extract_strip_data(results, surface_name="wing"):
    """
    Extract one wing's strip data from AVLWrapper.

    Supports the legacy dictionary results structure and
    the newer strip_forces result interface.
    """

    try:
        case = results[1]
    except (KeyError, IndexError, TypeError):
        case = results["operating_point"]

    if isinstance(case, dict):
        strips = case.get("StripForces")
    else:
        strips = getattr(case, "strip_forces", None)

        if strips is None:
            strips = case["StripForces"]

    if strips is None:
        raise KeyError(
            "No StripForces output found. Enable AVL "
            "strip-force output in the solver session."
        )

    # Find the requested surface regardless of capitalization.
    for name, data in strips.items():
        if name.lower().strip() == surface_name.lower():
            return data

    raise KeyError(
        f"Surface '{surface_name}' not found in strip forces. "
        f"Available surfaces: {list(strips)}"
    )


def _strip_array(data, *keys):
    """Read a strip quantity using supported AVL key aliases."""

    for key in keys:
        if key in data:
            return np.asarray(data[key], dtype=float)

    return None


# ============================================================
# 4. SPANWISE AERODYNAMIC DISTRIBUTIONS
# ============================================================

def plot_distributions(
    results,
    flight_data,
    surface_name="wing",
    output_dir=OUTPUT_DIR,
):
    """
    Plot aerodynamic distributions for the right semi-wing.

    Quantities:
    - Local sectional lift coefficient cl
    - Lift per unit span [N/m]
    - Sectional induced drag coefficient, if available

    Uses actual AVL strip results.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _extract_strip_data(results, surface_name)

    # Common AVL strip-force field names.
    y = _strip_array(data, "Yle", "Y LE", "yle")
    chord = _strip_array(data, "Chord", "chord")
    cl = _strip_array(data, "cl", "CL")

    # AVL sectional induced drag is typically labeled cd.
    cdi = _strip_array(data, "cd", "CDi")

    if y is None or chord is None or cl is None:
        raise KeyError(
            "Missing required AVL strip fields.\n"
            f"Available fields: {list(data.keys())}"
        )

    flight = flight_data["flight_condition"]

    rho = float(flight["density"])
    velocity = float(flight["velocity"])
    q = 0.5 * rho * velocity**2

    # Keep the positive-y semi-wing only.
    mask = y >= -1e-10

    y = y[mask]
    chord = chord[mask]
    cl = cl[mask]

    if cdi is not None:
        cdi = cdi[mask]

    # Sort strips from root to tip.
    order = np.argsort(y)

    y = y[order]
    chord = chord[order]
    cl = cl[order]

    if cdi is not None:
        cdi = cdi[order]

    # Local sectional lift per unit span.
    # Assumes AVL cl is referenced to local strip chord.
    lift_per_span = q * chord * cl

    nplots = 3 if cdi is not None else 2

    fig, axes = plt.subplots(
        nplots, 1,
        figsize=(9, 3.1 * nplots),
        sharex=True,
    )

    # 1. Sectional lift coefficient.
    axes[0].plot(y, cl, lw=2)
    axes[0].set_ylabel(r"$c_l$")
    axes[0].set_title("Spanwise aerodynamic distributions")

    # 2. Lift distribution.
    axes[1].plot(y, lift_per_span, lw=2)
    axes[1].set_ylabel(r"$dL/dy$ [N/m]")

    # 3. Sectional induced drag coefficient.
    if cdi is not None:
        axes[2].plot(y, cdi, lw=2)
        axes[2].set_ylabel(r"$c_{d,i}$")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.axhline(0, lw=0.8, color="black", alpha=0.4)

    axes[-1].set_xlabel("Spanwise position Y [m]")

    fig.tight_layout()

    path = output_dir / "02_spanwise_distributions.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {path}")
    return path


# ============================================================
# 5. GENERATE ALL PLOTS
# ============================================================

def plot_all(
    geometry_data,
    mesh_data,
    controls_data,
    flight_data,
    results,
    output_dir=OUTPUT_DIR,
):
    """Generate the wing mesh and aerodynamic plots."""

    print("\nGenerating AVL plots...")

    mesh_path = plot_wing_mesh(
        geometry_data,
        mesh_data,
        controls_data,
        output_dir,
    )

    distribution_path = plot_distributions(
        results,
        flight_data,
        surface_name=geometry_data["surfaces"][0]["name"],
        output_dir=output_dir,
    )

    print("[PASS] Plot generation complete.")

    return mesh_path, distribution_path
