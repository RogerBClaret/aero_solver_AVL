
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


# ============================================================
# 2. PLOT WING PLANFORM AND CONTROL SURFACES
# ============================================================


def plot_wing_planform(
    geometry_data,
    controls_data,
    output_dir=OUTPUT_DIR,
):
    """
    Plot the right semi-wing planform and control surfaces.

    Shows:
    - Leading and trailing edges
    - Flap and aileron regions
    - Control-surface hinge lines

    No mesh or wing-section lines are drawn.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wing = geometry_data["surfaces"][0]
    planform = wing["planform"]
    sections = validate_sections(wing["sections"])
    controls = controls_data.get("control_surfaces", [])

    fig, ax = plt.subplots(figsize=(11, 5))

    # --------------------------------------------------------
    # 1. Wing outline
    # --------------------------------------------------------
    eta = np.linspace(0.0, 1.0, 300)

    leading = np.array([
        _wing_point(planform, sections, e, 0.0)
        for e in eta
    ])

    trailing = np.array([
        _wing_point(planform, sections, e, 1.0)
        for e in eta
    ])

    outline = np.vstack((leading, trailing[::-1]))

    ax.fill(
        outline[:, 1], outline[:, 0],
        color="0.90", zorder=1
    )

    # --------------------------------------------------------
    # 2. Control surfaces
    # --------------------------------------------------------
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

        eta_ctrl = np.linspace(start, end, 100)
        hinge_fraction = np.linspace(h0, h1, 100)

        hinge = np.array([
            _wing_point(planform, sections, e, h)
            for e, h in zip(eta_ctrl, hinge_fraction)
        ])

        trailing_ctrl = np.array([
            _wing_point(planform, sections, e, 1.0)
            for e in eta_ctrl
        ])

        region = np.vstack((
            hinge,
            trailing_ctrl[::-1]
        ))

        color = colors[i % len(colors)]

        ax.fill(
            region[:, 1], region[:, 0],
            color=color,
            alpha=0.45,
            label=control["name"],
            zorder=2,
        )

        ax.plot(
            hinge[:, 1], hinge[:, 0],
            color=color,
            linewidth=1.8,
            zorder=3,
        )

    # --------------------------------------------------------
    # 3. External wing boundaries only
    # --------------------------------------------------------
    closed = np.vstack((outline, outline[0]))

    ax.plot(
        closed[:, 1], closed[:, 0],
        color="black",
        linewidth=1.5,
        zorder=4,
    )

    # --------------------------------------------------------
    # 4. Plot formatting
    # --------------------------------------------------------
    ax.set_title("Right semi-wing | planform and controls")
    ax.set_xlabel("Spanwise position Y [m]")
    ax.set_ylabel("Chordwise position X [m]")

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.grid(alpha=0.15)

    if controls:
        ax.legend(loc="best")

    fig.tight_layout()

    path = output_dir / "01_wing_planform.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {path}")
    return path


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
# 4.1 EXTRACT AVL SPANWISE STRIP DATA
# ============================================================

def _extract_strip_data(results, surface_name="wing"):
    """
    Extract spanwise strip-force results for one AVL surface.

    Handles the dictionary-based results returned by AVLWrapper.
    """

    # Retrieve the operating case.
    try:
        case = results[1]
    except (KeyError, IndexError, TypeError):
        case = results["operating_point"]

    # Extract strip-force results.
    if isinstance(case, dict):
        strips = case.get("StripForces")
    else:
        strips = getattr(case, "strip_forces", None)

    if strips is None:
        raise KeyError(
            "StripForces not found in AVL results. "
            f"Available result keys: {list(case.keys())}"
        )

    # Find the requested wing surface.
    for name, data in strips.items():
        if name.strip().lower() == surface_name.strip().lower():
            return data

    raise KeyError(
        f"Surface '{surface_name}' not found. "
        f"Available surfaces: {list(strips.keys())}"
    )


def _strip_array(data, *keys):
    """Extract an aerodynamic quantity using possible key names."""

    for key in keys:
        if key in data:
            return np.asarray(data[key], dtype=float)

    return None


# ============================================================
# 5. GENERATE ALL PLOTS
# ============================================================

def plot_all(
    geometry_data,
    controls_data,
    flight_data,
    results,
    output_dir=OUTPUT_DIR,
):
    """Generate the wing mesh and aerodynamic plots."""

    print("\nGenerating AVL plots...")

    planform_path = plot_wing_planform(
        geometry_data,
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

    return planform_path, distribution_path
