
"""
Extract standardized optimization metrics from AVL results.

All metrics use SI units, except aerodynamic coefficients
and angles (degrees).

This module does not depend on the active objective,
constraints, or optimization algorithm.
"""

from math import isfinite

import numpy as np


# ============================================================
# 1. ACCESS AVL RESULTS
# ============================================================

def _get_case(results):
    """Extract the operating case from AVLWrapper results."""

    try:
        return results[1]
    except (KeyError, IndexError, TypeError):
        return results["operating_point"]


def _get_output(case, name):
    """Support dictionary and object-based AVL results."""

    if isinstance(case, dict):
        return case.get(name)

    attribute = {
        "Totals": "totals",
        "StripForces": "strip_forces",
    }.get(name)

    if attribute is not None:
        output = getattr(case, attribute, None)

        if output is not None:
            return output

    try:
        return case[name]
    except (KeyError, TypeError, AttributeError):
        return None


def _number(data, key):
    """Read and validate a scalar AVL result."""

    if key not in data:
        raise KeyError(f"Missing AVL result: {key}")

    value = float(data[key])

    if not isfinite(value):
        raise ValueError(f"Nonfinite AVL result: {key}")

    return value


# ============================================================
# 2. SPANWISE ROOT BENDING MOMENT
# ============================================================

def calculate_root_bending(strip_forces, q, surface_name):
    """
    Estimate lift-induced root bending moment for one wing.

    M_root = integral(y * dL)

    Uses AVL strip locations, chord, lift coefficient,
    and strip width (or strip area).

    Only positive-Y strips are included.

    This is an aerodynamic lift-load estimate, not a
    complete structural bending moment calculation.
    """

    if strip_forces is None:
        raise KeyError("AVL StripForces output is unavailable.")

    # Select the requested aerodynamic surface.
    matching = [
        data
        for name, data in strip_forces.items()
        if str(name).strip().lower() == surface_name.lower()
    ]

    if len(matching) != 1:
        raise KeyError(
            f"Cannot uniquely identify '{surface_name}' "
            f"in AVL strip forces."
        )

    strips = matching[0]

    def array(*names):
        for name in names:
            if name in strips:
                return np.asarray(strips[name], dtype=float)
        return None

    y = array("Yle", "Y LE")
    chord = array("Chord", "chord")
    cl = array("cl")

    width = array("Width", "width")
    area = array("Area", "area")

    if y is None or cl is None:
        raise KeyError(
            "Strip data must contain Yle and cl. "
            f"Available keys: {list(strips)}"
        )

    if area is None and (width is None or chord is None):
        raise KeyError(
            "Strip data require Area or both Width and Chord. "
            f"Available keys: {list(strips)}"
        )

    # AVL may provide positive and negative semi-wing strips.
    # Integrate only the right semi-wing.
    mask = y >= 0.0

    y = y[mask]
    cl = cl[mask]

    if area is not None:
        strip_area = area[mask]
    else:
        strip_area = width[mask] * chord[mask]

    if not (
        len(y) == len(cl) == len(strip_area)
        and len(y) > 0
    ):
        raise ValueError("Inconsistent or empty AVL strip data.")

    if not all(
        np.isfinite(a).all()
        for a in (y, cl, strip_area)
    ):
        raise ValueError("Nonfinite AVL strip data.")

    if np.any(strip_area <= 0):
        raise ValueError("Strip areas must be positive.")

    # Approximate lift contributed by each spanwise strip.
    strip_lift = q * strip_area * cl

    # Magnitude of the right-wing lift-induced root moment.
    return float(abs(np.sum(y * strip_lift)))


# ============================================================
# 3. CALCULATE ALL OPTIMIZATION METRICS
# ============================================================

def calculate_metrics(
    results,
    flight_data,
    geometry_data,
    include_spanloads=True,
):
    """
    Calculate standardized metrics for one AVL analysis.

    Parameters
    ----------
    results : AVLWrapper results
        Output from session.get_results().

    flight_data : dict
        Current flight-condition configuration.

    geometry_data : dict
        Current aircraft geometry configuration.

    include_spanloads : bool
        If True, calculate the root bending moment
        from AVL strip forces.

    Returns
    -------
    dict
        Standardized metrics for objectives and constraints.
    """

    case = _get_case(results)
    totals = _get_output(case, "Totals")

    if totals is None:
        raise KeyError("AVL Totals output is unavailable.")

    flight = flight_data["flight_condition"]

    velocity = float(flight["velocity"])
    density = float(flight["density"])

    if velocity <= 0 or density <= 0:
        raise ValueError("Invalid flight condition.")

    q = 0.5 * density * velocity**2

    # --------------------------------------------------------
    # GEOMETRIC REFERENCE QUANTITIES
    # --------------------------------------------------------

    Sref = _number(totals, "Sref")
    Bref = _number(totals, "Bref")
    Cref = _number(totals, "Cref")

    if min(Sref, Bref, Cref) <= 0:
        raise ValueError("Invalid aerodynamic references.")

    wing = geometry_data["surfaces"][0]
    sections = wing["sections"]

    root = min(sections, key=lambda s: float(s["y"]))
    tip = max(sections, key=lambda s: float(s["y"]))

    taper_ratio = (
        float(tip["chord"]) / float(root["chord"])
    )

    # --------------------------------------------------------
    # AERODYNAMIC COEFFICIENTS
    # --------------------------------------------------------

    CL = _number(totals, "CLtot")
    CD = _number(totals, "CDtot")
    CDi = _number(totals, "CDind")

    Cm = _number(totals, "Cmtot")
    Cl = _number(totals, "Cltot")
    Cn = _number(totals, "Cntot")

    e = _number(totals, "e")

    # This is the final AoA, including AVL lift trim.
    alpha = _number(totals, "Alpha")

    # A zero drag coefficient makes L/D undefined.
    if CD == 0:
        raise ValueError("Cannot calculate L/D with CD=0.")

    # --------------------------------------------------------
    # DIMENSIONAL FORCES
    # --------------------------------------------------------

    lift = q * Sref * CL
    drag = q * Sref * CD
    induced_drag = q * Sref * CDi

    # --------------------------------------------------------
    # STANDARDIZED METRICS
    # --------------------------------------------------------

    metrics = {
        # Geometry
        "Sref": Sref,
        "Bref": Bref,
        "Cref": Cref,
        "AR": Bref**2 / Sref,
        "taper_ratio": taper_ratio,

        # Aerodynamics
        "CL": CL,
        "CD": CD,
        "CDi": CDi,
        "Cm": Cm,
        "Cl": Cl,
        "Cn": Cn,
        "e": e,
        "LD": CL / CD,
        "alpha_cruise": alpha,

        # Dimensional forces
        "lift": lift,
        "drag": drag,
        "induced_drag": induced_drag,
    }

    # --------------------------------------------------------
    # OPTIONAL SPANWISE LOAD METRIC
    # --------------------------------------------------------

    if include_spanloads:
        strips = _get_output(case, "StripForces")

        metrics["root_bending_moment"] = (
            calculate_root_bending(
                strips,
                q,
                surface_name=wing["name"],
            )
        )

    return metrics







