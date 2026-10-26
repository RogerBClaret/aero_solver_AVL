
"""Registry of available optimization metrics."""

import math


def wing_sections(geometry):
    return sorted(
        geometry["surfaces"][0]["sections"],
        key=lambda s: s["y"]
    )


# GEOMETRY METRICS
# Full-wing equivalent geometry, calculated from half-wing input.

def half_span(geometry):
    return wing_sections(geometry)[-1]["y"]


def full_span(geometry):
    return 2 * half_span(geometry)


def half_area(geometry):
    sections = wing_sections(geometry)

    return sum(
        0.5 * (a["chord"] + b["chord"]) *
        (b["y"] - a["y"])
        for a, b in zip(sections[:-1], sections[1:])
    )


def full_area(geometry):
    return 2 * half_area(geometry)


def aspect_ratio(geometry):
    return full_span(geometry)**2 / full_area(geometry)


def taper_ratio(geometry):
    sections = wing_sections(geometry)
    return sections[-1]["chord"] / sections[0]["chord"]


def sweep_LE(geometry):
    sections = wing_sections(geometry)
    root, tip = sections[0], sections[-1]

    return math.degrees(math.atan2(
        tip["x_le"] - root["x_le"],
        tip["y"] - root["y"]
    ))


def dihedral(geometry):
    sections = wing_sections(geometry)
    root, tip = sections[0], sections[-1]

    return math.degrees(math.atan2(
        tip["z_le"] - root["z_le"],
        tip["y"] - root["y"]
    ))


def washout(geometry):
    sections = wing_sections(geometry)
    return sections[0]["twist"] - sections[-1]["twist"]


GEOMETRY_METRICS = {
    "half_span": half_span,
    "full_span": full_span,
    "half_area": half_area,
    "full_area": full_area,
    "aspect_ratio": aspect_ratio,
    "taper_ratio": taper_ratio,
    "sweep_LE": sweep_LE,
    "dihedral": dihedral,
    "washout": washout,
}


# AERODYNAMIC METRICS
# Coefficients from the single isolated half-wing AVL calculation.

AERO_METRICS = {
    "CL": "CLtot",
    "CDi": "CDind",
    "Cm": "Cmtot",
    "Cl": "Cltot",
    "Cn": "Cntot",
}

CONTROL_COEFFICIENTS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")


def evaluate_metric(name, geometry, results=None, flight=None):
    """Evaluate any registered geometry or aerodynamic metric."""

    if name in ("Lift_N", "InducedDrag_N", "dynamic_pressure"):
        if flight is None:
            raise ValueError(f"{name} requires flight data")

        if name == "dynamic_pressure":
            return dynamic_pressure(flight)

        if results is None:
            raise ValueError(f"{name} requires AVL results")

        if name == "Lift_N":
            return lift_N(geometry, results, flight)

        if name == "InducedDrag_N":
            return induced_drag_N(geometry, results, flight)
        
    if name in GEOMETRY_METRICS:
        return float(GEOMETRY_METRICS[name](geometry))

    if results is None:
        raise ValueError(f"{name} requires AVL results")

    # Accept the single-case results dictionary.
    case = results.get("operating_point", results)

    if name in AERO_METRICS:
        return float(
            case["Totals"][AERO_METRICS[name]]
        )

    if name == "CL_CDi":
        totals = case["Totals"]
        return (
            float(totals["CLtot"]) /
            float(totals["CDind"])
        )

    # Control derivatives: dCl_ddeg.aileron
    if "." in name:
        metric, control = name.split(".", 1)

        if metric.startswith("d") and metric.endswith("_ddeg"):
            coefficient = metric[1:-5]

            if coefficient not in CONTROL_COEFFICIENTS:
                raise KeyError(f"Unknown derivative: {metric}")

            key = f"{coefficient}{control}"

            return float(
                case["StabilityDerivatives"][key]
            )

    raise KeyError(f"Unknown metric: {name}")


def available_metrics(controls=None):
    """List all metric names usable for optimization."""

    names = (
        list(GEOMETRY_METRICS)
        + list(AERO_METRICS)
        + [
            "CL_CDi",
            "dynamic_pressure",
            "Lift_N",
            "InducedDrag_N",
        ]
    )

    if controls is not None:
        for control in controls["control_surfaces"]:
            for coefficient in CONTROL_COEFFICIENTS:
                names.append(
                    f"d{coefficient}_ddeg.{control['name']}"
                )

    return names




def dynamic_pressure(flight):
    """Dynamic pressure [Pa]."""
    condition = flight["flight_condition"]
    rho = float(condition["density"])
    V = float(condition["velocity"])

    return 0.5 * rho * V**2


def lift_N(geometry, results, flight):
    """Lift [N] from the modeled HALF-WING."""
    q = dynamic_pressure(flight)
    S = half_area(geometry)
    CL = evaluate_metric("CL", geometry, results)

    return q * S * CL


def induced_drag_N(geometry, results, flight):
    """Induced drag [N] from the modeled HALF-WING."""
    q = dynamic_pressure(flight)
    S = half_area(geometry)
    CDi = evaluate_metric("CDi", geometry, results)

    return q * S * CDi
