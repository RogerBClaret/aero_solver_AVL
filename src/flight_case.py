
"""
Construct an AVL operating case.

Supports:
- Fixed angle of attack
- Fixed control deflections
- Optional single-variable AVL constraint

Note: CL targeting is not equivalent to complete aircraft trim.
"""

from avlwrapper import Case, Parameter


def build_case(flight_data, controls_data):
    """Build a single AVL operating condition."""

    flight = flight_data["flight_condition"]

    # Names must match the control-surface definitions.
    controls = controls_data.get("control_surfaces", [])
    names = [control["name"] for control in controls]

    deflections = flight.get("controls", {})

    unknown = set(deflections) - set(names)

    if unknown:
        raise ValueError(f"Unknown controls: {unknown}")

    # Freestream conditions and fixed angle of attack.
    kwargs = {
        "alpha": float(flight.get("alpha", 0.0)),
        "beta": float(flight.get("beta", 0.0)),
        "velocity": float(flight["velocity"]),
        "density": float(flight["density"]),
    }

    if kwargs["velocity"] <= 0 or kwargs["density"] <= 0:
        raise ValueError("Velocity and density must be positive.")

    # Control deflections (default: neutral).
    for name in names:
        kwargs[name] = float(deflections.get(name, 0.0))

    # Optional AVL constraint.
    trim = flight.get("trim", {})

    if trim.get("enabled", False):
        variable = trim["variable"]
        constraint = trim["constraint"]
        target = float(trim["target"])

        allowed = {"alpha"} | set(names)

        if variable not in allowed:
            raise ValueError(f"Invalid trim variable: {variable}")

        kwargs[variable] = Parameter(
            name=variable,
            constraint=constraint,
            value=target,
        )

    return Case(
        name="operating_point",
        **kwargs,
    )
