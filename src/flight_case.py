
"""
Construct an AVL operating case.

Supports:
- Fixed angle of attack and control deflections
- AVL constraints (CL, Cm, Cl, etc.)
- Dimensional Lift constraint [N], converted to CL

Note: Solving for lift does not imply pitching-moment trim.
"""

from avlwrapper import Case, Parameter

from .wing_geometry import calculate_reference


def build_case(flight_data, controls_data, geometry_data=None):
    """Build one AVL operating case, with optional trim."""

    flight = flight_data["flight_condition"]

    # --------------------------------------------------------
    # 1. FLIGHT CONDITIONS
    # --------------------------------------------------------

    controls = controls_data.get("control_surfaces", [])
    names = [control["name"] for control in controls]
    deflections = flight.get("controls", {})

    unknown = set(deflections) - set(names)

    if unknown:
        raise ValueError(f"Unknown controls: {unknown}")

    kwargs = {
        "alpha": float(flight.get("alpha", 0.0)),
        "beta": float(flight.get("beta", 0.0)),
        "velocity": float(flight["velocity"]),
        "density": float(flight["density"]),
    }

    velocity = kwargs["velocity"]
    density = kwargs["density"]

    if velocity <= 0 or density <= 0:
        raise ValueError("Velocity and density must be positive.")

    for name in names:
        kwargs[name] = float(deflections.get(name, 0.0))

    # --------------------------------------------------------
    # 2. OPTIONAL TRIM
    # --------------------------------------------------------

    trim = flight.get("trim", {})

    if trim.get("enabled", False):
        variable = trim["variable"]
        constraint = trim["constraint"]
        target = float(trim["target"])

        allowed = {"alpha"} | set(names)

        if variable not in allowed:
            raise ValueError(f"Invalid trim variable: {variable}")

        # ----------------------------------------------------
        # DIMENSIONAL LIFT CONSTRAINT
        #
        # AVL cannot directly constrain lift in Newtons.
        # Convert the requested lift to CL using the
        # reference area of the CURRENT wing geometry.
        # ----------------------------------------------------

        if constraint == "Lift":

            if variable != "alpha":
                raise ValueError(
                    "Lift trim currently requires variable: alpha."
                )

            if geometry_data is None:
                raise ValueError(
                    "geometry_data is required for Lift trim."
                )

            wing = geometry_data["surfaces"][0]

            ref = calculate_reference(
                wing["planform"],
                wing["sections"],
            )

            Sref = ref["Sref"]
            q = 0.5 * density * velocity**2

            target = target / (q * Sref)
            constraint = "CL"

        # ----------------------------------------------------
        # AVL PARAMETER
        # ----------------------------------------------------

        kwargs[variable] = Parameter(
            name=variable,
            constraint=constraint,
            value=target,
        )

    return Case(
        name="operating_point",
        **kwargs,
    )
