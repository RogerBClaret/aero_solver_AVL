
"""
Evaluate one candidate wing design using AVL.

Workflow
--------
1. Apply design variables to copies of the baseline inputs.
2. Build AVL geometry and flight case.
3. Execute AVL.
4. Extract standardized aerodynamic metrics.

The baseline input dictionaries are never modified.
"""

import numpy as np

from avlwrapper import Session

from ..avl_geometry import build_geometry
from ..flight_case import build_case

from .variables import apply_variables, get_bounds, denormalize
from .metrics import calculate_metrics


def evaluate_design(
    x,
    variables,
    baseline,
    normalized=True,
    include_spanloads=False,
):
    """
    Evaluate a candidate design using AVL.

    Parameters
    ----------
    x : array-like
        Candidate design vector.

    variables : list
        Active variables from design_variables.yaml.

    baseline : dict
        Original input configurations containing:
        geometry, mesh, controls, flight.

    normalized : bool
        If True, x contains normalized variables in [0,1].
        Otherwise, x contains dimensional values.

    include_spanloads : bool
        Whether to calculate the root bending moment.

    Returns
    -------
    dict
        Candidate values, standardized metrics and status.
    """

    # --------------------------------------------------------
    # 1. DESIGN VECTOR
    # --------------------------------------------------------

    x = np.asarray(x, dtype=float)

    bounds = get_bounds(variables)

    if x.shape != (len(variables),):
        raise ValueError("Invalid design-vector dimensions.")

    if not np.isfinite(x).all():
        raise ValueError("Design vector contains nonfinite values.")

    # Convert normalized coordinates to physical values.
    x_physical = (
        denormalize(x, bounds)
        if normalized else x.copy()
    )

    # Bounds are checked rather than silently clipping.
    tolerance = 1e-10 * np.maximum(
        1.0, bounds[:, 1] - bounds[:, 0]
    )

    if np.any(x_physical < bounds[:, 0] - tolerance) or np.any(
        x_physical > bounds[:, 1] + tolerance
    ):
        raise ValueError("Candidate outside design-variable bounds.")

    # --------------------------------------------------------
    # 2. BUILD CANDIDATE INPUTS
    # --------------------------------------------------------

    # apply_variables() performs a deep copy.
    candidate = apply_variables(
        variables,
        x_physical,
        baseline,
    )

    geometry_data = candidate["geometry"]
    mesh_data = candidate["mesh"]
    controls_data = candidate["controls"]
    flight_data = candidate["flight"]

    # Prevent conflicts between prescribed and trimmed variables.
    trim = flight_data["flight_condition"].get("trim", {})

    if trim.get("enabled", False):
        trimmed_variable = trim["variable"]

        active_names = [
            var["name"] for var in variables
        ]

        if trimmed_variable == "alpha" and "alpha" in active_names:
            raise ValueError(
                "Alpha cannot be optimized while AVL trims alpha."
            )

        if trimmed_variable in (
            flight_data["flight_condition"].get("controls", {})
        ):
            for var in variables:
                if (
                    var["name"] == "control_deflection"
                    and var.get("control") == trimmed_variable
                ):
                    raise ValueError(
                        f"Control '{trimmed_variable}' cannot be "
                        "optimized while AVL trims it."
                    )

    # --------------------------------------------------------
    # 3. BUILD AVL MODEL
    # --------------------------------------------------------

    geometry = build_geometry(
        geometry_data,
        mesh_data,
        controls_data,
    )

    # Pass CURRENT candidate geometry for dimensional lift trim.
    case = build_case(
        flight_data,
        controls_data,
        geometry_data,
    )

    # --------------------------------------------------------
    # 4. EXECUTE AVL
    # --------------------------------------------------------

    session = Session(
        geometry=geometry,
        cases=[case],
    )

    results = session.get_results()

    # --------------------------------------------------------
    # 5. EXTRACT METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        results,
        flight_data,
        geometry_data,
        include_spanloads=include_spanloads,
    )

    # --------------------------------------------------------
    # 6. RETURN EVALUATION
    # --------------------------------------------------------

    return {
        "success": True,
        "x_normalized": (
            x.copy()
            if normalized
            else (x_physical - bounds[:, 0]) /
                 (bounds[:, 1] - bounds[:, 0])
        ),
        "x_physical": x_physical,
        "metrics": metrics,
    }
