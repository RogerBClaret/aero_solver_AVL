
"""
Constraint management for AVL optimization.

Responsibilities
----------------
1. Load and validate active constraints from YAML.
2. Evaluate constraints using standardized metrics.
3. Normalize residuals for numerical conditioning.
4. Convert constraints to forms suitable for SciPy.

No constraint names or physical limits are hardcoded.
"""

from pathlib import Path
from math import isfinite

import numpy as np
import yaml


# ============================================================
# 1. LOAD CONSTRAINTS
# ============================================================

def load_constraints(path):
    """
    Load active optimization constraints.

    An empty list is allowed (unconstrained optimization).
    """

    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid constraints YAML file.")

    constraints = config.get("constraints", [])

    if constraints is None:
        constraints = []

    if not isinstance(constraints, list):
        raise ValueError("Constraints must be a list.")

    names = set()

    for constraint in constraints:
        if not isinstance(constraint, dict):
            raise ValueError("Invalid constraint definition.")

        required = {"name", "metric", "operator", "value"}
        missing = required - set(constraint)

        if missing:
            raise ValueError(
                f"Missing constraint fields: {sorted(missing)}"
            )

        name = constraint["name"]
        metric = constraint["metric"]
        operator = constraint["operator"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Constraint name must be nonempty.")

        if not isinstance(metric, str) or not metric.strip():
            raise ValueError(f"Invalid metric for {name}.")

        if name in names:
            raise ValueError(f"Duplicate constraint: {name}")

        names.add(name)

        if operator not in ("<=", ">=", "=="):
            raise ValueError(
                f"Invalid operator '{operator}' for {name}."
            )

        value = float(constraint["value"])

        if not isfinite(value):
            raise ValueError(f"Nonfinite target for {name}.")

        # Store the validated numerical value.
        constraint["value"] = value

    return constraints


# ============================================================
# 2. EVALUATE ONE CONSTRAINT
# ============================================================

def evaluate_constraint(constraint, metrics):
    """
    Return the normalized constraint residual.

    Conventions
    -----------
    <= : (target - metric) / scale >= 0
    >= : (metric - target) / scale >= 0
    == : (metric - target) / scale == 0

    The scale uses the target magnitude by default.
    An optional positive 'scale' can override it.
    """

    name = constraint["name"]
    metric_name = constraint["metric"]

    if metric_name not in metrics:
        raise KeyError(
            f"Metric '{metric_name}' required by '{name}' "
            f"is unavailable."
        )

    actual = float(metrics[metric_name])
    target = float(constraint["value"])

    if not isfinite(actual):
        raise ValueError(
            f"Nonfinite metric '{metric_name}' for '{name}'."
        )

    scale = float(
        constraint.get("scale", max(abs(target), 1.0))
    )

    if not isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid scale for '{name}'.")

    operator = constraint["operator"]

    if operator == "<=":
        return (target - actual) / scale

    if operator == ">=":
        return (actual - target) / scale

    if operator == "==":
        return (actual - target) / scale

    raise ValueError(f"Unknown operator: {operator}")


# ============================================================
# 3. EVALUATE ALL CONSTRAINTS
# ============================================================

def evaluate_constraints(constraints, metrics):
    """Return a dictionary of normalized residuals."""

    return {
        c["name"]: evaluate_constraint(c, metrics)
        for c in constraints
    }


# ============================================================
# 4. CHECK FEASIBILITY
# ============================================================

def check_feasibility(
    constraints,
    metrics,
    inequality_tol=1e-6,
    equality_tol=1e-4,
):
    """
    Check whether all constraints are satisfied.

    Tolerances apply to normalized residuals.

    Returns
    -------
    bool
        True if all constraints are satisfied.
    """

    for constraint in constraints:
        residual = evaluate_constraint(constraint, metrics)

        if constraint["operator"] == "==":
            if abs(residual) > equality_tol:
                return False

        else:
            if residual < -inequality_tol:
                return False

    return True


# ============================================================
# 5. SCIPY CONSTRAINT CONVERSION
# ============================================================

def scipy_constraint_bounds(constraints, equality_tol=1e-4):
    """
    Return lower/upper residual bounds for SciPy.

    Suitable for scipy.optimize.NonlinearConstraint:

    Inequality:  0 <= g(x) <= infinity
    Equality:    0 == g(x) == 0

    COBYLA cannot enforce exact equality directly.
    Its adapter should use a tolerance band instead.

    The evaluator callback is added later in problem.py.
    """

    lower = []
    upper = []

    for constraint in constraints:
        lower.append(0.0)

        if constraint["operator"] == "==":
            # Represent equality using a tolerance band.
            lower[-1] = -equality_tol
            upper.append(equality_tol)
        else:
            upper.append(np.inf)

    return np.array(lower), np.array(upper)




