
"""
Objective-function management for AVL optimization.

Responsibilities
----------------
1. Load the active objective from objectives.yaml.
2. Validate the objective name, metric and direction.
3. Extract the requested metric from evaluation results.
4. Convert maximization to minimization for SciPy.

No objective metrics or numerical values are hardcoded.
The evaluator is responsible for calculating the metrics.
"""

from pathlib import Path
from math import isfinite

import yaml


# ============================================================
# 1. LOAD OBJECTIVE CONFIGURATION
# ============================================================

def load_objective(path):
    """
    Load exactly one active objective from YAML.

    Parameters
    ----------
    path : str or Path
        Path to objectives.yaml.

    Returns
    -------
    dict
        Active objective definition.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid objectives YAML file.")

    objectives = config.get("objective")

    if not isinstance(objectives, list) or len(objectives) != 1:
        raise ValueError(
            "Exactly one active objective must be defined."
        )

    objective = objectives[0]

    if not isinstance(objective, dict):
        raise ValueError("Objective must be a dictionary.")

    # Required configuration fields.
    required = {"name", "metric", "direction"}

    missing = required - set(objective)

    if missing:
        raise ValueError(
            f"Missing objective fields: {sorted(missing)}"
        )

    if not isinstance(objective["name"], str) or not objective["name"].strip():
        raise ValueError("Objective name must be a nonempty string.")

    if not isinstance(objective["metric"], str) or not objective["metric"].strip():
        raise ValueError("Objective metric must be a nonempty string.")

    if objective["direction"] not in ("minimize", "maximize"):
        raise ValueError(
            "Objective direction must be minimize or maximize."
        )

    return objective


# ============================================================
# 2. EVALUATE OBJECTIVE
# ============================================================

def evaluate_objective(objective, metrics):
    """
    Evaluate the configured objective using calculated metrics.

    Parameters
    ----------
    objective : dict
        Active objective configuration.

    metrics : dict
        Standardized metrics from the design evaluator.

    Returns
    -------
    float
        Signed objective value to MINIMIZE with SciPy.

    Notes
    -----
    For minimization: f = metric
    For maximization: f = -metric
    """

    metric_name = objective["metric"]

    if metric_name not in metrics:
        raise KeyError(
            f"Objective metric '{metric_name}' not available. "
            f"Available metrics: {sorted(metrics)}"
        )

    value = float(metrics[metric_name])

    if not isfinite(value):
        raise ValueError(
            f"Objective metric '{metric_name}' is not finite."
        )

    direction = objective["direction"]

    if direction == "minimize":
        return value

    if direction == "maximize":
        return -value

    raise ValueError(f"Invalid objective direction: {direction}")


# ============================================================
# 3. ORIGINAL OBJECTIVE VALUE
# ============================================================

def objective_value(objective, metrics):
    """
    Return the physical objective value without sign changes.

    Useful for optimization histories, reports and plots.
    """

    metric_name = objective["metric"]

    if metric_name not in metrics:
        raise KeyError(
            f"Objective metric '{metric_name}' not available."
        )

    value = float(metrics[metric_name])

    if not isfinite(value):
        raise ValueError(
            f"Objective metric '{metric_name}' is not finite."
        )

    return value
