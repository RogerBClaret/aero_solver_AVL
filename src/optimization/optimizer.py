
"""Optimization algorithm dispatcher for the HALF-WING AVL model.

Uses one AVL operating point per candidate design.
Objectives and constraints come from Evaluator.
"""

from pathlib import Path

import numpy as np
import yaml

from scipy.optimize import (
    minimize,
    differential_evolution,
    NonlinearConstraint,
)

from src.optimization.variables import (
    initial_vector,
    variable_bounds,
)


SUPPORTED = {
    "SLSQP",
    "COBYLA",
    "COBYQA",
    "trust-constr",
    "differential_evolution",
}


def load_algorithm(path="input/optimization_algorithm.yaml"):
    """Read algorithm settings from YAML."""

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    settings = config["optimizer"]

    if settings["algorithm"] not in SUPPORTED:
        raise ValueError(
            f"Unsupported algorithm: {settings['algorithm']}. "
            f"Choose from {sorted(SUPPORTED)}"
        )

    return settings


def _constraints_for_scipy(evaluator, safe_residuals):
    constraints = []

    for i, item in enumerate(evaluator.constraints):
        lower = 0.0
        upper = 0.0 if item["operator"] == "==" else np.inf

        constraints.append(
            NonlinearConstraint(
                lambda x, i=i: safe_residuals(x)[i],
                lb=lower,
                ub=upper,
            )
        )

    return constraints


def run_optimization(
    evaluator,
    variables,
    settings=None,
):
    """Run the selected SciPy optimization algorithm.

    Parameters
    ----------
    evaluator:
        Existing Evaluator instance.

    variables:
        Active variables from load_variables().

    settings:
        Optimizer settings dictionary. If None, read YAML.

    Returns
    -------
    scipy.optimize.OptimizeResult
    """

    if settings is None:
        settings = load_algorithm()

    algorithm = settings["algorithm"]

    if algorithm not in SUPPORTED:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if not variables:
        raise ValueError("No active optimization variables")

    x0 = initial_vector(variables)
    bounds = variable_bounds(variables)
    # Keep evaluated candidates within variable bounds.
    # SciPy may request points slightly outside the bounds.
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    
    def safe_x(x):
        return np.clip(np.asarray(x, dtype=float), lower, upper)


    # Penalize candidate geometries that AVL cannot evaluate.
    # This does NOT modify AVLWrapper.

    failed_candidates = set()

    def safe_evaluate(x):
        x = safe_x(x)
        key = tuple(x)

        try:
            return evaluator.evaluate(x)

        except (FileNotFoundError, ValueError) as exc:
            if key not in failed_candidates:
                failed_candidates.add(key)
                print(
                    f"\n[AVL FAILED] Candidate {np.round(x, 5)}"
                    f"\nReason: {exc}"
                )

            return None


    def safe_objective(x):
        result = safe_evaluate(x)

        if result is None:
            return 1e6  # Bad candidate: large objective penalty

        return result["objective"]


    def safe_residuals(x):
        result = safe_evaluate(x)

        if result is None:
            # Negative residual = constraint violation
            return -1e3 * np.ones(len(evaluator.constraints))

        return result["residuals"]

        
    maxiter = int(settings.get("max_iterations", 100))
    tol = float(settings.get("tolerance", 1e-5))
    display = bool(settings.get("display", True))

    if maxiter < 1 or tol <= 0:
        raise ValueError("Invalid optimizer settings")

    print(f"\nOPTIMIZER: {algorithm}")
    print(f"Active variables: {len(variables)}")
    print(f"Maximum iterations: {maxiter}")

    # Evaluate the baseline once.
    baseline = evaluator.evaluate(x0)

    print(f"Initial objective: {baseline['objective']:.8f}")
    print(f"Initial feasible: {baseline['feasible']}")

    # The evaluator already caches the last design, avoiding
    # duplicate AVL runs for objective/constraint requests.
    if algorithm == "differential_evolution":

        result = differential_evolution(
            func=safe_objective,
            bounds=bounds,
            constraints=_constraints_for_scipy(evaluator, safe_residuals),
            maxiter=maxiter,
            tol=tol,
            popsize=int(settings.get("population_size", 8)),
            mutation=settings.get("mutation", [0.5, 1.0]),
            recombination=float(
                settings.get("recombination", 0.7)
            ),
            seed=int(settings.get("seed", 42)),
            polish=bool(settings.get("polish", False)),
            disp=display,
            workers=1,
        )

    else:
        # Different SciPy algorithms require different options.
        options = {"disp": display}

        if algorithm == "SLSQP":
            options.update(
                maxiter=maxiter,
                ftol=tol,
            )

        elif algorithm == "COBYLA":
            options.update(
                maxiter=maxiter,
                tol=tol,
                rhobeg=0.1,
            )

        elif algorithm == "COBYQA":
            options.update(
                maxfev=maxiter,
                final_tr_radius=tol,
            )

        elif algorithm == "trust-constr":
            options.update(
                maxiter=maxiter,
                gtol=tol,
            )

        if algorithm == "SLSQP":
            constraints = [
                {
                    "type": (
                        "eq" if item["operator"] == "==" else "ineq"
                    ),
                    "fun": lambda x, i=i: safe_residuals(x)[i],
                }
                for i, item in enumerate(evaluator.constraints)
            ]
        else:
            constraints = _constraints_for_scipy(
                evaluator, safe_residuals
            )

        result = minimize(
            fun=safe_objective,
            x0=x0,
            method=algorithm,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

    # Re-evaluate the returned design to obtain its actual
    # objectives, constraint values, and AVL outputs.
    result.x = safe_x(result.x)
    final = safe_evaluate(result.x)

    if final is None:
        raise RuntimeError(
            "Optimizer returned an AVL-invalid design. "
            "No valid final result is available."
        )

    # Attach project-specific data to SciPy's result.
    result["evaluation"] = final
    result["feasible"] = final["feasible"]
    result["avl_evaluations"] = evaluator.n_evaluations

    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Algorithm: {algorithm}")
    print(f"Solver success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final objective: {final['objective']:.8f}")
    print(f"Feasible: {final['feasible']}")
    print(f"AVL evaluations: {evaluator.n_evaluations}")

    print("\nOptimized variables:")
    for var, value in zip(variables, result.x):
        print(f"  {var['name']}: {value:.6f}")

    print("\nConstraints:")
    for item in final["constraints"]:
        status = "PASS" if item["satisfied"] else "FAIL"

        print(
            f"  [{status}] {item['metric']} "
            f"{item['operator']} {item['limit']}"
            f" | actual={item['value']:.6f}"
        )

    return result
