
"""
Run AVL optimization using COBYLA or Differential Evolution.

The optimization problem provides:
- Normalized design variables
- Objective function
- Constraint residuals
- Cached AVL evaluations

This module is independent of aircraft geometry and physics.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
import cma

from scipy.optimize import OptimizeResult

from scipy.optimize import (
    minimize,
    differential_evolution,
    NonlinearConstraint,
)

from ..input_loader import ROOT
from .constraints import check_feasibility


# ============================================================
# 1. LOAD OPTIMIZATION ALGORITHM
# ============================================================

def load_algorithm(path=None):
    """Read and validate the selected optimization algorithm."""

    path = (
        Path(path)
        if path is not None
        else ROOT / "input" / "optimization" / "algorithm.yaml"
    )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid algorithm configuration.")

    settings = config["algorithm"]

    if not isinstance(settings, dict):
        raise ValueError("Algorithm settings must be a dictionary.")

    method = settings["method"].upper().replace("-", "_")

    if method not in ("COBYLA", "DE", "CMA_ES"):
        raise ValueError(f"Unsupported algorithm: {method}")

    return {**settings, "method": method}

# ============================================================
# 2. CONSTRUCT OPTIMIZER CONSTRAINTS
# ============================================================

def _cobyla_constraints(problem, equality_tol):
    """
    Convert constraints to COBYLA inequalities g(x) >= 0.

    Equalities are represented by:
        -tol <= residual <= tol

    Variable bounds are explicitly enforced in [0,1].
    """

    constraints = []

    for i, constraint in enumerate(problem.constraints):

        if constraint["operator"] == "==":
            constraints.extend([
                {
                    "type": "ineq",
                    "fun": lambda x, j=i: (
                        equality_tol - problem.constraint_values(x)[j]
                    ),
                },
                {
                    "type": "ineq",
                    "fun": lambda x, j=i: (
                        equality_tol + problem.constraint_values(x)[j]
                    ),
                },
            ])

        else:
            constraints.append({
                "type": "ineq",
                "fun": lambda x, j=i: (
                    problem.constraint_values(x)[j]
                ),
            })

    return constraints


def _de_constraints(problem, equality_tol):
    """Construct SciPy nonlinear constraints for DE."""

    if not problem.constraints:
        return ()

    lower = []
    upper = []

    for constraint in problem.constraints:
        if constraint["operator"] == "==":
            lower.append(-equality_tol)
            upper.append(equality_tol)
        else:
            lower.append(0.0)
            upper.append(np.inf)

    return (
        NonlinearConstraint(
            problem.constraint_values,
            np.asarray(lower),
            np.asarray(upper),
        ),
    )


# ============================================================
# 3. RUN OPTIMIZATION
# ============================================================

def run_optimization(problem, algorithm_path=None):
    """
    Execute the configured optimization algorithm.

    Returns
    -------
    dict
        SciPy result, best feasible design and evaluation history.

    Notes
    -----
    All variables are normalized to [0,1].
    Failed AVL evaluations raise exceptions during development.
    """

    settings = load_algorithm(algorithm_path)
    method = settings["method"]

    # Tolerances apply to normalized constraint residuals.
    equality_tol = float(settings.get("equality_tol", 1e-4))
    inequality_tol = float(settings.get("catol", 1e-6))

    if equality_tol <= 0 or inequality_tol < 0:
        raise ValueError("Invalid constraint tolerances.")

    # Track distinct evaluations, not repeated cache requests.
    history = []
    seen = set()

    def evaluate(x):
        """Evaluate one candidate and record its results."""

        x = np.asarray(x, dtype=float)
        result = problem.evaluate(x)

        key = tuple(x.tolist())

        if key not in seen:
            seen.add(key)

            feasible = check_feasibility(
                problem.constraints,
                result["metrics"],
                inequality_tol=inequality_tol,
                equality_tol=equality_tol,
            )

            history.append({
                "evaluation": len(history) + 1,
                "x_normalized": x.copy(),
                "x_physical": result["x_physical"].copy(),
                "objective": result["objective"]["value"],
                "scipy_objective": result["objective"]["scipy_value"],
                "constraints": result["constraints"].copy(),
                "metrics": result["metrics"].copy(),
                "feasible": feasible,
            })

        return result


    # --------------------------------------------------------
    # HANDLE OUT-OF-BOUNDS OPTIMIZER TRIAL POINTS
    # --------------------------------------------------------

    def bound_violation(x):
        """
        Measure violation of normalized variable bounds.

        Returns zero for valid designs.
        Does not modify or clip the design vector.
        """
        x = np.asarray(x, dtype=float)

        lower = np.maximum(-x, 0.0)
        upper = np.maximum(x - 1.0, 0.0)

        return float(np.sum(lower + upper))


    def objective(x):
        """Return objective or penalty for invalid trials."""

        violation = bound_violation(x)

        if violation > 0:
            # No AVL evaluation for out-of-bounds designs.
            return 1e6 * (1.0 + violation)

        return evaluate(x)["objective"]["scipy_value"]


    def constraint_values(x):
        """Return constraint residuals without invalid AVL runs."""

        violation = bound_violation(x)

        if violation > 0:
            # Negative residuals indicate infeasibility.
            return np.full(
                len(problem.constraints),
                -violation,
                dtype=float,
            )

        result = evaluate(x)

        return np.asarray(
            [
                result["constraints"][c["name"]]
                for c in problem.constraints
            ],
            dtype=float,
        )


    # The constraint builders expect constraint_values(x).
    # They can use a lightweight adapter with the same interface.
    adapter = SimpleNamespace(
        constraints=problem.constraints,
        constraint_values=constraint_values,
    )

    # --------------------------------------------------------
    # COBYLA
    # --------------------------------------------------------

    if method == "COBYLA":

        constraints = _cobyla_constraints(
            adapter, equality_tol
        )

        solver_result = minimize(
            objective,
            x0=problem.x0.copy(),
            method="COBYLA",

            # Explicit normalized variable bounds.
            bounds=[(0.0, 1.0)] * len(problem.variables),

            constraints=constraints,
            options={
                "maxiter": int(settings.get("maxiter", 500)),
                # "rhobeg": float(settings.get("rhobeg", 0.5)),
                "tol": float(settings.get("tol", 1e-5)),
                "catol": float(settings.get("catol", 1e-4)),
                "disp": bool(settings.get("disp", False)),
            },
        )

    # --------------------------------------------------------
    # DIFFERENTIAL EVOLUTION
    # --------------------------------------------------------

    elif method == "DE":

        constraints = _de_constraints(
            adapter, equality_tol
        )

        solver_result = differential_evolution(
            objective,
            bounds=[(0.0, 1.0)] * len(problem.variables),
            constraints=constraints,
            maxiter=int(settings.get("maxiter", 100)),
            popsize=int(settings.get("popsize", 10)),
            strategy=settings.get("strategy", "best1bin"),
            mutation=settings.get("mutation", (0.5, 1.0)),
            recombination=float(
                settings.get("recombination", 0.7)
            ),
            tol=float(settings.get("tol", 1e-4)),
            atol=float(settings.get("atol", 0.0)),
            polish=bool(settings.get("polish", False)),
            seed=settings.get("seed", None),
            disp=bool(settings.get("disp", False)),
            x0=problem.x0.copy(),
            workers=1,
        )


    # --------------------------------------------------------
    # CMA-ES
    # --------------------------------------------------------

    elif method == "CMA_ES":

        n_variables = len(problem.variables)

        sigma0 = float(settings.get("sigma0", 0.2))
        popsize = int(settings.get("popsize", 16))
        maxiter = int(settings.get("maxiter", 100))

        if sigma0 <= 0:
            raise ValueError("CMA-ES sigma0 must be positive.")

        if popsize < 2 or maxiter < 1:
            raise ValueError("Invalid CMA-ES population or iterations.")

        options = {
            "bounds": [
                [0.0] * n_variables,
                [1.0] * n_variables,
            ],
            "popsize": popsize,
            "maxiter": maxiter,
            "seed": settings.get("seed", 42),
            "verbose": -9,
            "verb_disp": 1 if settings.get("disp", False) else 0,
            "verb_log": 0,
        }

        es = cma.CMAEvolutionStrategy(
            problem.x0.tolist(),
            sigma0,
            options,
        )

        # CMA-ES updates its distribution from candidate
        # rankings. We rank all feasible candidates ahead of
        # infeasible candidates instead of using a fixed
        # objective penalty.
        def constraint_violation(result):
            total = 0.0

            for constraint in problem.constraints:

                residual = float(
                    result["constraints"][constraint["name"]]
                )

                if constraint["operator"] == "==":
                    violation = max(
                        0.0,
                        abs(residual) - equality_tol,
                    )
                else:
                    violation = max(
                        0.0,
                        -residual - inequality_tol,
                    )

                total += violation

            return total

        for generation in range(maxiter):

            if es.stop():
                break

            solutions = es.ask()

            ranking_data = []


            # Evaluate the population.
            fitness = np.empty(len(solutions))
            violations = np.empty(len(solutions))

            for i, x in enumerate(solutions):

                candidate = evaluate(x)

                fitness[i] = candidate["objective"]["scipy_value"]
                violations[i] = constraint_violation(candidate)

            feasible_mask = violations == 0.0

            if np.any(feasible_mask):

                # Keep actual objective values for feasible
                # designs, preserving fitness variation.
                feasible_values = fitness[feasible_mask]

                worst_feasible = np.max(feasible_values)
                objective_scale = max(
                    np.ptp(feasible_values), 1.0
                )

                # Rank infeasible candidates after all
                # feasible candidates, using their violation.
                fitness[~feasible_mask] = (
                    worst_feasible
                    + objective_scale
                    * (1.0 + violations[~feasible_mask])
                )

            else:
                # If no feasible design exists, prioritize
                # reducing normalized constraint violations.
                fitness = violations.copy()

            es.tell(solutions, fitness.tolist())


            if settings.get("disp", False):

                feasible_count = sum(
                    item[0] == 0 for item in ranking_data
                )

                feasible_history = [
                    item for item in history
                    if item["feasible"]
                ]

                best_value = (
                    min(
                        feasible_history,
                        key=lambda item: item["scipy_objective"],
                    )["objective"]
                    if feasible_history else float("nan")
                )

                print(
                    f"Generation {generation + 1:4d} | "
                    f"Feasible {feasible_count:3d}/{len(solutions)} | "
                    f"Best objective {best_value:.6f}"
                )

        stop_reasons = es.stop()

        # CMA-ES returns its preferred design, which is
        # not necessarily feasible. The common reporting
        # section will independently select the best
        # feasible candidate from the history.
        x_final = es.result.xbest

        if x_final is None:
            x_final = problem.x0.copy()

        # Reuse SciPy's result structure so the rest of
        # optimizer.py remains solver-independent.
        solver_result = OptimizeResult(
            x=np.asarray(x_final, dtype=float),
            success=(
                bool(stop_reasons)
                and "maxiter" not in stop_reasons
            ),
            message=(
                str(stop_reasons)
                if stop_reasons
                else "Maximum configured generations reached."
            ),
            nit=es.countiter,
            nfev=len(history),
        )


    # --------------------------------------------------------
    # 4. IDENTIFY BEST FEASIBLE DESIGN
    # --------------------------------------------------------

    # Include the final solver point in the evaluation record,
    # provided it lies within the normalized bounds.
    final_x = np.asarray(solver_result.x, dtype=float)

    if (
        np.isfinite(final_x).all()
        and np.all(final_x >= 0)
        and np.all(final_x <= 1)
    ):
        evaluate(final_x)

    feasible_designs = [
        item for item in history if item["feasible"]
    ]

    best = (
        min(
            feasible_designs,
            key=lambda item: item["scipy_objective"],
        )
        if feasible_designs else None
    )

    # --------------------------------------------------------
    # 5. RETURN RESULTS
    # --------------------------------------------------------

    return {
        "method": method,
        "success": bool(solver_result.success),
        "message": str(solver_result.message),
        "solver_result": solver_result,
        "best_feasible": best,
        "history": history,
        "evaluations": len(history),
        "cache_hits": problem.cache_hits,
    }


# ============================================================
# 4. PRINT OPTIMIZATION SUMMARY
# ============================================================

def print_optimization_summary(result, problem):
    """Display solver status and the best feasible design."""

    print("\n" + "=" * 60)
    print(" OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"Algorithm:       {result['method']}")
    print(f"Solver success:  {result['success']}")
    print(f"Termination:     {result['message']}")
    print(f"AVL evaluations: {result['evaluations']}")
    print(f"Cache hits:      {result['cache_hits']}")

    best = result["best_feasible"]

    if best is None:
        print("\nNo feasible design found.")
        return

    print("\nBest feasible design:")

    for variable, value in zip(
        problem.variables, best["x_physical"]
    ):
        name = variable["name"]

        if "section_id" in variable:
            name += f"[section {variable['section_id']}]"

        if "control" in variable:
            name += f"[{variable['control']}]"

        print(f"  {name:<28} {value:12.6f}")

    print("\nObjective:")
    print(
        f"  {problem.objective['metric']}: "
        f"{best['objective']:.6f}"
    )

    print("\nConstraints:")

    for constraint in problem.constraints:
        name = constraint["name"]
        actual = best["metrics"][constraint["metric"]]

        print(
            f"  {name:<24}"
            f"{actual:12.6f} "
            f"{constraint['operator']} "
            f"{constraint['value']}"
        )

    print("\nBest design feasible: True")
    print("=" * 60)
