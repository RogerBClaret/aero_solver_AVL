
"""
02 - Validate the complete AVL optimization problem.

Run from project root:
    python tests/02_test_problem_definition.py

No optimization algorithm is executed.

Tests configuration, AVL evaluation, constraint handling,
caching, design perturbations and baseline immutability.
"""
# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

import sys
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from src import load_inputs
from src.optimization.problem import OptimizationProblem
from src.optimization.variables import get_initial_vector


def header(title):
    """Print a section heading."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def print_evaluation(problem, result):
    """Display objective, metrics and constraint margins."""

    obj = result["objective"]

    print(
        f"\nObjective: {obj['name']}\n"
        f"Value:     {obj['value']:.6f}\n"
        f"Feasible:  {result['feasible']}"
    )

    print("\nConstraints:")

    for constraint in problem.constraints:
        name = constraint["name"]
        metric_name = constraint["metric"]

        actual = result["metrics"][metric_name]
        target = constraint["value"]
        residual = result["constraints"][name]

        # Equality residuals must be close to zero.
        if constraint["operator"] == "==":
            satisfied = abs(residual) <= 1e-4
        else:
            satisfied = residual >= -1e-6

        status = "PASS" if satisfied else "FAIL"

        print(
            f"  [{status}] {name:<22}"
            f" actual={actual:>11.5f}"
            f" target={target:>11.5f}"
            f" residual={residual:+.5f}"
        )


def main():

    # ========================================================
    # 1. LOAD OPTIMIZATION PROBLEM
    # ========================================================

    header("1. PROBLEM CONFIGURATION")

    geometry, mesh, controls, flight = load_inputs()

    baseline = {
        "geometry": geometry,
        "mesh": mesh,
        "controls": controls,
        "flight": flight,
    }

    problem = OptimizationProblem(baseline)
    problem.summary()

    # Store a copy of the original baseline vector.
    x_original = get_initial_vector(
        problem.variables, baseline
    ).copy()

    # ========================================================
    # 2. EVALUATE BASELINE
    # ========================================================

    header("2. BASELINE AVL EVALUATION")

    result = problem.evaluate_baseline()

    assert result["success"]
    assert np.isfinite(result["objective"]["scipy_value"])

    print_evaluation(problem, result)

    print("\n[PASS] Baseline AVL evaluation.")

    # ========================================================
    # 3. TEST EVALUATION CACHE
    # ========================================================

    header("3. CACHE VALIDATION")

    evaluations_before = problem.evaluation_count

    # Request exactly the same candidate.
    cached = problem.evaluate(problem.x0)

    assert problem.evaluation_count == evaluations_before
    assert problem.cache_hits >= 1

    assert np.isclose(
        cached["objective"]["value"],
        result["objective"]["value"],
    )

    print(f"AVL evaluations: {problem.evaluation_count}")
    print(f"Cache hits:      {problem.cache_hits}")
    print("[PASS] Cached result reused without running AVL.")

    # ========================================================
    # 4. PERTURB ONE DESIGN VARIABLE
    # ========================================================

    header("4. DESIGN PERTURBATION")

    x_candidate = problem.x0.copy()

    # Select the first variable with room for perturbation.
    # No specific geometric variable is assumed.
    step = 0.02

    index = next(
        (
            i for i, value in enumerate(x_candidate)
            if value + step <= 1.0
            or value - step >= 0.0
        ),
        None,
    )

    if index is None:
        raise ValueError("No variable available for perturbation.")

    if x_candidate[index] + step <= 1.0:
        x_candidate[index] += step
    else:
        x_candidate[index] -= step

    variable = problem.variables[index]

    print(f"Modified variable: {variable['name']}")
    print(
        f"Normalized: "
        f"{problem.x0[index]:.5f} -> "
        f"{x_candidate[index]:.5f}"
    )

    candidate = problem.evaluate(x_candidate)

    print_evaluation(problem, candidate)

    # Ensure the candidate design really changed.
    assert not np.array_equal(
        candidate["x_physical"],
        result["x_physical"],
    )

    assert candidate["success"]

    print("\n[PASS] Candidate evaluated successfully.")

    # ========================================================
    # 5. VERIFY BASELINE IMMUTABILITY
    # ========================================================

    header("5. BASELINE INTEGRITY")

    x_after = get_initial_vector(
        problem.variables, baseline
    )

    assert np.array_equal(x_original, x_after)

    print("[PASS] Baseline inputs remain unchanged.")

    # ========================================================
    # 6. SUMMARY
    # ========================================================

    header("6. FINAL TEST SUMMARY")

    print(f"Active variables: {len(problem.variables)}")
    print(f"Active constraints: {len(problem.constraints)}")
    print(f"Objective: {problem.objective['name']}")
    print(f"AVL evaluations: {problem.evaluation_count}")
    print(f"Cache hits: {problem.cache_hits}")

    print("\n[PASS] OPTIMIZATION PROBLEM DEFINITION VALIDATED")


if __name__ == "__main__":
    main()
