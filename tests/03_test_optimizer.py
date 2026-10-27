print('La paciència és la mare de la ciència')

"""
03 - Run, validate and visualize AVL optimization.

Run from project root:
    python tests/03_test_optimizer.py

Uses the optimization algorithm selected in algorithm.yaml.
"""

import sys
from pathlib import Path

import numpy as np
from avlwrapper import Session

# ------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from src import load_inputs
from src.avl_geometry import build_geometry
from src.flight_case import build_case
from src.plotter import plot_all

from src.optimization.problem import OptimizationProblem
from src.optimization.optimizer import (
    run_optimization,
    print_optimization_summary,
)
from src.optimization.variables import apply_variables
from src.optimization.plotter import plot_optimization_history


OUTPUT_DIR = ROOT / "outputs"


def header(title):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. LOAD INPUTS AND DEFINE OPTIMIZATION PROBLEM
    # ========================================================

    header("1. OPTIMIZATION PROBLEM")

    geometry, mesh, controls, flight = load_inputs()

    baseline = {
        "geometry": geometry,
        "mesh": mesh,
        "controls": controls,
        "flight": flight,
    }

    problem = OptimizationProblem(baseline)
    problem.summary()

    # ========================================================
    # 2. BASELINE EVALUATION
    # ========================================================

    header("2. BASELINE EVALUATION")

    baseline_result = problem.evaluate_baseline()

    print(
        f"Baseline objective: "
        f"{baseline_result['objective']['value']:.6f}"
    )
    print(f"Baseline feasible: {baseline_result['feasible']}")

    # ========================================================
    # 3. RUN OPTIMIZATION
    # ========================================================

    header("3. RUN OPTIMIZATION")

    result = run_optimization(problem)

    print_optimization_summary(result, problem)

    # ========================================================
    # 4. VALIDATE BEST FEASIBLE DESIGN
    # ========================================================

    header("4. OPTIMUM VALIDATION")

    best = result["best_feasible"]

    if best is None:
        print("No feasible candidate found.")
        print("Skipping optimized-aircraft visualization.")
    else:

        # Reuse the problem's cached AVL evaluation.
        validated = problem.evaluate(best["x_normalized"])

        assert validated["success"]

        assert np.isclose(
            validated["objective"]["value"],
            best["objective"],
            rtol=1e-9,
            atol=1e-10,
        )

        for constraint in problem.constraints:
            name = constraint["name"]

            assert np.isfinite(best["constraints"][name])

            print(
                f"{name:<25} "
                f"residual = {best['constraints'][name]:+.6f}"
            )

        print("\n[PASS] Best feasible result is consistent.")

        # Compare objective values in the signed minimization
        # convention, so this also supports maximization.
        f_initial = baseline_result["objective"]["scipy_value"]
        f_best = best["scipy_objective"]

        print(
            f"\nBaseline objective: "
            f"{baseline_result['objective']['value']:.6f}"
        )
        print(f"Optimized objective: {best['objective']:.6f}")

        if f_best < f_initial:
            print("[PASS] Objective improved relative to baseline.")
        else:
            print("[INFO] No objective improvement over baseline.")

    # ========================================================
    # 5. PLOT OPTIMIZATION CONVERGENCE
    # ========================================================

    header("5. OPTIMIZATION HISTORY")

    figures = plot_optimization_history(
        result,
        problem,
        show=False,
        save_dir=OUTPUT_DIR,
    )

    print(f"\nSaved {len(figures)} convergence figures.")
    print(f"Directory: {OUTPUT_DIR}")

    # ========================================================
    # 6. RECONSTRUCT OPTIMIZED AIRCRAFT
    # ========================================================

    if best is not None:

        header("6. RECONSTRUCT OPTIMIZED AIRCRAFT")

        # Apply the optimized physical design variables
        # to fresh copies of the original configurations.
        candidate = apply_variables(
            problem.variables,
            best["x_physical"],
            baseline,
        )

        optimized_geometry_data = candidate["geometry"]
        optimized_mesh_data = candidate["mesh"]
        optimized_controls_data = candidate["controls"]
        optimized_flight_data = candidate["flight"]

        # Build the actual optimized AVL geometry.
        optimized_geometry = build_geometry(
            optimized_geometry_data,
            optimized_mesh_data,
            optimized_controls_data,
        )

        # Build the flight case using updated geometry
        # so lift trim uses the optimized reference area.
        optimized_case = build_case(
            optimized_flight_data,
            optimized_controls_data,
            optimized_geometry_data,
        )

        # Rerun AVL for visualization and spanwise loads.
        optimized_session = Session(
            geometry=optimized_geometry,
            cases=[optimized_case],
        )

        optimized_results = optimized_session.get_results()

        print("[PASS] Optimized aircraft reconstructed.")
        print("[PASS] Optimized AVL case executed.")

        # ====================================================
        # 7. VISUALIZE USING EXISTING src/plotter.py
        # ====================================================

        header("7. OPTIMIZED AIRCRAFT VISUALIZATION")

        plot_all(
            geometry_data=optimized_geometry_data,
            controls_data=optimized_controls_data,
            flight_data=optimized_flight_data,
            results=optimized_results,
            output_dir=OUTPUT_DIR,
        )

    # ========================================================
    # 8. FINAL SUMMARY
    # ========================================================

    header("8. TEST SUMMARY")

    print(f"Algorithm:             {result['method']}")
    print(f"Solver success:        {result['success']}")
    print(f"Unique evaluations:    {result['evaluations']}")
    print(f"Cache hits:            {result['cache_hits']}")
    print(f"Feasible design found: {best is not None}")

    print("\n[PASS] OPTIMIZER TEST COMPLETED")


if __name__ == "__main__":
    main()
