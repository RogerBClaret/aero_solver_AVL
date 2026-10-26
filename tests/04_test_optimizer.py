import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = (
    Path(__file__).resolve().parents[1]
    if "__file__" in globals()
    else Path.cwd()
)
sys.path.insert(0, str(ROOT))

from src.build_model import load_inputs
from src.optimization.variables import (
    load_variables,
    initial_vector,
)
from src.optimization.evaluator import Evaluator
from src.optimization.optimizer import (
    load_algorithm,
    run_optimization,
)


def print_design(label, variables, output):
    print(f"\n{'=' * 55}")
    print(label)
    print("=" * 55)

    print("\nDESIGN VARIABLES")
    for var, value in zip(variables, output["x"]):
        print(f"  {var['name']:35s}: {value:.6f}")

    print("\nOBJECTIVES")
    for obj in output["objectives"]:
        print(
            f"  {obj['metric']:25s}: {obj['value']:.8f}"
        )

    print(
        f"\n  Weighted objective: "
        f"{output['objective']:.8f}"
    )

    print("\nCONSTRAINTS")
    for c in output["constraints"]:
        status = "PASS" if c["satisfied"] else "FAIL"

        print(
            f"  [{status}] {c['metric']:25s} "
            f"{c['operator']} {c['limit']:.6f} "
            f"| actual={c['value']:.6f} "
            f"| residual={c['residual']:.6f}"
        )

    print(f"\nFeasible: {output['feasible']}")


def test_optimizer():
    # 1. Load model and active variables.
    inputs = load_inputs()

    variables = load_variables(
        ROOT / "input" / "optimization_variables.yaml",
        inputs,
    )

    evaluator = Evaluator(
        inputs=inputs,
        variables=variables,
        config_path=ROOT / "input" / "optimization.yaml",
    )

    settings = load_algorithm(
        ROOT / "input" / "optimization_algorithm.yaml"
    )

    # test.
    # settings["algorithm"] = "COBYLA"
    # settings["max_iterations"] = 3000
    # settings["display"] = False

    # 2. Evaluate baseline.
    x0 = initial_vector(variables)
    baseline = evaluator.evaluate(x0)

    print_design("INITIAL DESIGN", variables, baseline)

    # 3. Run optimization.
    result = run_optimization(
        evaluator=evaluator,
        variables=variables,
        settings=settings,
    )

    final = result["evaluation"]

    print_design("FINAL DESIGN", variables, final)

    # 4. Baseline vs final comparison.
    print(f"\n{'=' * 55}")
    print("BASELINE vs OPTIMIZED")
    print("=" * 55)

    print("\nDESIGN VARIABLES")

    for i, var in enumerate(variables):
        initial = baseline["x"][i]
        optimized = final["x"][i]

        print(
            f"  {var['name']:35s} "
            f"{initial:10.5f} -> {optimized:10.5f}"
        )

    print("\nOBJECTIVES")

    for initial_obj, final_obj in zip(
        baseline["objectives"],
        final["objectives"],
    ):
        print(
            f"  {initial_obj['metric']:25s} "
            f"{initial_obj['value']:.8f} -> "
            f"{final_obj['value']:.8f}"
        )

    print("\nCONSTRAINTS")

    for initial_c, final_c in zip(
        baseline["constraints"],
        final["constraints"],
    ):
        print(
            f"  {initial_c['metric']:25s} "
            f"{initial_c['value']:.6f} -> "
            f"{final_c['value']:.6f} "
            f"({ 'PASS' if final_c['satisfied'] else 'FAIL' })"
        )

    print(
        f"\nWeighted objective: "
        f"{baseline['objective']:.8f} -> "
        f"{final['objective']:.8f}"
    )

    print(f"AVL evaluations: {evaluator.n_evaluations}")
    print(f"Solver success: {result.success}")
    print(f"Final feasible: {final['feasible']}")

    # 5. Plot evaluation history.
    history = evaluator.history

    evaluations = [
        entry["evaluation"] for entry in history
    ]

    objectives = [
        entry["objective"] for entry in history
    ]

    best_objective = np.minimum.accumulate(objectives)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        evaluations,
        objectives,
        "o-",
        label="Evaluated objective",
        alpha=0.6,
    )

    ax.plot(
        evaluations,
        best_objective,
        "-",
        linewidth=2,
        label="Best objective found",
    )

    ax.axhline(
        baseline["objective"],
        linestyle="--",
        label="Initial objective",
    )

    ax.set_xlabel("AVL evaluation")
    ax.set_ylabel("Weighted objective")
    ax.set_title(f"{settings['algorithm']} Optimization Convergence")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()

    # 6. Plot variable evolution.
    fig, axes = plt.subplots(
        len(variables),
        1,
        figsize=(10, max(3, 2.2 * len(variables))),
        sharex=True,
        squeeze=False,
    )

    for i, var in enumerate(variables):
        values = [
            entry["x"][i] for entry in history
        ]

        ax = axes[i, 0]
        ax.plot(evaluations, values, "o-")
        ax.axhline(
            baseline["x"][i],
            linestyle="--",
            alpha=0.5,
        )

        ax.set_ylabel(var["name"])
        ax.grid(True)

    axes[-1, 0].set_xlabel("AVL evaluation")
    fig.suptitle("Design Variable Evolution")
    fig.tight_layout()
    plt.show()

    # 7. Validation.
    assert np.isfinite(final["objective"])
    assert np.all(np.isfinite(final["x"]))

    print(
        f"\nOptimization completed using "
        f"{settings['algorithm']}."
    )

if __name__ == "__main__":
    test_optimizer()
