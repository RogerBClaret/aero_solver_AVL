
import sys
from pathlib import Path

import numpy as np

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
    variable_bounds,
)
from src.optimization.evaluator import Evaluator


def test_evaluator():

    # 1. Load baseline configuration
    inputs = load_inputs()

    variables = load_variables(
        ROOT / "input" / "optimization_variables.yaml",
        inputs
    )

    evaluator = Evaluator(
        inputs=inputs,
        variables=variables,
        config_path=ROOT / "input" / "optimization.yaml"
    )

    x0 = initial_vector(variables)
    bounds = variable_bounds(variables)

    names = [v["name"] for v in variables]

    # 2. Define candidate designs
    designs = {
        "baseline": x0.copy(),
        "larger_tip_chord": x0.copy(),
        "less_tip_twist": x0.copy(),
    }

    # Modify only selected active variables
    chord_idx = names.index("geometry.tip_chord")
    twist_idx = names.index("geometry.tip_twist")

    designs["larger_tip_chord"][chord_idx] = 1.2
    designs["less_tip_twist"][twist_idx] = -1.0

    # 3. Evaluate candidates
    for design_name, x in designs.items():

        print(f"\n{'=' * 50}")
        print(f"DESIGN: {design_name}")
        print(f"{'=' * 50}")

        assert np.all(x >= bounds[:, 0])
        assert np.all(x <= bounds[:, 1])

        output = evaluator.evaluate(x)

        print("\nDESIGN VARIABLES")
        for name, value in zip(names, x):
            print(f"  {name}: {value:.5f}")

        print("\nOBJECTIVES")
        for obj in output["objectives"]:
            print(
                f"  {obj['metric']}: "
                f"{obj['value']:.8f}"
            )

        print(
            f"\nWeighted objective: "
            f"{output['objective']:.8f}"
        )

        print("\nCONSTRAINTS")
        for constraint in output["constraints"]:
            status = (
                "PASS" if constraint["satisfied"]
                else "FAIL"
            )

            print(
                f"  [{status}] "
                f"{constraint['metric']} "
                f"{constraint['operator']} "
                f"{constraint['limit']}"
                f" | actual={constraint['value']:.6f}"
                f" | residual={constraint['residual']:.6f}"
            )

        print(f"\nFeasible: {output['feasible']}")

        assert np.isfinite(output["objective"])
        assert np.all(np.isfinite(output["residuals"]))

    # 4. Check evaluation count
    assert evaluator.n_evaluations == 3

    # 5. Verify cache (no extra AVL run)
    evaluator.evaluate(designs["less_tip_twist"])

    assert evaluator.n_evaluations == 3

    # 6. Verify SciPy interface
    f = evaluator.objective(designs["less_tip_twist"])
    constraints = evaluator.scipy_constraints()

    assert np.isfinite(f)

    for constraint in constraints:
        assert np.isfinite(
            constraint["fun"](designs["less_tip_twist"])
        )

    assert evaluator.n_evaluations == 3

    print("\n" + "=" * 50)
    print("SUCCESS: Evaluator validated.")
    print(f"AVL evaluations: {evaluator.n_evaluations}")
    print("=" * 50)


if __name__ == "__main__":
    test_evaluator()

