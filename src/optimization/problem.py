
"""
Assemble and evaluate an AVL optimization problem.

Connects:
- Design variables
- Objective function
- Constraints
- AVL design evaluator

All design vectors are normalized to [0, 1].
Identical evaluations are cached to avoid repeated AVL runs.

This module does not execute an optimization algorithm.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from .variables import (
    load_variables,
    get_initial_vector,
    get_bounds,
    normalize,
)

from .objectives import (
    load_objective,
    evaluate_objective,
    objective_value,
)

from .constraints import (
    load_constraints,
    evaluate_constraints,
    check_feasibility,
)

from .evaluator import evaluate_design

from ..input_loader import ROOT


class OptimizationProblem:
    """Configuration-driven AVL optimization problem."""

    def __init__(
        self,
        baseline,
        config_dir=None,
        include_spanloads=None,
    ):
        """
        Initialize and validate the optimization problem.

        Parameters
        ----------
        baseline : dict
            Baseline geometry, mesh, controls and flight inputs.

        config_dir : str or Path, optional
            Directory containing optimization YAML files.

        include_spanloads : bool, optional
            Override automatic strip-load calculation.
        """

        self.baseline = deepcopy(baseline)

        folder = (
            Path(config_dir)
            if config_dir is not None
            else ROOT / "input" / "optimization"
        )

        # ----------------------------------------------------
        # 1. LOAD CONFIGURATION
        # ----------------------------------------------------

        self.variables = load_variables(
            folder / "design_variables.yaml"
        )

        self.objective = load_objective(
            folder / "objectives.yaml"
        )

        self.constraints = load_constraints(
            folder / "constraints.yaml"
        )

        # ----------------------------------------------------
        # 2. INITIAL DESIGN AND BOUNDS
        # ----------------------------------------------------

        self.bounds = get_bounds(self.variables)

        self.x0_physical = get_initial_vector(
            self.variables, self.baseline
        )

        self.x0 = normalize(
            self.x0_physical, self.bounds
        )

        if not np.isfinite(self.x0).all():
            raise ValueError("Invalid initial design vector.")

        if np.any(self.x0 < 0) or np.any(self.x0 > 1):
            raise ValueError(
                "Baseline design lies outside variable bounds."
            )

        # ----------------------------------------------------
        # 3. CONFIGURATION CONSISTENCY
        # ----------------------------------------------------

        trim = self.baseline["flight"]["flight_condition"].get(
            "trim", {}
        )

        if trim.get("enabled", False):
            trimmed = trim["variable"]

            for variable in self.variables:
                name = variable["name"]

                if trimmed == "alpha" and name == "alpha":
                    raise ValueError(
                        "Cannot optimize alpha while AVL trims alpha."
                    )

                if (
                    name == "control_deflection"
                    and variable.get("control") == trimmed
                ):
                    raise ValueError(
                        f"Cannot optimize '{trimmed}' deflection "
                        "while AVL trims that control."
                    )

        # ----------------------------------------------------
        # 4. REQUIRED ANALYSIS OUTPUTS
        # ----------------------------------------------------

        required_metrics = {
            self.objective["metric"],
            *(c["metric"] for c in self.constraints),
        }

        needs_spanloads = (
            "root_bending_moment" in required_metrics
        )

        if include_spanloads is False and needs_spanloads:
            raise ValueError(
                "Root bending moment is required, but "
                "spanwise-load calculation was disabled."
            )

        self.include_spanloads = (
            needs_spanloads
            if include_spanloads is None
            else bool(include_spanloads)
        )

        # ----------------------------------------------------
        # 5. EVALUATION CACHE
        # ----------------------------------------------------

        self._cache = {}
        self.evaluation_count = 0
        self.cache_hits = 0

    # ========================================================
    # EVALUATE ONE CANDIDATE
    # ========================================================

    def evaluate(self, x):
        """
        Evaluate a normalized candidate design.

        Returns
        -------
        dict
            Design variables, metrics, objective,
            constraint residuals and feasibility.

        Solver failures propagate as exceptions.
        """

        x = np.asarray(x, dtype=float)

        if x.shape != self.x0.shape:
            raise ValueError("Invalid candidate dimensions.")

        if not np.isfinite(x).all():
            raise ValueError("Nonfinite design vector.")

        if np.any(x < 0) or np.any(x > 1):
            raise ValueError(
                "Normalized design vector outside [0,1]."
            )

        # Exact cache key: no rounding of design variables.
        key = tuple(x.tolist())

        if key in self._cache:
            self.cache_hits += 1
            return deepcopy(self._cache[key])

        # ----------------------------------------------------
        # 1. RUN AVL AND CALCULATE METRICS
        # ----------------------------------------------------

        analysis = evaluate_design(
            x=x,
            variables=self.variables,
            baseline=self.baseline,
            normalized=True,
            include_spanloads=self.include_spanloads,
        )

        metrics = analysis["metrics"]

        # ----------------------------------------------------
        # 2. EVALUATE OBJECTIVE
        # ----------------------------------------------------

        f = evaluate_objective(
            self.objective, metrics
        )

        physical_objective = objective_value(
            self.objective, metrics
        )

        # ----------------------------------------------------
        # 3. EVALUATE CONSTRAINTS
        # ----------------------------------------------------

        residuals = evaluate_constraints(
            self.constraints, metrics
        )

        feasible = check_feasibility(
            self.constraints, metrics
        )

        # ----------------------------------------------------
        # 4. ASSEMBLE EVALUATION RESULT
        # ----------------------------------------------------

        result = {
            "success": True,
            "x_normalized": analysis["x_normalized"],
            "x_physical": analysis["x_physical"],
            "metrics": metrics,
            "objective": {
                "name": self.objective["name"],
                "metric": self.objective["metric"],
                "direction": self.objective["direction"],
                "value": physical_objective,
                "scipy_value": f,
            },
            "constraints": residuals,
            "feasible": feasible,
        }

        # Store only successful evaluations.
        self._cache[key] = deepcopy(result)
        self.evaluation_count += 1

        return result

    # ========================================================
    # CONVENIENCE METHODS
    # ========================================================

    def objective_function(self, x):
        """Return the scalar objective to minimize."""

        return self.evaluate(x)["objective"]["scipy_value"]

    def constraint_values(self, x):
        """Return normalized constraint residuals in YAML order."""

        residuals = self.evaluate(x)["constraints"]

        return np.array(
            [residuals[c["name"]] for c in self.constraints],
            dtype=float,
        )

    def evaluate_baseline(self):
        """Evaluate the original aircraft configuration."""

        return self.evaluate(self.x0)

    def clear_cache(self):
        """Clear cached evaluations and reset counters."""

        self._cache.clear()
        self.evaluation_count = 0
        self.cache_hits = 0

    def summary(self):
        """Print the configured optimization problem."""

        print("\n" + "=" * 55)
        print(" AVL OPTIMIZATION PROBLEM")
        print("=" * 55)

        print("\nDesign variables:")

        for i, variable in enumerate(self.variables):
            identifier = variable["name"]

            if "section_id" in variable:
                identifier += f"[section {variable['section_id']}]"

            if "control" in variable:
                identifier += f"[{variable['control']}]"

            lower, upper = self.bounds[i]

            print(
                f"  {identifier:<28}"
                f"{self.x0_physical[i]:>9.4f}"
                f"  [{lower:.4f}, {upper:.4f}]"
            )

        print("\nObjective:")
        print(
            f"  {self.objective['direction']} "
            f"{self.objective['metric']}"
        )

        print("\nConstraints:")

        if not self.constraints:
            print("  None")

        for constraint in self.constraints:
            print(
                f"  {constraint['name']:<24}"
                f"{constraint['metric']} "
                f"{constraint['operator']} "
                f"{constraint['value']}"
            )

        print(f"\nSpanwise loads: {self.include_spanloads}")
        print("=" * 55)
