
"""Evaluate objectives and constraints for one half-wing AVL design."""

import copy
from pathlib import Path

import numpy as np
import yaml

from avlwrapper import Session

from src.build_model import build_geometry, build_case
from src.optimization.variables import apply_variables
from src.optimization.metric_registry import evaluate_metric


class Evaluator:

    def __init__(
        self,
        inputs,
        variables,
        config_path="input/optimization.yaml"
    ):
        self.inputs = copy.deepcopy(inputs)
        self.variables = variables

        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.objectives = self.config.get("objectives", [])
        self.constraints = self.config.get("constraints", [])

        if not self.objectives:
            raise ValueError("At least one objective is required")

        self.n_evaluations = 0
        self.history = []

        self._last_x = None
        self._last_result = None

    def evaluate(self, x):
        """Evaluate one candidate design using ONE AVL case.

        Returns objectives, constraint residuals and metrics.

        Constraint convention:
            g(x) >= 0 means feasible.
        """

        x = np.asarray(x, dtype=float)

        # Reuse the last evaluation when optimizer requests
        # objective and constraints at the same design.
        if (
            self._last_x is not None
            and np.array_equal(x, self._last_x)
        ):
            return self._last_result

        # 1. Apply active variables to copies of YAML data
        geom, mesh, controls, flight = apply_variables(
            x, self.variables, self.inputs
        )

        # 2. Build ONE physical half-wing
        geometry = build_geometry(
            geom, mesh, controls
        )

        # 3. Build exactly ONE operating point
        case = build_case(flight, controls)

        # 4. Run AVL once
        results = Session(
            geometry=geometry,
            cases=[case]
        ).get_results()

        # 5. Metric accessor (geometry + aerodynamic)
        metric_cache = {}

        def metric(name):
            if name not in metric_cache:
                value = float(
                    evaluate_metric(
                        name,
                        geom,
                        results,
                        flight=flight
                    )
                )
                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite metric: {name}"
                    )
                metric_cache[name] = value
            return metric_cache[name]

        # 6. Objectives
        objective_values = []
        weighted_objective = 0.0

        for obj in self.objectives:
            value = metric(obj["metric"])
            direction = obj["direction"]

            weight = float(obj.get("weight", 1.0))
            scale = float(obj.get("scale", 1.0))

            if scale <= 0 or weight < 0:
                raise ValueError(
                    "Objective scale must be positive "
                    "and weight nonnegative"
                )

            if direction == "minimize":
                contribution = value / scale

            elif direction == "maximize":
                contribution = -value / scale

            elif direction == "target":
                target = float(obj["target"])
                contribution = (
                    (value - target) / scale
                ) ** 2

            else:
                raise ValueError(
                    f"Unknown objective direction: {direction}"
                )

            weighted_objective += weight * contribution

            objective_values.append({
                "metric": obj["metric"],
                "value": value,
                "contribution": weight * contribution,
            })

        # 7. Constraints: positive residual = feasible
        constraint_values = []
        residuals = []

        for constraint in self.constraints:
            name = constraint["metric"]
            value = metric(name)
            limit = float(constraint["value"])
            operator = constraint["operator"]

            if operator == ">=":
                residual = value - limit

            elif operator == "<=":
                residual = limit - value

            elif operator == "==":
                residual = value - limit

            else:
                raise ValueError(
                    f"Unknown constraint operator: {operator}"
                )

            residuals.append(residual)

            constraint_values.append({
                "metric": name,
                "value": value,
                "operator": operator,
                "limit": limit,
                "residual": residual,
                "satisfied": (
                    abs(residual) <= 1e-8
                    if operator == "=="
                    else residual >= -1e-8
                )
            })

        self.n_evaluations += 1

        output = {
            "x": x.copy(),
            "objective": float(weighted_objective),
            "objectives": objective_values,
            "constraints": constraint_values,
            "residuals": np.asarray(
                residuals, dtype=float
            ),
            "metrics": metric_cache,
            "feasible": all(
                c["satisfied"] for c in constraint_values
            ),
            "results": results,
        }

        self.history.append({
            "evaluation": self.n_evaluations,
            "x": x.tolist(),
            "objective": output["objective"],
            "feasible": output["feasible"],
            "metrics": metric_cache.copy(),
        })

        self._last_x = x.copy()
        self._last_result = output

        return output

    def objective(self, x):
        """Scalar objective for scipy.optimize."""
        return self.evaluate(x)["objective"]

    def constraint_residuals(self, x):
        """Signed residuals; equality constraints require zero."""
        return self.evaluate(x)["residuals"]

    def scipy_constraints(self):
        """Build scipy.optimize compatible constraints."""

        constraints = []

        for i, item in enumerate(self.constraints):
            kind = (
                "eq" if item["operator"] == "=="
                else "ineq"
            )

            constraints.append({
                "type": kind,
                "fun": lambda x, i=i: (
                    self.constraint_residuals(x)[i]
                ),
            })

        return constraints
