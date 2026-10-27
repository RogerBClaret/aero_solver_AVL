
"""
Plot optimization convergence from recorded evaluation history.

Figures:
1. Objective evolution
2. Design-variable evolution
3. Constraint evolution

No AVL evaluations are performed.
"""

import numpy as np
import matplotlib.pyplot as plt


def _variable_label(variable):
    """Construct a unique label from variable configuration."""

    label = variable["name"]

    if "section_id" in variable:
        label += f" [section {variable['section_id']}]"

    if "control" in variable:
        label += f" [{variable['control']}]"

    return label


def plot_optimization_history(
    result,
    problem,
    show=True,
    save_dir=None,
):
    """
    Plot objective, design variables and constraints.

    Parameters
    ----------
    result : dict
        Output from run_optimization().

    problem : OptimizationProblem
        Problem definition containing variables, bounds,
        objective and constraints.

    show : bool
        Display figures.

    save_dir : str or Path, optional
        Directory in which to save the figures.

    Returns
    -------
    dict
        Matplotlib figures indexed by plot type.
    """

    history = result["history"]

    if not history:
        raise ValueError("Optimization history is empty.")

    evaluations = np.asarray(
        [item["evaluation"] for item in history]
    )

    feasible = np.asarray(
        [item["feasible"] for item in history],
        dtype=bool,
    )

    figures = {}

    # ========================================================
    # 1. OBJECTIVE EVOLUTION
    # ========================================================

    objective = np.asarray(
        [item["objective"] for item in history],
        dtype=float,
    )

    signed_objective = np.asarray(
        [item["scipy_objective"] for item in history],
        dtype=float,
    )

    # Running best feasible objective in the original
    # physical metric, using the signed optimization
    # objective to handle minimization or maximization.
    best_indices = []
    best_index = None

    for i in range(len(history)):

        if feasible[i]:
            if (
                best_index is None
                or signed_objective[i]
                < signed_objective[best_index]
            ):
                best_index = i

        best_indices.append(best_index)

    best_objective = np.asarray(
        [
            objective[i] if i is not None else np.nan
            for i in best_indices
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        evaluations,
        objective,
        "o-",
        markersize=3,
        linewidth=1,
        alpha=0.55,
        label="Evaluated objective",
    )

    ax.plot(
        evaluations,
        best_objective,
        linewidth=2.5,
        label="Best feasible objective",
    )


    # --------------------------------------------------------
    # SCALE Y-AXIS AROUND THE BEST FEASIBLE OBJECTIVES
    # --------------------------------------------------------

    valid_best = best_objective[np.isfinite(best_objective)]

    if valid_best.size > 0:

        # Use the convergence curve rather than all trial
        # objectives, which may contain extreme outliers.
        y_min = np.min(valid_best)
        y_max = np.max(valid_best)

        span = y_max - y_min

        # Avoid a zero-height axis if convergence is flat.
        margin = max(
            0.05 * span,
            0.01 * max(abs(y_min), abs(y_max)),
            1e-9,
        )

        ax.set_ylim(
            y_min - margin,
            y_max + margin,
        )


    ax.scatter(
        evaluations[~feasible],
        objective[~feasible],
        marker="x",
        label="Infeasible evaluations",
        zorder=3,
    )

    ax.set(
        xlabel="Unique AVL evaluation",
        ylabel=problem.objective["metric"],
        title="Objective convergence",
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    figures["objective"] = fig

    # ========================================================
    # 2. DESIGN-VARIABLE EVOLUTION
    # ========================================================

    n_variables = len(problem.variables)

    design_vectors = np.asarray(
        [item["x_physical"] for item in history],
        dtype=float,
    )

    fig, axes = plt.subplots(
        n_variables,
        1,
        figsize=(10, max(3 * n_variables, 4)),
        sharex=True,
        squeeze=False,
    )

    for i, variable in enumerate(problem.variables):

        ax = axes[i, 0]
        lower, upper = problem.bounds[i]

        ax.plot(
            evaluations,
            design_vectors[:, i],
            "o-",
            markersize=3,
            linewidth=1,
        )

        ax.axhline(
            lower,
            linestyle="--",
            linewidth=1,
            label="Lower bound",
        )

        ax.axhline(
            upper,
            linestyle="--",
            linewidth=1,
            label="Upper bound",
        )

        ax.set_ylabel(_variable_label(variable))
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc="best")

    axes[-1, 0].set_xlabel("Unique AVL evaluation")
    fig.suptitle("Design-variable evolution")
    fig.tight_layout()

    figures["variables"] = fig

    # ========================================================
    # 3. CONSTRAINT EVOLUTION
    # ========================================================

    n_constraints = len(problem.constraints)

    if n_constraints:

        fig, axes = plt.subplots(
            n_constraints,
            1,
            figsize=(10, max(3 * n_constraints, 4)),
            sharex=True,
            squeeze=False,
        )

        for i, constraint in enumerate(problem.constraints):

            ax = axes[i, 0]

            metric_name = constraint["metric"]
            target = float(constraint["value"])
            operator = constraint["operator"]

            values = np.asarray(
                [
                    item["metrics"][metric_name]
                    for item in history
                ],
                dtype=float,
            )

            ax.plot(
                evaluations,
                values,
                "o-",
                markersize=3,
                linewidth=1,
                label="Actual value",
            )

            ax.axhline(
                target,
                linestyle="--",
                linewidth=1.5,
                label=f"Limit ({operator} {target:g})",
            )

            # Shade the feasible side of each inequality.
            if operator == "<=":
                ax.axhspan(
                    min(values.min(), target),
                    target,
                    alpha=0.08,
                )

            elif operator == ">=":
                ax.axhspan(
                    target,
                    max(values.max(), target),
                    alpha=0.08,
                )

            # Highlight violated evaluations using the
            # residuals recorded by the optimizer.
            residuals = np.asarray(
                [
                    item["constraints"][constraint["name"]]
                    for item in history
                ],
                dtype=float,
            )

            if operator == "==":
                violated = ~np.isclose(
                    residuals, 0.0, atol=1e-4
                )
            else:
                violated = residuals < -1e-6

            ax.scatter(
                evaluations[violated],
                values[violated],
                marker="x",
                zorder=4,
                label="Violation",
            )

            ax.set_ylabel(metric_name)
            ax.set_title(constraint["name"])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        axes[-1, 0].set_xlabel("Unique AVL evaluation")
        fig.suptitle("Constraint evolution")
        fig.tight_layout()

        figures["constraints"] = fig

    # ========================================================
    # 4. OPTIONAL FIGURE EXPORT
    # ========================================================

    if save_dir is not None:

        from pathlib import Path

        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)

        for name, fig in figures.items():
            fig.savefig(
                folder / f"optimization_{name}.png",
                dpi=200,
                bbox_inches="tight",
            )

    if show:
        plt.show()

    return figures
