# AVL-Based Wing Analysis and Optimization

A modular Python framework for parametric wing geometry, aerodynamic analysis, lift-constrained trim, and single-objective wing optimization using **AVL (Athena Vortex Lattice)** through **AVLWrapper**.

The framework is intended for early-stage aerodynamic design studies, including exploration of wing span, chord distribution, sweep, twist, and control-surface geometry. Aircraft definitions, operating conditions, design variables, objectives, constraints, and optimizer settings are maintained in YAML files rather than hard-coded into the optimization routines.

> **Scope:** This is an aerodynamic analysis and optimization workflow, not a full aircraft performance or structural sizing tool. The drag reported by the current AVL workflow is principally inviscid/induced drag; realistic total aircraft drag requires additional modeling.

## Features

- Parametric, section-based wing geometry, including sweep, dihedral, taper, and twist.
- Airfoil definitions from external coordinate files.
- Flap and aileron definitions with configurable spanwise extents and hinge positions.
- Automatic AVL geometry generation and operating-point execution.
- Trim to a specified dimensional lift using AVL's angle-of-attack constraint.
- Aerodynamic and geometric metrics, including lift, drag, induced drag, span efficiency, aspect ratio, and trimmed angle of attack.
- Configuration-driven single-objective optimization with nonlinear constraints.
- **COBYLA**, **Differential Evolution (DE)**, and **CMA-ES** optimization backends.
- Cached aerodynamic evaluations, optimization history, convergence plots, and final-geometry visualization.

## Repository layout

```text
.
├── input/
│   ├── geometry.yaml                 # Wing planform, sections, and airfoils
│   ├── mesh.yaml                     # AVL panel discretization
│   ├── control_surfaces.yaml         # Flaps, ailerons, and hinge definitions
│   ├── flight_condition.yaml         # Velocity, density, trim, controls
│   ├── airfoils/                     # Airfoil coordinate files
│   └── optimization/
│       ├── design_variables.yaml     # Active variables and bounds
│       ├── objectives.yaml           # Active optimization objective
│       ├── constraints.yaml          # Aerodynamic/geometric constraints
│       └── algorithm.yaml            # Optimizer and convergence settings
├── src/
│   ├── input_loader.py               # YAML input loading
│   ├── wing_geometry.py              # Wing reference geometry
│   ├── avl_geometry.py               # AVL geometry construction
│   ├── flight_case.py                # AVL operating case and trim
│   ├── plotter.py                    # Wing and AVL-result visualization
│   └── optimization/
│       ├── variables.py              # Design-variable mapping/normalization
│       ├── metrics.py                # Aerodynamic/geometric performance metrics
│       ├── objectives.py             # Objective evaluation
│       ├── constraints.py            # Constraint residuals and feasibility
│       ├── evaluator.py              # One complete candidate AVL evaluation
│       ├── problem.py                # Optimization problem and caching
│       ├── optimizer.py              # COBYLA, DE, CMA-ES
│       └── plotter.py                # Optimization convergence plots
├── tests/
│   ├── 01_test_run_model.py
│   ├── 02_test_problem_definition.py
│   └── 03_test_optimizer.py
└── outputs/                         # Generated figures/results
```

The project may also contain a local `avlwrapper/` copy. If so, consult that copy's own copyright notices and license before distributing or modifying it.

## Requirements and installation

- Python 3.10+ recommended (use a version compatible with your installed dependencies).
- An **AVL executable** that is compatible with the selected AVLWrapper version and accessible through its configuration or executable search path.
- Python packages: `numpy`, `scipy`, `PyYAML`, `matplotlib`, `avlwrapper`; `cma` is required for CMA-ES.

From the project root:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell
python -m pip install numpy scipy pyyaml matplotlib avlwrapper cma
```

If you are using the project's local `avlwrapper/` implementation, ensure the local code and AVL executable are compatible; installing an additional PyPI version may not be necessary. Current upstream AVLWrapper versions can have different AVL executable version requirements, so check the documentation for the **version actually used**.

## Configuration

### Geometry and dimensions

The baseline example is a symmetric wing modeled through the right semi-wing and mirrored in AVL. The geometry configuration uses:

- `semi_span`: **half** the full wingspan, in meters;
- `root_chord`: dimensional root chord, in meters;
- `sweep_le`: leading-edge sweep angle, in degrees (negative means forward sweep);
- `dihedral`: dihedral angle, in degrees;
- `sections[].y`: spanwise coordinate normalized by semi-span, from 0 to 1;
- `sections[].chord`: local chord normalized by root chord;
- `sections[].twist`: local twist angle, in degrees.

For the symmetric configuration, full span is `Bref = 2 * semi_span`. As a result, a **2–8 m full-span design range** corresponds to **1–4 m semi-span bounds**. Keep the distinction explicit when editing `design_variables.yaml` and setting constraints on `Bref`.

### Flight condition and lift trim

The flight-condition YAML defines airspeed, air density, initial controls, and the optional trim condition. In the current workflow, AVL can trim the angle of attack to match a specified **dimensional lift**. The required lift coefficient is updated for each candidate wing area:

```text
CL_target = Lift_target / (0.5 * density * velocity**2 * Sref)
```

If angle of attack is selected as AVL's trim variable, it must **not** also be an active optimization design variable.

### Optimization

Edit `input/optimization/design_variables.yaml` to activate variables and specify physical bounds. The optimizer internally normalizes each variable to `[0, 1]`.

Choose **one active objective** in `objectives.yaml` and define limits in `constraints.yaml`. Inequality constraints are evaluated as normalized residuals, with nonnegative values indicating satisfaction. An equality constraint uses a near-zero residual and requires an explicit numerical tolerance in the optimizer.

Select the algorithm in `algorithm.yaml`:

```yaml
algorithm:
  method: CMA_ES  # Alternatives: COBYLA, DE
  maxiter: 150
  popsize: 20
  sigma0: 0.3
  seed: 42
  catol: 1.0e-6
  equality_tol: 1.0e-4
  disp: true
```

These are illustrative CMA-ES settings; use the configuration parameters appropriate to the chosen solver. In particular, DE uses mutation/recombination settings and COBYLA uses trust-region settings. CMA-ES stopping behavior should be checked carefully, especially if using a locally modified fitness-ranking or penalty implementation.

## Running analyses and optimization

Run commands from the **project root**, so input paths resolve correctly.

```bash
python tests/01_test_run_model.py
python tests/02_test_problem_definition.py
python tests/03_test_optimizer.py
```

- `01_test_run_model.py` solves the AVL model and related plots.
- `02_test_problem_definition.py` checks the optimization configuration, baseline evaluation, cache, perturbation response, and input immutability.
- `03_test_optimizer.py` runs the selected optimizer, reports the best feasible design, plots the recorded optimization history, and rebuilds the final aircraft for AVL-based visualization.

The exact script behavior follows the local version of each test file.

### Programmatic interface

```python
from src import load_inputs
from src.optimization.problem import OptimizationProblem
from src.optimization.optimizer import run_optimization, print_optimization_summary
from src.optimization.plotter import plot_optimization_history

geometry, mesh, controls, flight = load_inputs()

baseline = {
    "geometry": geometry,
    "mesh": mesh,
    "controls": controls,
    "flight": flight,
}

problem = OptimizationProblem(baseline)
result = run_optimization(problem)
print_optimization_summary(result, problem)

plot_optimization_history(
    result,
    problem,
    show=True,
    save_dir="outputs/optimization",
)
```

The `best_feasible` entry is the best recorded candidate satisfying the optimizer's feasibility tolerances; it is **not necessarily the final population member or a globally optimal design**. `result["history"]` contains valid, distinct recorded evaluations. Its horizontal axis represents **evaluations**, not necessarily solver iterations or generations.

## Outputs and interpretation

The analysis reports geometric quantities (`Sref`, `Bref`, `Cref`, `AR`), aerodynamic coefficients (`CL`, `CD`, `CDi`, `e`), dimensional forces (`lift`, `drag`, `induced_drag`), trimmed angle of attack, and other available metrics. Optional spanwise strip data are used to estimate wing-root bending moment.

Optimization figures include objective evolution, physical design-variable histories, and constraint histories. Existing wing plots use the modeled planform and the actual AVL results; AVL mesh visualization, where needed, should use AVL's own geometry display rather than a separately reconstructed approximation.

**Modeling limitations:**

- AVL is primarily a linear potential-flow/vortex-lattice analysis method. It does not, by itself, establish viscous drag, stall onset, separated-flow behavior, aeroelastic response, or structural feasibility.
- The present drag objective can favor extreme sweep, span, taper, or twist when missing physical effects are not constrained or modeled.
- The current root-bending estimate is based on aerodynamic strip loads and is not a complete structural load-case calculation.
- Optimizer convergence does not prove global optimality. Compare different starting points/seeds and verify unusual designs independently.
- Numerical feasibility can depend on specified constraint tolerances; inspect physical values and margins, not only a solver's success flag.

## Acknowledgements and third-party software

**AVL — Athena Vortex Lattice.** Aerodynamic analysis is performed by AVL, developed by **Harold Youngren and Mark Drela**. AVL originated in Youngren's work in the MIT Athena software collection and has been developed further by Youngren and Drela. The present repository uses AVL as its underlying aerodynamic solver; the Python optimization framework is not an implementation or replacement of AVL.

- AVL documentation: https://web.mit.edu/drela/Public/web/avl/
- AVL User Primer: https://web.mit.edu/drela/Public/web/avl/AVL_User_Primer.pdf

**AVLWrapper.** Python-based geometry definition, case execution, and result extraction use **AVLWrapper**, authored and maintained by **Reno Elmendorp**. Credit and licensing for AVLWrapper remain with its contributors. This project builds a separate, configuration-driven analysis and optimization layer on top of that interface.

- AVLWrapper repository: https://github.com/renoelmendorp/AVLWrapper
- AVLWrapper package: https://pypi.org/project/avlwrapper/

The optimizers and numerical utilities also rely on **SciPy** and, when CMA-ES is selected, **pycma** (`cma`). The project acknowledges the authors and maintainers of these external tools.

### Licensing

This README does **not** assign a license to the original project code. If distributing the repository, add an explicit license for your own code and preserve the respective notices and license terms for AVL, AVLWrapper, and any bundled third-party files. In particular, the published AVLWrapper package identifies the **GNU GPL v3** as its license; check the license of the specific version incorporated into this project.
