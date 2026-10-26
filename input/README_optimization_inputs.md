# Optimization input files

This guide describes how to configure the **current single-operating-point, half-wing AVL optimizer**. Place it in `input/README.md`.

## Files

| File | Purpose |
|---|---|
| `geometry.yaml` | Baseline half-wing sections and reference point |
| `mesh.yaml` | AVL discretization |
| `control_surfaces.yaml` | Baseline control geometry and names |
| `flight_condition.yaml` | The single operating point and optional AVL trim |
| `optimization_variables.yaml` | **Only the variables allowed to change** and their bounds |
| `optimization.yaml` | Optimization objectives and constraints |
| `optimization_algorithm.yaml` | Algorithm configuration (used when `optimizer.py` is implemented) |

**Rules:** One AVL `Surface`, one `Case` named `operating_point`, and no mirrored wing (`geometry.symmetric: false`). Full-wing geometric quantities are computed analytically from the half-wing geometry; aerodynamic coefficients come from the isolated half-wing calculation.

## 1. Selecting active design variables

The current `variables.py` expects **grouped mappings**, not a list. Only the entries present in `optimization_variables.yaml` are optimized; anything omitted stays at its value in the baseline input files. Initial values are read automatically from those files. Bounds include the initial value. The `scale` key is optional and is used only if normalized design vectors are requested.

Example `optimization_variables.yaml`:

```yaml
design_variables:
  geometry:
    tip_chord:
      bounds: [0.5, 1.5]
    tip_twist:
      bounds: [-6.0, 2.0]

  controls:
    aileron:
      start:
        bounds: [0.60, 0.85]
      end:
        bounds: [0.85, 1.00]
      hinge_root:
        bounds: [0.65, 0.90]
      hinge_tip:
        bounds: [0.65, 0.90]

  flight:
    alpha:
      bounds: [-3.0, 12.0]
    # velocity:
    #   bounds: [50.0, 150.0]
    # controls:
    #   inboard:
    #     bounds: [0.0, 25.0]
    #   aileron:
    #     bounds: [-15.0, 15.0]
```

### Supported variables (current `variable_registry.py`)

| Group | Available names | Meaning |
|---|---|---|
| `geometry` | `half_span` | Tip-section y coordinate, metres |
| | `root_chord`, `tip_chord` | Section chords, metres |
| | `root_twist`, `tip_twist` | Section twist angles, degrees |
| | `tip_x_le`, `tip_z_le` | Tip leading-edge coordinates, metres |
| `controls.<control_name>` | `start`, `end` | Span fractions from 0 at root to 1 at tip |
| | `hinge_root`, `hinge_tip` | Local chord fractions for hinge |
| `flight` | `alpha`, `beta` | Angles, degrees |
| | `velocity` | Flight speed, m/s |
| | `density` | Air density, kg/m³ |
| `flight.controls` | Any control name present in `control_surfaces.yaml` | Control deflection, degrees |

The geometry registry currently references `sections[0]` (root) and `sections[1]` (tip). It assumes **two baseline geometry sections**. Names under `controls` must match the `name` keys in `control_surfaces.yaml` (e.g. `inboard`, `aileron`).

**Not currently supported as direct design variables:** `full_area`, `aspect_ratio`, `taper_ratio`, `sweep_LE`, `dihedral`, and `trim.target`. They *are* available as metrics or could be added to the variable registry later. To alter them now, select the underlying span, chord, or leading-edge coordinates.

### Fixed vs optimized vs trimmed

- **Fixed:** Omit the parameter from `optimization_variables.yaml`; its baseline YAML value stays unchanged.
- **Optimized:** Add the parameter and `[lower, upper]` bounds.
- **Trimmed:** Specify it under `flight_condition.trim`. AVL solves it internally. Do **not** also list it as an active design variable.

Example of a fixed-speed, AVL-trimmed-AoA operating point in `flight_condition.yaml`:

```yaml
flight_condition:
  velocity: 100.0
  density: 1.225
  alpha: 5.0
  controls:
    inboard: 0.0
    aileron: 0.0
  trim:
    enabled: true
    variable: alpha
    constraint: CL
    target: 0.5
```

Here `flight.alpha` must **not** appear under `design_variables`. The current `build_case()` supports at most **one trim variable** at a time: `alpha` or a defined control name. Trim target optimization is **not implemented yet**. The constraint must be supported by AVL for the selected trim variable.

## 2. Objectives and constraints

Example `optimization.yaml`:

```yaml
objectives:
  - metric: CDi
    direction: minimize
    weight: 1.0
    scale: 0.01

constraints:
  - metric: CL
    operator: ">="
    value: 0.20
  - metric: full_area
    operator: "<="
    value: 18.0
  - metric: aspect_ratio
    operator: ">="
    value: 5.0
  - metric: dCl_ddeg.aileron
    operator: "<="
    value: -0.001
```

### Objectives

Each entry needs `metric` and `direction`:

| Field | Allowed values / behavior |
|---|---|
| `metric` | Registered scalar metric (table below) |
| `direction` | `minimize`, `maximize`, or `target` |
| `target` | Required when `direction: target` |
| `weight` | Optional nonnegative multiplier; default `1.0` |
| `scale` | Optional **positive objective normalization factor**; default `1.0` |

The evaluator creates **one weighted scalar objective**:

- Minimize: `weight * metric / scale`
- Maximize: `-weight * metric / scale`
- Target: `weight * ((metric - target) / scale)**2`

`scale` changes the optimizer's numerical objective, **not** the aerodynamic quantity or its units. When minimizing a single objective, it can usually be omitted. For multiple objectives, select scales and weights deliberately.

### Constraints

Each entry requires `metric`, `operator`, and `value`.

| Operator | Condition | Evaluator residual |
|---|---|---|
| `>=` | metric ≥ value | `metric - value` |
| `<=` | metric ≤ value | `value - metric` |
| `==` | metric = value | `metric - value` |

Inequality residuals **≥ 0** are feasible; equality residuals must be **0**. The current evaluator uses a tolerance of `1e-8` when reporting satisfaction. It doesn't automatically scale constraints.

### Supported metric names (current `metric_registry.py`)

**Geometry (full-wing equivalents where indicated):**

| Metric | Meaning |
|---|---|
| `half_span`, `half_area` | Modeled half-wing dimensions (m, m²) |
| `full_span`, `full_area` | Twice half-span and half-area (m, m²); not a full-wing AVL run |
| `aspect_ratio` | `full_span**2 / full_area` |
| `taper_ratio` | Tip chord / root chord |
| `sweep_LE` | Root-to-tip leading-edge sweep, degrees |
| `dihedral` | Root-to-tip dihedral, degrees |
| `washout` | Root twist − tip twist, degrees |

**Aerodynamics (isolated half-wing AVL results):**

| Metric | Meaning |
|---|---|
| `CL` | Lift coefficient |
| `CDi` | Induced drag coefficient |
| `Cm` | Pitching moment coefficient |
| `Cl` | Rolling moment coefficient (lowercase l) |
| `Cn` | Yawing moment coefficient |
| `CL_CDi` | Lift / induced-drag coefficient ratio |

**Control effectiveness:** For each named control (e.g., `aileron` or `inboard`), use:

```text
dCL_ddeg.aileron
dCD_ddeg.aileron
dCY_ddeg.aileron
dCl_ddeg.aileron
dCm_ddeg.aileron
dCn_ddeg.aileron
```

Replace `aileron` with any defined control name. These are **local derivatives in coefficient per degree** from AVL stability derivatives at the single operating point. They are not full-aircraft control-authority predictions. `CD` here is a derivative key from AVL, not a declaration that total viscous drag is modeled. Use `dCl_ddeg.aileron` for rolling moment sensitivity, not `dCL_ddeg.aileron` (lift sensitivity).

**Important:** `CL_CDi` excludes profile/viscous drag. AVL's half-wing aerodynamic solution should not be interpreted as the full symmetric wing solution. Changes in wing area alter dimensional drag at a given dynamic pressure; minimizing `CDi` alone is not the same as minimizing total drag force.

## 3. Algorithm configuration

The planned `optimization_algorithm.yaml` format is:

```yaml
optimizer:
  algorithm: SLSQP
  max_iterations: 150
  tolerance: 1.0e-6
  display: true
  seed: 42
```

Intended choices: `SLSQP`, `COBYLA`, and `differential_evolution`. **The algorithm dispatcher (`optimizer.py`) has not yet been implemented**, so this file currently does not start optimization or change evaluator behavior.

## 4. Evaluation workflow

For each candidate vector, `Evaluator.evaluate(x)`:

1. Copies the baseline YAML input dictionaries and applies only selected design variables.
2. Validates geometry/control placement; invalid geometries raise an error.
3. Builds **one** half-wing and **one** AVL operating point.
4. Runs AVL and queries only the metrics requested by objectives and constraints.
5. Returns the scalar weighted objective, individual metric values, constraint residuals, and a feasible/infeasible flag.

To test the setup:

```bash
python tests/test_variables.py
python tests/test_metrics.py
python tests/test_evaluator.py
```

The evaluator currently caches **only the most recently evaluated** design vector and does not itself perform an optimization search. For changing wing geometry, `build_geometry()` recomputes reference area from the half-wing, reference span as the full-span equivalent, and reference chord as MAC.
