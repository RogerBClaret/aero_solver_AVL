
"""Apply active design variables to the single-case AVL inputs.

Inputs: (geometry, mesh, controls, flight)
Only the half-wing is modeled. Original inputs are not modified.
"""

import copy
from pathlib import Path

import numpy as np
import yaml

from .variable_registry import (
    GEOMETRY_VARIABLES,
    CONTROL_VARIABLES,
    FLIGHT_VARIABLES,
)

ROOT = Path(__file__).resolve().parents[2]


def _find(items, name):
    matches = [item for item in items if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one item named '{name}'")
    return matches[0]


def _get_section(geometry, path):
    """Resolve e.g. sections[1].chord."""
    section_part, field = path.split(".")
    index = int(section_part.split("[")[1].split("]")[0])
    section = geometry["surfaces"][0]["sections"][index]
    return section, field


def _register(variables, name, spec, initial):
    if not isinstance(spec, dict) or "bounds" not in spec:
        raise ValueError(f"{name}: missing bounds")

    bounds = spec["bounds"]
    if len(bounds) != 2:
        raise ValueError(f"{name}: expected [lower, upper]")

    lower, upper = map(float, bounds)
    initial = float(initial)
    scale = float(spec.get("scale", 1.0))

    if not np.isfinite(
        [lower, upper, initial, scale]
    ).all():
        raise ValueError(f"{name}: non-finite value")

    if lower >= upper:
        raise ValueError(f"{name}: invalid bounds")

    if not lower <= initial <= upper:
        raise ValueError(
            f"{name}: baseline {initial} outside bounds"
        )

    if scale <= 0:
        raise ValueError(f"{name}: scale must be positive")

    variables.append({
        "name": name,
        "initial": initial,
        "bounds": (lower, upper),
        "scale": scale,
    })


def load_variables(path=None, inputs=None):
    """Load only active variables from optimization YAML."""

    if inputs is None:
        from src.build_model import load_inputs
        inputs = load_inputs()

    path = Path(path) if path else (
        ROOT / "input" / "optimization_variables.yaml"
    )

    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    selected = config.get("design_variables", {})
    if not isinstance(selected, dict):
        raise ValueError("design_variables must be a mapping")

    geometry, _, controls, flight_data = inputs
    flight = flight_data["flight_condition"]

    trim = flight.get("trim", {})
    trimmed = (
        trim.get("variable")
        if trim.get("enabled", False)
        else None
    )

    variables = []

    for group in selected:
        if group not in ("geometry", "controls", "flight"):
            raise ValueError(f"Unknown variable group: {group}")

    # Wing geometry
    for name, spec in selected.get("geometry", {}).items():
        if name not in GEOMETRY_VARIABLES:
            raise ValueError(f"Unknown geometry variable: {name}")

        obj, field = _get_section(
            geometry, GEOMETRY_VARIABLES[name]
        )
        _register(variables, f"geometry.{name}", spec, obj[field])

    # Flap / aileron geometry
    for control_name, settings in selected.get(
        "controls", {}
    ).items():
        control = _find(
            controls["control_surfaces"], control_name
        )

        for name, spec in settings.items():
            if name not in CONTROL_VARIABLES:
                raise ValueError(
                    f"Unknown control geometry variable: {name}"
                )

            group, field = CONTROL_VARIABLES[name]
            initial = control[group][field]

            _register(
                variables,
                f"controls.{control_name}.{name}",
                spec,
                initial
            )

    # Single operating point
    for name, spec in selected.get("flight", {}).items():

        if name == "controls":
            for control_name, control_spec in spec.items():
                _find(
                    controls["control_surfaces"], control_name
                )

                if trimmed == control_name:
                    raise ValueError(
                        f"{control_name} is trimmed by AVL "
                        "and cannot be optimized"
                    )

                initial = flight.get(
                    "controls", {}
                ).get(control_name, 0.0)

                _register(
                    variables,
                    f"flight.controls.{control_name}",
                    control_spec,
                    initial
                )

        else:
            if name not in FLIGHT_VARIABLES:
                raise ValueError(f"Unknown flight variable: {name}")

            if name == trimmed:
                raise ValueError(
                    f"{name} is trimmed by AVL "
                    "and cannot be optimized"
                )

            if name not in flight:
                raise ValueError(
                    f"Missing baseline flight value: {name}"
                )

            _register(
                variables,
                f"flight.{name}",
                spec,
                flight[name]
            )

    names = [v["name"] for v in variables]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate optimization variable")

    return variables


def initial_vector(variables, normalized=False):
    x = np.array(
        [v["initial"] for v in variables],
        dtype=float
    )

    if normalized:
        x /= np.array(
            [v["scale"] for v in variables]
        )

    return x


def variable_bounds(variables, normalized=False):
    bounds = np.array(
        [v["bounds"] for v in variables],
        dtype=float
    ).reshape(-1, 2)

    if normalized:
        bounds /= np.array(
            [v["scale"] for v in variables]
        )[:, None]

    return bounds


def validate_design(geometry, controls, flight_data):
    """Reject invalid candidate designs before running AVL."""

    if geometry["geometry"].get("symmetric", False):
        raise ValueError("Only half-wing analysis is supported")

    sections = sorted(
        geometry["surfaces"][0]["sections"],
        key=lambda s: s["y"]
    )

    if len(sections) < 2 or sections[0]["y"] != 0:
        raise ValueError("Invalid half-wing sections")

    for a, b in zip(sections[:-1], sections[1:]):
        if b["y"] <= a["y"]:
            raise ValueError("Wing section y must increase")
        if a["chord"] <= 0 or b["chord"] <= 0:
            raise ValueError("Wing chords must be positive")

    items = controls["control_surfaces"]

    for ctrl in items:
        start = ctrl["span"]["start"]
        end = ctrl["span"]["end"]

        if not 0 <= start < end <= 1:
            raise ValueError(
                f"Invalid span limits for {ctrl['name']}"
            )

        for fraction in ctrl["hinge"].values():
            if not 0 < fraction < 1:
                raise ValueError(
                    f"Invalid hinge for {ctrl['name']}"
                )

    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a["surface"] != b["surface"]:
                continue

            overlap = (
                max(a["span"]["start"], b["span"]["start"])
                < min(a["span"]["end"], b["span"]["end"])
            )

            if overlap:
                raise ValueError(
                    f"Controls overlap: {a['name']}, {b['name']}"
                )

    flight = flight_data["flight_condition"]

    if flight["velocity"] <= 0 or flight["density"] <= 0:
        raise ValueError("Velocity and density must be positive")


def apply_variables(x, variables, inputs, normalized=False):
    """Return modified (geometry, mesh, controls, flight) copies."""

    x = np.asarray(x, dtype=float)

    if len(x) != len(variables):
        raise ValueError("Design vector length mismatch")

    modified = copy.deepcopy(inputs)
    geometry, mesh, controls, flight_data = modified
    flight = flight_data["flight_condition"]

    for value, variable in zip(x, variables):
        value = float(value)

        if normalized:
            value *= variable["scale"]

        lower, upper = variable["bounds"]

        if not np.isfinite(value):
            raise ValueError(f"Non-finite {variable['name']}")

        if not lower - 1e-9 <= value <= upper + 1e-9:
            raise ValueError(
                f"{variable['name']} outside bounds"
            )

        parts = variable["name"].split(".")

        if parts[0] == "geometry":
            obj, field = _get_section(
                geometry, GEOMETRY_VARIABLES[parts[1]]
            )
            obj[field] = value

        elif parts[0] == "controls":
            ctrl = _find(
                controls["control_surfaces"], parts[1]
            )
            group, field = CONTROL_VARIABLES[parts[2]]
            ctrl[group][field] = value

        elif parts[0] == "flight":
            if parts[1] == "controls":
                flight.setdefault("controls", {})[
                    parts[2]
                ] = value
            else:
                flight[parts[1]] = value

    validate_design(geometry, controls, flight_data)

    # Reference quantities are recalculated by build_geometry().
    # No extra AVL case and no mirrored wing.
    return geometry, mesh, controls, flight_data
