
"""
Design-variable management for wing optimization.

Active variables are defined in design_variables.yaml.
Commented variables remain fixed at their baseline values.

Optimizer coordinates are normalized to [0, 1].
Baseline input dictionaries are never modified.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


# ============================================================
# 1. VARIABLE LOCATIONS
# ============================================================

# Map scalar variable names to their original YAML inputs.
# Format: variable_name -> (input_file, parent_keys, parameter)

SCALAR_VARIABLES = {
    "semi_span": ("geometry", ("surfaces", 0, "planform"), "semi_span"),
    "root_chord": ("geometry", ("surfaces", 0, "planform"), "root_chord"),
    "sweep_le": ("geometry", ("surfaces", 0, "planform"), "sweep_le"),
    "dihedral": ("geometry", ("surfaces", 0, "planform"), "dihedral"),

    "alpha": ("flight", ("flight_condition",), "alpha"),
    "velocity": ("flight", ("flight_condition",), "velocity"),
    "density": ("flight", ("flight_condition",), "density"),
}


# Variables requiring a section_id or control name.
SECTION_VARIABLES = {
    "section_y": "y",
    "section_chord": "chord",
    "section_twist": "twist",
}

CONTROL_VARIABLES = {
    "control_span_start": ("span", "start"),
    "control_span_end": ("span", "end"),
    "control_hinge_root": ("hinge", "root_chord_fraction"),
    "control_hinge_tip": ("hinge", "tip_chord_fraction"),
    "control_deflection": None,
}


# ============================================================
# 2. LOAD DESIGN VARIABLES
# ============================================================

def load_variables(path):
    """Load and validate the active design-variable list."""

    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    variables = config["design_variables"]

    if not isinstance(variables, list) or not variables:
        raise ValueError("At least one active variable is required.")

    identifiers = set()

    for var in variables:
        name = var["name"]
        bounds = var["bounds"]

        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
        ):
            raise ValueError(f"Invalid bounds for {name}.")

        lower, upper = map(float, bounds)

        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"Invalid bounds for {name}.")

        if name in SECTION_VARIABLES:
            identifier = (name, var["section_id"])

        elif name in CONTROL_VARIABLES:
            identifier = (name, var["control"])

        elif name in SCALAR_VARIABLES:
            identifier = (name,)

        else:
            raise ValueError(f"Unknown design variable: {name}")

        if identifier in identifiers:
            raise ValueError(f"Duplicate variable: {identifier}")

        identifiers.add(identifier)

    return variables


# ============================================================
# 3. RESOLVE VARIABLE LOCATION
# ============================================================

def _resolve(configs, var):
    """
    Return (parent_dictionary, key) for a design variable.

    This enables both reading and modifying variable values.
    """

    name = var["name"]

    if name in SCALAR_VARIABLES:
        source, path, key = SCALAR_VARIABLES[name]
        parent = configs[source]

        for part in path:
            parent = parent[part]

        return parent, key

    if name in SECTION_VARIABLES:
        sections = configs["geometry"]["surfaces"][0]["sections"]
        section_id = var["section_id"]

        matches = [
            s for s in sections
            if s["section_id"] == section_id
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Section ID {section_id} not found uniquely."
            )

        return matches[0], SECTION_VARIABLES[name]

    if name in CONTROL_VARIABLES:
        control_name = var["control"]

        if name == "control_deflection":
            parent = configs["flight"]["flight_condition"]["controls"]

            if control_name not in parent:
                raise ValueError(
                    f"Control deflection not found: {control_name}"
                )

            return parent, control_name

        controls = configs["controls"]["control_surfaces"]

        matches = [
            c for c in controls
            if c["name"] == control_name
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Control '{control_name}' not found uniquely."
            )

        parent = matches[0]
        path = CONTROL_VARIABLES[name]

        for part in path[:-1]:
            parent = parent[part]

        return parent, path[-1]

    raise ValueError(f"Unsupported variable: {name}")


# ============================================================
# 4. DESIGN VECTOR OPERATIONS
# ============================================================

def get_initial_vector(variables, configs):
    """Return baseline design-variable values."""

    return np.array(
        [
            float(_resolve(configs, var)[0][_resolve(configs, var)[1]])
            for var in variables
        ],
        dtype=float,
    )


def get_bounds(variables):
    """Return physical design-variable bounds."""

    return np.array(
        [var["bounds"] for var in variables],
        dtype=float,
    )


def normalize(values, bounds):
    """Convert physical variables to normalized [0,1] values."""

    return (np.asarray(values) - bounds[:, 0]) / (
        bounds[:, 1] - bounds[:, 0]
    )


def denormalize(values, bounds):
    """Convert normalized variables to physical values."""

    return bounds[:, 0] + np.asarray(values) * (
        bounds[:, 1] - bounds[:, 0]
    )


def apply_variables(variables, values, configs):
    """
    Apply physical design values to independent input copies.

    Returns modified configurations without changing baseline.
    """

    values = np.asarray(values, dtype=float)

    if len(values) != len(variables):
        raise ValueError("Design-vector length mismatch.")

    if not np.isfinite(values).all():
        raise ValueError("Design vector contains nonfinite values.")

    updated = deepcopy(configs)

    for var, value in zip(variables, values):
        parent, key = _resolve(updated, var)
        parent[key] = float(value)

    return updated
