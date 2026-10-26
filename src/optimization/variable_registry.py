
"""Supported design variables for the single half-wing model."""

# Wing section variables.
# Currently assumes root section 0 and tip section 1.
GEOMETRY_VARIABLES = {
    "half_span": "sections[1].y",
    "root_chord": "sections[0].chord",
    "tip_chord": "sections[1].chord",
    "root_twist": "sections[0].twist",
    "tip_twist": "sections[1].twist",
    "tip_x_le": "sections[1].x_le",
    "tip_z_le": "sections[1].z_le",
}

# Control geometry variables, selected by CONTROL NAME.
CONTROL_VARIABLES = {
    "start": ("span", "start"),
    "end": ("span", "end"),
    "hinge_root": ("hinge", "root_chord_fraction"),
    "hinge_tip": ("hinge", "tip_chord_fraction"),
}

# Variables for the single operating point.
FLIGHT_VARIABLES = {
    "alpha",
    "beta",
    "velocity",
    "density",
}
