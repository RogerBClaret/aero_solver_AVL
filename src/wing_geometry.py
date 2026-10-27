
"""
Wing geometry calculations.

Input conventions:
- y: normalized semi-span position [0,1]
- chord: local chord / root chord
- twist: geometric incidence [deg]
- sweep_le: leading-edge sweep [deg]
- dihedral: angle above horizontal [deg]

All dimensional outputs use SI units.
"""

from math import radians, tan, isfinite


def validate_sections(sections):
    """Validate and sort user-defined wing sections."""

    sections = sorted(sections, key=lambda s: float(s["y"]))

    if len(sections) < 2:
        raise ValueError("At least two sections are required.")

    ids = [s["section_id"] for s in sections]
    ys = [float(s["y"]) for s in sections]

    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Section IDs must be nonnegative integers.")

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate section IDs.")

    if abs(ys[0]) > 1e-10 or abs(ys[-1] - 1) > 1e-10:
        raise ValueError("Sections must span y=0 to y=1.")

    if any(b <= a for a, b in zip(ys[:-1], ys[1:])):
        raise ValueError("Section positions must strictly increase.")

    for section in sections:
        for key in ("y", "chord", "twist"):
            if not isfinite(float(section[key])):
                raise ValueError(f"Invalid section {key}.")

        if float(section["chord"]) <= 0:
            raise ValueError("Section chord must be positive.")

    return sections


def interpolate_section(sections, eta):
    """
    Interpolate chord ratio and twist at normalized span eta.

    section_id is intentionally not interpolated: additional
    AVL sections do not represent user-defined design sections.
    """

    if not 0 <= eta <= 1:
        raise ValueError("Normalized span must be in [0,1].")

    for a, b in zip(sections[:-1], sections[1:]):
        ya, yb = float(a["y"]), float(b["y"])

        if ya <= eta <= yb:
            t = (eta - ya) / (yb - ya)

            return {
                "y": eta,
                "chord": float(a["chord"]) + t * (
                    float(b["chord"]) - float(a["chord"])
                ),
                "twist": float(a["twist"]) + t * (
                    float(b["twist"]) - float(a["twist"])
                ),
            }

    raise ValueError(f"Cannot interpolate section at {eta}.")


def section_coordinates(planform, section):
    """
    Convert a normalized section to dimensional geometry.

    Sweep and dihedral are constant along the semi-span.
    """

    bs = float(planform["semi_span"])
    cr = float(planform["root_chord"])

    sweep = radians(float(planform["sweep_le"]))
    dihedral = radians(float(planform["dihedral"]))

    y = bs * float(section["y"])

    return {
        "x_le": y * tan(sweep),
        "y_le": y,
        "z_le": y * tan(dihedral),
        "chord": cr * float(section["chord"]),
        "twist": float(section["twist"]),
    }


def calculate_reference(planform, sections):
    """
    Calculate full-wing aerodynamic reference quantities.

    Sref = 2 * integral(c dy)
    Bref = 2 * semi_span
    Cref = MAC = integral(c² dy) / integral(c dy)

    The reference point is placed at the quarter-MAC
    aerodynamic reference location, not at the aircraft CG.

    Each section interval has linearly varying chord.
    """

    sections = validate_sections(sections)

    bs = float(planform["semi_span"])
    cr = float(planform["root_chord"])
    sweep = radians(float(planform["sweep_le"]))
    dihedral = radians(float(planform["dihedral"]))

    if not isfinite(bs) or not isfinite(cr) or bs <= 0 or cr <= 0:
        raise ValueError("Semi-span and root chord must be positive.")

    if not all(isfinite(v) for v in (sweep, dihedral)):
        raise ValueError("Sweep and dihedral must be finite.")

    if abs(sweep) >= radians(90) or abs(dihedral) >= radians(90):
        raise ValueError("Invalid sweep or dihedral angle.")

    area_half = 0.0
    c2_integral = 0.0
    yc_integral = 0.0

    for a, b in zip(sections[:-1], sections[1:]):
        y0 = bs * float(a["y"])
        y1 = bs * float(b["y"])

        c0 = cr * float(a["chord"])
        c1 = cr * float(b["chord"])

        dy = y1 - y0

        # Integrals for a linearly tapered panel.
        area_half += dy * (c0 + c1) / 2
        c2_integral += dy * (
            c0**2 + c0 * c1 + c1**2
        ) / 3

        # First spanwise moment of chord distribution.
        yc_integral += dy / 6 * (
            y0 * (2*c0 + c1)
            + y1 * (c0 + 2*c1)
        )

    sref = 2 * area_half
    bref = 2 * bs
    cref = c2_integral / area_half

    # Chord-weighted span station of the MAC.
    y_mac = yc_integral / area_half

    # Quarter-chord position at the MAC station.
    xref = y_mac * tan(sweep) + 0.25 * cref
    zref = y_mac * tan(dihedral)

    return {
        "Sref": sref,
        "Bref": bref,
        "Cref": cref,
        "AR": bref**2 / sref,
        "MAC_y": y_mac,
        "reference_point": (xref, 0.0, zref),
    }

