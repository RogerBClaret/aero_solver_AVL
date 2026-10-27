
"""
End-to-end AVL model test.

Run from the project root:
    python tests/01_test_run_model.py

Checks:
1. YAML input loading
2. Wing geometry and reference quantities
3. AVL model construction
4. Flight-case construction
5. AVL execution
6. Aerodynamic results

Requires the AVL executable and avlwrapper.
"""

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

import sys
from pathlib import Path
from math import isclose

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from avlwrapper import Session
from src import load_inputs, build_geometry, build_case, plot_all
from src.wing_geometry import (
    validate_sections,
    section_coordinates,
    calculate_reference,
)


def header(title):
    """Print a readable section heading."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def main():

    # ========================================================
    # 1. LOAD INPUTS
    # ========================================================

    header("1. LOADING INPUT FILES")

    geometry_data, mesh_data, controls_data, flight_data = (
        load_inputs()
    )

    wing = geometry_data["surfaces"][0]
    planform = wing["planform"]
    controls = controls_data["control_surfaces"]
    flight = flight_data["flight_condition"]

    print(f"Aircraft: {geometry_data['geometry']['name']}")
    print(f"Wing:     {wing['name']}")
    print(f"Airfoil:  {wing['airfoil']['type']}")
    print(f"Controls: {[c['name'] for c in controls]}")
    print("YAML inputs loaded successfully.")

    # ========================================================
    # 2. RECONSTRUCT WING GEOMETRY
    # ========================================================

    header("2. WING GEOMETRY")

    sections = validate_sections(wing["sections"])

    print(f"Semi-span:  {planform['semi_span']:.3f} m")
    print(f"Root chord: {planform['root_chord']:.3f} m")
    print(f"LE sweep:   {planform['sweep_le']:.3f} deg")
    print(f"Dihedral:   {planform['dihedral']:.3f} deg")
    print(f"User sections: {len(sections)}")

    print(
        "\n ID    eta      X_LE      Y_LE      Z_LE"
        "      chord     twist"
    )

    for section in sections:
        p = section_coordinates(planform, section)

        print(
            f"{section['section_id']:3d}"
            f"  {section['y']:6.3f}"
            f"  {p['x_le']:8.3f}"
            f"  {p['y_le']:8.3f}"
            f"  {p['z_le']:8.3f}"
            f"  {p['chord']:9.3f}"
            f"  {p['twist']:8.3f}"
        )

    # ========================================================
    # 3. REFERENCE QUANTITIES
    # ========================================================

    header("3. AERODYNAMIC REFERENCE QUANTITIES")

    ref = calculate_reference(planform, sections)

    print(f"Wing area (Sref):    {ref['Sref']:.4f} m²")
    print(f"Full span (Bref):    {ref['Bref']:.4f} m")
    print(f"MAC (Cref):          {ref['Cref']:.4f} m")
    print(f"Aspect ratio:        {ref['AR']:.4f}")
    print(f"MAC span station:    {ref['MAC_y']:.4f} m")
    print(f"Reference point:     {ref['reference_point']}")

    # Regression checks for the current baseline geometry.
    # Update these if the baseline design changes.
    assert isclose(ref["Sref"], 6.0, rel_tol=1e-6)
    assert isclose(ref["Bref"], 8.0, rel_tol=1e-6)
    assert isclose(ref["Cref"], 7 / 9, rel_tol=1e-6)
    assert isclose(ref["AR"], 32 / 3, rel_tol=1e-6)

    print("\n[PASS] Baseline reference quantities.")

    # ========================================================
    # 4. BUILD AVL GEOMETRY
    # ========================================================

    header("4. BUILDING AVL MODEL")

    # Control boundaries create additional AVL sections.
    stations = sorted(
        {float(s["y"]) for s in sections}
        | {
            float(value)
            for c in controls
            for value in (
                c["span"]["start"],
                c["span"]["end"],
            )
        }
    )

    print(f"Original sections: {len(sections)}")
    print(f"AVL sections:      {len(stations)}")
    print(f"Span stations:     {stations}")

    mesh = mesh_data["mesh"]

    print(
        f"Mesh: {mesh['chordwise_panels']} chordwise x "
        f"{mesh['spanwise_panels']} spanwise"
    )
    print(
        f"Spacing: {mesh['chord_spacing']} / "
        f"{mesh['span_spacing']}"
    )

    geometry = build_geometry(
        geometry_data,
        mesh_data,
        controls_data,
    )

    print(f"AVL geometry object: {type(geometry).__name__}")
    print("[PASS] AVL geometry constructed.")

    # ========================================================
    # 5. BUILD FLIGHT CASE
    # ========================================================

    header("5. FLIGHT CONDITION")

    case = build_case(flight_data, controls_data)

    velocity = float(flight["velocity"])
    density = float(flight["density"])
    q = 0.5 * density * velocity**2

    print(f"Velocity:          {velocity:.3f} m/s")
    print(f"Density:           {density:.4f} kg/m³")
    print(f"Dynamic pressure:  {q:.2f} Pa")
    print(f"Input alpha:       {flight['alpha']:.3f} deg")
    print(f"Control inputs:    {flight['controls']}")
    print(f"Constraint enabled: {flight['trim']['enabled']}")

    if flight["trim"]["enabled"]:
        trim = flight["trim"]
        print(
            f"Constraint: {trim['variable']} -> "
            f"{trim['constraint']} = {trim['target']}"
        )

    print("[PASS] Flight case constructed.")

    # ========================================================
    # 6. RUN AVL
    # ========================================================

    header("6. RUNNING AVL")

    print("Creating AVL session...")

    session = Session(
        geometry=geometry,
        cases=[case],
    )

    print("Executing operating case...")

    # This launches the external AVL solver.
    results = session.get_results()

    # AVLWrapper versions differ in their results interface:
    # recent releases index by case number; older ones may
    # use the case name.
    try:
        result = results[1]
    except (KeyError, IndexError, TypeError):
        result = results["operating_point"]

    # Get the total-force coefficient dictionary.
    totals = (
        result.totals
        if hasattr(result, "totals")
        else result["Totals"]
    )

    print("[PASS] AVL execution completed.")

    # ========================================================
    # 7. AERODYNAMIC RESULTS
    # ========================================================

    header("7. AERODYNAMIC RESULTS")

    print("\nAvailable total-result keys:")
    print(list(totals.keys()))

    # AVL's standard total-force labels.
    cl = float(totals["CLtot"])
    cd = float(totals["CDtot"])
    cdi = float(totals["CDind"])
    cm = float(totals["Cmtot"])

    lift = q * ref["Sref"] * cl
    drag = q * ref["Sref"] * cd
    induced_drag = q * ref["Sref"] * cdi

    print(f"\nCL:                 {cl:10.6f}")
    print(f"CD total (AVL):     {cd:10.6f}")
    print(f"CD induced:         {cdi:10.6f}")
    print(f"Cm:                 {cm:10.6f}")

    print(f"\nLift:               {lift:10.2f} N")
    print(f"Drag (AVL):         {drag:10.2f} N")
    print(f"Induced drag:       {induced_drag:10.2f} N")

    if cd > 0:
        print(f"AVL CL/CD:          {cl / cd:10.3f}")

    # Basic numerical sanity checks.
    from math import isfinite

    assert all(isfinite(x) for x in (cl, cd, cdi, cm))
    assert cdi >= -1e-8, "Unexpected negative induced drag."

    print("\n[PASS] Aerodynamic result sanity checks.")

    # ========================================================
    # 8. RESULT PLOTTING
    # ========================================================
    
    # Generate geometry and aerodynamic distribution plots.
    plot_all(
        geometry_data,
        controls_data,
        flight_data,
        results,
    )

    print("\n Study case plots generated in output folder")

    header("MODEL TEST COMPLETED SUCCESSFULLY")

    # AVL visualization
    session.show_geometry()


def main_short():
    """Run AVL and print a compact aerodynamic summary."""

    ##### CORE START ######

    # 1. Load inputs and construct AVL objects.
    geometry_data, mesh_data, controls_data, flight_data = load_inputs()

    geometry = build_geometry(
        geometry_data, mesh_data, controls_data
    )
    case = build_case(flight_data, controls_data)

    # 2. Run AVL.
    session = Session(geometry=geometry, cases=[case])
    results = session.get_results()

    ##### CORE END ######

    # 3. Extract aerodynamic coefficients.
    try:
        result = results[1]
    except (KeyError, IndexError, TypeError):
        result = results["operating_point"]

    totals = result["Totals"]

    CL = totals["CLtot"]
    CD = totals["CDtot"]
    CDi = totals["CDind"]
    Cm = totals["Cmtot"]
    e = totals["e"]

    # 4. Calculate dimensional aerodynamic forces.
    flight = flight_data["flight_condition"]
    V = float(flight["velocity"])
    rho = float(flight["density"])

    q = 0.5 * rho * V**2

    S = totals["Sref"]
    L = q * S * CL
    D = q * S * CD

    # 5. Print results.
    print("\n" + "=" * 45)
    print(" AVL SIMULATION RESULTS")
    print("=" * 45)

    print(f"Alpha       : {totals['Alpha']:10.4f} deg")
    print(f"CL          : {CL:10.6f}")
    print(f"CD          : {CD:10.6f}")
    print(f"CDi         : {CDi:10.6f}")
    print(f"Cm          : {Cm:10.6f}")
    print(f"Span eff. e : {e:10.4f}")
    print(f"L/D (AVL)   : {CL/CD:10.3f}")

    print("-" * 45)
    print(f"Sref        : {S:10.3f} m²")
    print(f"Bref        : {totals['Bref']:10.3f} m")
    print(f"Cref        : {totals['Cref']:10.3f} m")
    print(f"Lift        : {L:10.2f} N")
    print(f"Drag        : {D:10.2f} N")
    print("=" * 45)

    # Generate geometry and aerodynamic distribution plots.
    plot_all(
        geometry_data,
        controls_data,
        flight_data,
        results,
    )

    # AVL visualization
    session.show_geometry()

    # return totals


if __name__ == "__main__":
    main()
    # main_short()
