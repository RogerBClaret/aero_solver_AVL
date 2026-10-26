
import sys
from pathlib import Path
import numpy as np

# Works in both Jupyter and normal Python scripts
ROOT = (
    Path(__file__).resolve().parents[1]
    if "__file__" in globals()
    else Path.cwd()
)
sys.path.insert(0, str(ROOT))

from avlwrapper import Session

from src.build_model import (
    load_inputs,
    build_geometry,
    build_case,
)

from src.optimization.metric_registry import (
    evaluate_metric,
    available_metrics,
)


def test_metrics():

    # 1. LOAD INPUTS
    geom, mesh, controls, flight = load_inputs()

    assert geom["geometry"]["symmetric"] is False

    # 2. BUILD ONE HALF-WING, ONE CASE
    geometry = build_geometry(geom, mesh, controls)
    case = build_case(flight, controls)

    assert len(geometry.surfaces) == 1
    assert geometry.surfaces[0].y_duplicate is None

    print("Running HALF-WING AVL...")

    results = Session(
        geometry=geometry,
        cases=[case]
    ).get_results()

    assert len(results) == 1
    assert "operating_point" in results

    # 3. GEOMETRY METRICS
    print("\n=== GEOMETRY METRICS ===")

    geometry_names = [
        "half_span",
        "full_span",
        "half_area",
        "full_area",
        "aspect_ratio",
        "taper_ratio",
        "sweep_LE",
        "dihedral",
        "washout",
    ]

    for name in geometry_names:
        value = evaluate_metric(name, geom)

        print(f"{name:22s}: {value:.6f}")
        assert np.isfinite(value)

    # Full-wing dimensions are calculated analytically.
    # AVL itself only analyzes the half-wing.
    assert np.isclose(
        evaluate_metric("full_span", geom),
        2 * evaluate_metric("half_span", geom)
    )

    assert np.isclose(
        evaluate_metric("full_area", geom),
        2 * evaluate_metric("half_area", geom)
    )

    # 4. AERODYNAMIC METRICS
    print("\n=== AERODYNAMIC METRICS ===")

    aero_names = [
        "CL",
        "CDi",
        "Cm",
        "Cl",
        "Cn",
        "CL_CDi",
    ]

    for name in aero_names:
        value = evaluate_metric(name, geom, results)

        print(f"{name:22s}: {value:.8f}")
        assert np.isfinite(value)

    # 5. CONTROL EFFECTIVENESS
    print("\n=== CONTROL EFFECTIVENESS ===")

    derivatives = [
        "dCL_ddeg",
        "dCD_ddeg",
        "dCY_ddeg",
        "dCl_ddeg",
        "dCm_ddeg",
        "dCn_ddeg",
    ]

    for control in controls["control_surfaces"]:
        control_name = control["name"]

        print(f"\n{control_name}")

        for derivative in derivatives:
            metric_name = f"{derivative}.{control_name}"

            value = evaluate_metric(
                metric_name, geom, results
            )

            print(f"  {derivative:15s}: {value:.8f}")
            assert np.isfinite(value)

    # Aileron must generate a rolling moment derivative.
    aileron_eff = evaluate_metric(
        "dCl_ddeg.aileron",
        geom,
        results
    )

    assert abs(aileron_eff) > 1e-10

    # 6. AVAILABLE OPTIMIZATION METRICS
    print("\n=== AVAILABLE METRICS ===")

    names = available_metrics(controls)

    for name in names:
        print(name)

    assert "full_area" in names
    assert "aspect_ratio" in names
    assert "CL" in names
    assert "CDi" in names
    assert "dCl_ddeg.aileron" in names

    print("\nSUCCESS: All metrics validated.")


if __name__ == "__main__":
    test_metrics()
