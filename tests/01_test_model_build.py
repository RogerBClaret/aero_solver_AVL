from avlwrapper import Session
from src.build_model import (
    load_inputs, build_geometry, build_case
)

geom, mesh, controls, flight = load_inputs()

geometry = build_geometry(geom, mesh, controls)
case = build_case(flight, controls)

session = Session(
    geometry=geometry,
    cases=[case]
)

results = session.get_results()

print(results["operating_point"]["Totals"])

