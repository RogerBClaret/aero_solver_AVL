
"""Read the aircraft configuration files from ./input."""

from pathlib import Path
import yaml


# Project root: one directory above ./src
ROOT = Path(__file__).resolve().parents[1]


def load_inputs(folder=None):
    """
    Load the four input files.

    Parameters
    ----------
    folder : str or Path, optional
        Input directory. Defaults to ./input.

    Returns
    -------
    tuple
        Geometry, mesh, controls and flight dictionaries.
    """

    folder = Path(folder) if folder else ROOT / "input"

    def read(filename):
        path = folder / filename

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML configuration: {path}")

        return data

    return (
        read("geometry.yaml"),
        read("mesh.yaml"),
        read("control_surfaces.yaml"),
        read("flight_condition.yaml"),
    )
