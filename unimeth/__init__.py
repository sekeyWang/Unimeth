"""Unimeth - Unified DNA Methylation Detection from Nanopore Reads."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def _version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return data.get("project", {}).get("version")


def _version_from_metadata() -> str:
    try:
        return version("unimeth")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _version_from_pyproject() or _version_from_metadata()
