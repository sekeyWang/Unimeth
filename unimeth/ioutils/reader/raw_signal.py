"""Utilities for opening raw signal files across supported formats."""
from pathlib import Path


POD5_SUFFIXES = {".pod5"}
SLOW5_SUFFIXES = {".slow5", ".blow5"}
SIGNAL_SUFFIXES = POD5_SUFFIXES | SLOW5_SUFFIXES


def normalize_suffixes(suffixes=None):
    """Return a normalized suffix set for raw signal file filtering."""
    if suffixes is None:
        return SIGNAL_SUFFIXES
    return {
        suffix.lower() if str(suffix).startswith(".") else f".{str(suffix).lower()}"
        for suffix in suffixes
    }


def is_pod5_path(path) -> bool:
    """Return True when path points to a POD5 file."""
    return Path(path).suffix.lower() in POD5_SUFFIXES


def is_slow5_path(path) -> bool:
    """Return True when path points to a SLOW5/BLOW5 file."""
    return Path(path).suffix.lower() in SLOW5_SUFFIXES


def is_signal_path(path) -> bool:
    """Return True when path points to a supported raw signal file."""
    return Path(path).suffix.lower() in SIGNAL_SUFFIXES


def collect_signal_paths(signal_path, suffixes=None, label=None):
    """Collect supported raw signal files from a file or directory."""
    suffixes = normalize_suffixes(suffixes)
    label = label or "POD5/SLOW5/BLOW5"

    path = Path(signal_path)
    if path.is_dir():
        paths = sorted(
            str(child)
            for child in path.rglob("*")
            if child.is_file() and Path(child).suffix.lower() in suffixes
        )
        if not paths:
            raise FileNotFoundError(f"No {label} files found under: {signal_path}")
        return paths

    if path.suffix.lower() not in suffixes:
        raise ValueError(f"Unsupported {label} raw signal file extension: {signal_path}")
    return [str(path)]


def open_pod5_file(path, **pod5_kwargs):
    """Open a POD5 file using the native pod5 reader."""
    import pod5 as p5

    return p5.DatasetReader(path, **pod5_kwargs)


def open_signal_file(path, **pod5_kwargs):
    """Open a POD5 or SLOW5/BLOW5 path with a common reader interface."""
    if is_slow5_path(path):
        from unimeth.ioutils.reader.slow5 import Slow5Reader

        return Slow5Reader(path)

    if is_pod5_path(path):
        return open_pod5_file(path, **pod5_kwargs)

    raise ValueError(f"Unsupported raw signal file extension: {path}")
