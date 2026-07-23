"""SLOW5/BLOW5 reader compatibility layer.

The rest of UniMeth consumes a small POD5-like interface:
``read_ids`` plus ``get_read(read_id)`` returning an object with
``signal`` and ``calibration.offset/scale``.  This module adapts pyslow5
records to that interface so the existing data flow can stay unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _import_pyslow5():
    try:
        import pyslow5
    except ImportError as exc:
        raise ImportError(
            "Reading SLOW5/BLOW5 input requires pyslow5. "
            "Install pyslow5 separately with `pip install pyslow5` "
            "or `conda install -c bioconda pyslow5`. "
            "When installing UniMeth with pip, you can also include pyslow5 "
            "with `pip install \"unimeth[slow5]\"`."
        ) from exc
    except ValueError as exc:
        if "numpy.dtype size changed" in str(exc):
            raise ImportError(
                "pyslow5 is installed but is binary-incompatible with the current "
                "NumPy build. Reinstall pyslow5 in this environment, preferably "
                "with `pip install --no-cache-dir --force-reinstall pyslow5`, or "
                "rebuild it against the installed NumPy."
            ) from exc
        raise
    return pyslow5


def _open_slow5(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"SLOW5/BLOW5 file not found: {path}")

    pyslow5 = _import_pyslow5()
    return pyslow5.Open(str(path), "r")


def _record_read_id(record):
    for key in ("read_id", "readID", "read_name"):
        if key in record:
            rid = record[key]
            return rid.decode() if isinstance(rid, bytes) else str(rid)
    raise KeyError("SLOW5 record does not contain a read_id field")


def _normalize_read_id(read_id):
    return read_id.decode() if isinstance(read_id, bytes) else str(read_id)


def _flatten_read_ids(read_ids):
    for read_id in read_ids:
        if isinstance(read_id, (list, tuple, np.ndarray)):
            yield from _flatten_read_ids(read_id)
        elif isinstance(read_id, (str, bytes)):
            yield _normalize_read_id(read_id)


def _record_signal(record):
    for key in ("signal", "raw_signal", "raw_signal_pa"):
        if key in record:
            return np.asarray(record[key])
    raise KeyError("SLOW5 record does not contain a signal field")


def _calibration_from_record(record):
    digitisation = float(record.get("digitisation", record.get("digitization", 1.0)))
    offset = float(record.get("offset", 0.0))
    signal_range = float(record.get("range", digitisation))

    scale = signal_range / digitisation if digitisation else 1.0
    return SimpleNamespace(offset=offset, scale=scale)


class Slow5Reader:
    """POD5 DatasetReader-compatible wrapper for one SLOW5/BLOW5 file."""

    def __init__(self, path):
        self.path = str(path)
        self._slow5 = _open_slow5(path)
        self.read_ids = []
        self._read_id_set = set()
        self._build_index()

    def _build_index(self):
        if hasattr(self._slow5, "get_read_ids"):
            self.read_ids = list(_flatten_read_ids(self._slow5.get_read_ids()))
            self._read_id_set = set(self.read_ids)
            return

        for record in self._slow5.seq_reads():
            read_id = _record_read_id(record)
            self.read_ids.append(read_id)
        self._read_id_set = set(self.read_ids)

    def get_read(self, read_id):
        read_id = str(read_id)
        record = self._fetch_record(read_id)
        if record is None:
            raise KeyError(f"Read {read_id} not found in {self.path}")

        return SimpleNamespace(
            read_id=read_id,
            signal=_record_signal(record),
            calibration=_calibration_from_record(record),
        )

    def _fetch_record(self, read_id):
        if self._read_id_set and read_id not in self._read_id_set:
            return None

        for kwargs in ({"aux": "all", "pA": False}, {"aux": "all"}, {}):
            try:
                return self._slow5.get_read(str(read_id), **kwargs)
            except TypeError:
                continue
            except Exception:
                break

        for record in self._slow5.seq_reads():
            if _record_read_id(record) == str(read_id):
                return record
        return None
