import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

READER_DIR = Path(__file__).resolve().parents[1] / "unimeth" / "ioutils" / "reader"

raw_signal_spec = importlib.util.spec_from_file_location(
    "raw_signal_module",
    READER_DIR / "raw_signal.py",
)
raw_signal_module = importlib.util.module_from_spec(raw_signal_spec)
raw_signal_spec.loader.exec_module(raw_signal_module)

slow5_spec = importlib.util.spec_from_file_location(
    "slow5_module",
    READER_DIR / "slow5.py",
)
slow5_module = importlib.util.module_from_spec(slow5_spec)
slow5_spec.loader.exec_module(slow5_module)

is_slow5_path = raw_signal_module.is_slow5_path
Slow5Reader = slow5_module.Slow5Reader


class FakeSlow5File:
    def __init__(self):
        self.records = [
            {
                "read_id": "read-1",
                "signal": [10, 11, 12],
                "digitisation": 8192.0,
                "offset": 3.0,
                "range": 1467.6,
            }
        ]

    def seq_reads(self):
        return iter(self.records)

    def get_read(self, read_id, aux="all"):
        for record in self.records:
            if record["read_id"] == read_id:
                return record
        raise KeyError(read_id)


class FakeChunkedReadIdsSlow5File(FakeSlow5File):
    def __init__(self):
        self.records = [
            {
                "read_id": "read-1",
                "signal": [10],
                "digitisation": 1.0,
                "offset": 0.0,
                "range": 1.0,
            },
            {
                "read_id": "read-2",
                "signal": [11],
                "digitisation": 1.0,
                "offset": 0.0,
                "range": 1.0,
            },
        ]

    def get_read_ids(self):
        return [["read-1", "read-2"], 4000]


class Slow5ReaderTest(unittest.TestCase):
    def test_detects_slow5_and_blow5_suffixes(self):
        self.assertTrue(is_slow5_path("reads.slow5"))
        self.assertTrue(is_slow5_path("reads.blow5"))
        self.assertFalse(is_slow5_path("reads.pod5"))

    def test_reader_exposes_pod5_like_interface(self):
        fake_pyslow5 = SimpleNamespace(Open=lambda path, mode: FakeSlow5File())

        with patch.object(slow5_module.Path, "is_file", return_value=True), \
             patch.dict(sys.modules, {"pyslow5": fake_pyslow5}):
            reader = Slow5Reader("reads.blow5")
            read = reader.get_read("read-1")

        self.assertEqual(reader.read_ids, ["read-1"])
        np.testing.assert_array_equal(read.signal, np.array([10, 11, 12]))
        self.assertEqual(read.calibration.offset, 3.0)
        self.assertAlmostEqual(read.calibration.scale, 1467.6 / 8192.0)

    def test_reader_flattens_chunked_read_ids(self):
        fake_pyslow5 = SimpleNamespace(Open=lambda path, mode: FakeChunkedReadIdsSlow5File())

        with patch.object(slow5_module.Path, "is_file", return_value=True), \
             patch.dict(sys.modules, {"pyslow5": fake_pyslow5}):
            reader = Slow5Reader("reads.blow5")

        self.assertEqual(reader.read_ids, ["read-1", "read-2"])


if __name__ == "__main__":
    unittest.main()
