import importlib.util
import tempfile
import unittest
from pathlib import Path


SIGNAL_MODULE_PATH = Path(__file__).resolve().parents[1] / "unimeth" / "ioutils" / "reader" / "raw_signal.py"
spec = importlib.util.spec_from_file_location("raw_signal_module", SIGNAL_MODULE_PATH)
raw_signal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_signal_module)

collect_signal_paths = raw_signal_module.collect_signal_paths
POD5_SUFFIXES = raw_signal_module.POD5_SUFFIXES
SLOW5_SUFFIXES = raw_signal_module.SLOW5_SUFFIXES


class SignalPathCollectionTest(unittest.TestCase):
    def test_single_signal_file_is_supported(self):
        self.assertEqual(collect_signal_paths("reads.blow5"), ["reads.blow5"])

    def test_rejects_unsupported_single_file(self):
        with self.assertRaises(ValueError):
            collect_signal_paths("reads.fast5")

    def test_recursively_collects_supported_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            expected = [
                root / "a.pod5",
                root / "b.blow5",
                nested / "c.slow5",
            ]
            ignored = [
                root / "notes.txt",
                nested / "d.fast5",
            ]
            for path in expected + ignored:
                path.write_text("", encoding="utf-8")

            self.assertEqual(
                collect_signal_paths(root),
                sorted(str(path) for path in expected),
            )

    def test_recursively_filters_to_pod5_suffixes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            pod5 = root / "reads.pod5"
            slow5 = nested / "reads.slow5"
            blow5 = nested / "reads.blow5"
            for path in (pod5, slow5, blow5):
                path.write_text("", encoding="utf-8")

            self.assertEqual(
                collect_signal_paths(root, suffixes=POD5_SUFFIXES, label="POD5"),
                [str(pod5)],
            )

    def test_recursively_filters_to_slow5_suffixes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            pod5 = root / "reads.pod5"
            slow5 = nested / "reads.slow5"
            blow5 = nested / "reads.blow5"
            for path in (pod5, slow5, blow5):
                path.write_text("", encoding="utf-8")

            self.assertEqual(
                collect_signal_paths(root, suffixes=SLOW5_SUFFIXES, label="SLOW5/BLOW5"),
                sorted([str(slow5), str(blow5)]),
            )

    def test_rejects_single_file_outside_expected_suffixes(self):
        with self.assertRaises(ValueError):
            collect_signal_paths("reads.pod5", suffixes=SLOW5_SUFFIXES, label="SLOW5/BLOW5")

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                collect_signal_paths(tmpdir)


if __name__ == "__main__":
    unittest.main()
