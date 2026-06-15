import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_bam_finalize_module():
    module_path = Path(__file__).resolve().parents[1] / "unimeth" / "ioutils" / "writer" / "bam_finalize.py"
    spec = importlib.util.spec_from_file_location("bam_finalize_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BamFinalizeTest(unittest.TestCase):
    def test_inference_engine_finalization_does_not_call_external_samtools(self):
        engine_path = Path(__file__).resolve().parents[1] / "unimeth" / "inference" / "engine.py"
        source = engine_path.read_text(encoding="utf-8")

        self.assertIn("finalize_part_bams", source)
        self.assertNotIn("samtools", source)
        self.assertNotIn("subprocess", source)

    def test_single_part_bam_is_renamed_and_indexed_with_pysam(self):
        calls = []

        def fake_index(*args):
            calls.append(("index", args))

        fake_pysam = SimpleNamespace(index=fake_index)

        with patch.dict(sys.modules, {"pysam": fake_pysam}):
            finalize_part_bams = load_bam_finalize_module().finalize_part_bams

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                part = tmp_path / "calls.part_0.bam"
                final = tmp_path / "calls.bam"
                part.write_bytes(b"bam")

                finalize_part_bams(str(final), [str(part)])

                self.assertTrue(final.exists())
                self.assertFalse(part.exists())
                self.assertEqual(calls, [("index", (str(final),))])

    def test_multiple_part_bams_are_merged_sorted_indexed_and_removed_with_pysam(self):
        calls = []

        def fake_merge(*args):
            calls.append(("merge", args))
            Path(args[3]).write_bytes(b"merged")

        def fake_sort(*args):
            calls.append(("sort", args))
            Path(args[3]).write_bytes(b"sorted")

        def fake_index(*args):
            calls.append(("index", args))

        fake_pysam = SimpleNamespace(
            merge=fake_merge,
            sort=fake_sort,
            index=fake_index,
        )

        with patch.dict(sys.modules, {"pysam": fake_pysam}):
            finalize_part_bams = load_bam_finalize_module().finalize_part_bams

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                parts = [
                    tmp_path / "calls.part_0.bam",
                    tmp_path / "calls.part_1.bam",
                ]
                for part in parts:
                    part.write_bytes(b"bam")
                final = tmp_path / "calls.bam"
                merged_unsorted = tmp_path / "calls.merged_unsorted.bam"

                finalize_part_bams(str(final), [str(part) for part in parts], threads=8)

                self.assertEqual(
                    calls,
                    [
                        (
                            "merge",
                            (
                                "-@",
                                "8",
                                "-f",
                                str(merged_unsorted),
                                str(parts[0]),
                                str(parts[1]),
                            ),
                        ),
                        ("sort", ("-@", "8", "-o", str(final), str(merged_unsorted))),
                        ("index", (str(final),)),
                    ],
                )
                self.assertTrue(final.exists())
                self.assertFalse(merged_unsorted.exists())
                self.assertFalse(parts[0].exists())
                self.assertFalse(parts[1].exists())


if __name__ == "__main__":
    unittest.main()
