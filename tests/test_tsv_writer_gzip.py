import gzip
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def load_tsv_module():
    module_path = Path(__file__).resolve().parents[1] / "unimeth" / "ioutils" / "writer" / "tsv.py"

    fake_torch = ModuleType("torch")
    fake_torch.Tensor = object
    fake_config = ModuleType("unimeth.config")
    fake_config.VOCAB = ["[PAD]"]

    fake_utils = ModuleType("unimeth.utils")
    fake_common = ModuleType("unimeth.utils.common")

    def make_output_path(output_path, create_parent=True):
        path = Path(output_path)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def merge_rank_files(*args, **kwargs):
        raise AssertionError("gzip merge should not use uncompressed merge_rank_files")

    fake_common.make_output_path = make_output_path
    fake_common.merge_rank_files = merge_rank_files

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "unimeth.config": fake_config,
            "unimeth.utils": fake_utils,
            "unimeth.utils.common": fake_common,
        },
    ):
        spec = importlib.util.spec_from_file_location("tsv_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TSVWriterGzipTest(unittest.TestCase):
    def test_gzip_merge_writes_final_gz_and_removes_rank_files(self):
        TSVWriter = load_tsv_module().TSVWriter

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "calls.tsv"
            writer = TSVWriter(str(output_path), num_processes=2, process_index=0, gzip_output=True)

            self.assertEqual(writer.output_path, Path(tmp) / "calls.tsv.gz")

            rank0 = writer.rank_output
            rank1 = writer.rank_output_for(1)
            rank0.write_text("a\t1\n", encoding="utf-8")
            rank1.write_text("b\t2\n", encoding="utf-8")

            writer.merge_outputs(is_main_process=True)

            self.assertFalse(rank0.exists())
            self.assertFalse(rank1.exists())
            with gzip.open(writer.output_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "a\t1\nb\t2\n")

    def test_existing_gz_suffix_is_not_duplicated(self):
        TSVWriter = load_tsv_module().TSVWriter

        with tempfile.TemporaryDirectory() as tmp:
            writer = TSVWriter(
                str(Path(tmp) / "calls.tsv.gz"),
                num_processes=1,
                process_index=0,
                gzip_output=True,
            )

            self.assertEqual(writer.output_path, Path(tmp) / "calls.tsv.gz")
            self.assertEqual(writer.rank_output.name, "calls_rank0.tsv")

    def test_gz_suffix_auto_enables_gzip_output(self):
        TSVWriter = load_tsv_module().TSVWriter

        with tempfile.TemporaryDirectory() as tmp:
            writer = TSVWriter(
                str(Path(tmp) / "calls.tsv.gz"),
                num_processes=1,
                process_index=0,
                gzip_output=False,
            )

            self.assertTrue(writer.gzip_output)
            writer.rank_output.write_text("a\t1\n", encoding="utf-8")
            writer.merge_outputs(is_main_process=True)

            with gzip.open(writer.output_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "a\t1\n")


if __name__ == "__main__":
    unittest.main()
