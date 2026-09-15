import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch


class FakeAlignmentFile:
    output = None

    def __init__(self, path, mode, header=None, **kwargs):
        self.path = path
        self.mode = mode
        self.header = header or {"HD": {"VN": "1.6"}}
        self.written = []
        if "w" in mode:
            FakeAlignmentFile.output = self

    def write(self, read):
        self.written.append(read)

    def close(self):
        pass


class FakeRead:
    query_name = "read1"

    def __init__(self):
        self.tags = {"mv": [5, 1, 0], "ts": 0, "sm": 0.0, "sd": 1.0}

    def get_forward_sequence(self):
        return "ACG"

    def has_tag(self, tag):
        return tag in self.tags

    def set_tag(self, tag=None, value=None, **kwargs):
        if value is None:
            self.tags.pop(tag, None)
        else:
            self.tags[tag] = value


def load_bam_aggregation_module():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_path = repo_root / "unimeth" / "ioutils" / "writer" / "bam_aggregation.py"
    bam_tags_path = repo_root / "unimeth" / "utils" / "bam_tags.py"
    bam_tags_spec = importlib.util.spec_from_file_location("unimeth.utils.bam_tags", bam_tags_path)
    bam_tags_module = importlib.util.module_from_spec(bam_tags_spec)
    bam_tags_spec.loader.exec_module(bam_tags_module)

    spec = importlib.util.spec_from_file_location("bam_aggregation_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    fake_pysam = SimpleNamespace(AlignmentFile=FakeAlignmentFile)
    fake_torch = SimpleNamespace(Tensor=object)
    fake_ioutils = ModuleType("unimeth.ioutils")
    fake_ioutils.__path__ = []
    fake_reader = ModuleType("unimeth.ioutils.reader")
    fake_reader.__path__ = []
    fake_bam = ModuleType("unimeth.ioutils.reader.bam")
    fake_bam.BamReader = object
    fake_utils = ModuleType("unimeth.utils")
    fake_utils.__path__ = []
    with patch.dict(sys.modules, {
        "pysam": fake_pysam,
        "torch": fake_torch,
        "unimeth.ioutils": fake_ioutils,
        "unimeth.ioutils.reader": fake_reader,
        "unimeth.ioutils.reader.bam": fake_bam,
        "unimeth.utils": fake_utils,
        "unimeth.utils.bam_tags": bam_tags_module,
    }):
        spec.loader.exec_module(module)
    return module


def make_record_reader(read):
    return SimpleNamespace(
        get_record=lambda output_record_key: read,
        close=lambda: None,
    )


class BamAggregationMvTagTest(unittest.TestCase):
    def test_writer_removes_mv_tag_by_default(self):
        module = load_bam_aggregation_module()
        read = FakeRead()
        writer = module.AggregationBAMWriter(
            output_path="out.bam",
            template_bam_path="in.bam",
            record_reader=make_record_reader(read),
        )
        buf = SimpleNamespace(
            read_id="read1",
            output_record_key=123,
            received={
                0: [
                    module.PatchPrediction(
                        prob=0.9,
                        methy_type="[CpG]",
                        read_pos=1,
                        ref_pos=10,
                        chr="chr1",
                        strand="+",
                        patch_idx=0,
                    )
                ]
            },
        )

        writer._write_read_to_bam(buf)

        self.assertNotIn("mv", read.tags)
        self.assertIn("MM", read.tags)
        self.assertEqual(FakeAlignmentFile.output.written, [read])

    def test_writer_can_keep_mv_tag_when_requested(self):
        module = load_bam_aggregation_module()
        read = FakeRead()
        writer = module.AggregationBAMWriter(
            output_path="out.bam",
            template_bam_path="in.bam",
            record_reader=make_record_reader(read),
            keep_mv=True,
        )
        buf = SimpleNamespace(
            read_id="read1",
            output_record_key=123,
            received={
                0: [
                    module.PatchPrediction(
                        prob=0.9,
                        methy_type="[CpG]",
                        read_pos=1,
                        ref_pos=10,
                        chr="chr1",
                        strand="+",
                        patch_idx=0,
                    )
                ]
            },
        )

        writer._write_read_to_bam(buf)

        self.assertIn("mv", read.tags)
        self.assertIn("MM", read.tags)
        self.assertEqual(FakeAlignmentFile.output.written, [read])

    def test_exit_with_exception_discards_buffer_without_writing_partial_reads(self):
        module = load_bam_aggregation_module()
        writer = module.AggregationBAMWriter(
            output_path="out.bam",
            template_bam_path="in.bam",
            record_reader=make_record_reader(FakeRead()),
        )
        buf = module.ReadBuffer(
            read_id="read1",
            expected=2,
            output_record_key=123,
        )
        buf.received[0].append(
            module.PatchPrediction(
                prob=0.9,
                methy_type="[CpG]",
                read_pos=1,
                ref_pos=10,
                chr="chr1",
                strand="+",
                patch_idx=0,
            )
        )
        writer.buffer["read1"] = buf
        write_calls = []
        writer._write_read_to_bam = lambda remaining_buf: write_calls.append(remaining_buf.read_id)

        suppressed = writer.__exit__(RuntimeError, RuntimeError("boom"), None)

        self.assertFalse(suppressed)
        self.assertEqual(write_calls, [])
        self.assertEqual(writer.buffer, {})
        self.assertEqual(writer.stats["reads_discarded_on_error"], 1)

    def test_exit_without_exception_flushes_remaining_buffer(self):
        module = load_bam_aggregation_module()
        writer = module.AggregationBAMWriter(
            output_path="out.bam",
            template_bam_path="in.bam",
            record_reader=make_record_reader(FakeRead()),
        )
        buf = module.ReadBuffer(
            read_id="read1",
            expected=1,
            output_record_key=123,
        )
        buf.received[0].append(
            module.PatchPrediction(
                prob=0.9,
                methy_type="[CpG]",
                read_pos=1,
                ref_pos=10,
                chr="chr1",
                strand="+",
                patch_idx=0,
            )
        )
        writer.buffer["read1"] = buf
        write_calls = []
        writer._write_read_to_bam = lambda remaining_buf: write_calls.append(remaining_buf.read_id)

        suppressed = writer.__exit__(None, None, None)

        self.assertFalse(suppressed)
        self.assertEqual(write_calls, ["read1"])
        self.assertEqual(writer.buffer, {})


if __name__ == "__main__":
    unittest.main()
