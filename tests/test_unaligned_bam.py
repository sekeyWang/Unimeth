import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_extract_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "unimeth" / "data" / "extract.py"

    coords_spec = importlib.util.spec_from_file_location(
        "unimeth.data.coords",
        root / "unimeth" / "data" / "coords.py",
    )
    coords_module = importlib.util.module_from_spec(coords_spec)
    coords_spec.loader.exec_module(coords_module)

    sites_spec = importlib.util.spec_from_file_location(
        "unimeth.data.sites",
        root / "unimeth" / "data" / "sites.py",
    )
    sites_module = importlib.util.module_from_spec(sites_spec)
    sites_spec.loader.exec_module(sites_module)

    fake_unimeth = SimpleNamespace()
    fake_data = SimpleNamespace(coords=coords_module, sites=sites_module)

    with patch.dict(
        sys.modules,
        {
            "unimeth": fake_unimeth,
            "unimeth.data": fake_data,
            "unimeth.data.coords": coords_module,
            "unimeth.data.sites": sites_module,
        },
    ):
        spec = importlib.util.spec_from_file_location("unimeth.data.extract", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class FakeUnalignedRead:
    query_name = "read-1"
    mapping_quality = 0
    reference_name = None
    reference_start = -1
    is_unmapped = True
    is_reverse = False
    modified_bases_forward = {}

    def get_forward_sequence(self):
        return "ACGTT"

    def get_tag(self, tag):
        tags = {
            "ts": 0,
            "mv": [1, 1, 1, 1, 1, 1],
            "sm": 0.0,
            "sd": 1.0,
        }
        return tags[tag]

    def has_tag(self, tag):
        return False

    def get_aligned_pairs(self):
        raise AssertionError("unaligned reads should not request aligned pairs")


class UnalignedBAMFeatureTest(unittest.TestCase):
    def test_unaligned_read_can_produce_read_level_feature(self):
        extract_module = load_extract_module()
        extractor = extract_module.SignalFeatureExtractor(
            SimpleNamespace(mapq_thres=10, pore_type="R10.4.1", chr="|", cpg=1, chg=0, chh=0, m6A=0)
        )
        pod5_read = SimpleNamespace(
            signal=[10, 11, 12, 13, 14, 15],
            calibration=SimpleNamespace(offset=0.0, scale=1.0),
        )

        fake_utils = SimpleNamespace()
        fake_bam_tags = SimpleNamespace(get_modifications=lambda bam_read, mod_key: {})
        with patch.dict(
            sys.modules,
            {
                "unimeth.utils": fake_utils,
                "unimeth.utils.bam_tags": fake_bam_tags,
            },
        ):
            feature = extractor.get_feature(FakeUnalignedRead(), pod5_read)

        self.assertIsNotNone(feature)
        self.assertEqual(feature["chr"], "*")
        self.assertEqual(feature["reference_start"], -1)
        self.assertEqual(feature["pred_pos"], [1])
        self.assertEqual(feature["ref_pos"], [-1])
        self.assertEqual(feature["strand"], "+")


if __name__ == "__main__":
    unittest.main()
