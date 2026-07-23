import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tomllib

from unimeth import __version__
from unimeth.config.args_config import create_argument_parser
from unimeth.inference.__main__ import normalize_signal_input

POD5_SUFFIXES = (".pod5",)
SLOW5_SUFFIXES = (".slow5", ".blow5")


class InferenceArgumentAliasesTest(unittest.TestCase):
    def test_short_path_aliases_share_legacy_destinations(self):
        parser = create_argument_parser("inference")

        args = parser.parse_args([
            "--pod5", "reads.pod5",
            "--bam", "reads.bam",
            "--model", "model.pt",
            "--out", "predictions.tsv",
            "--tsv_out", "predictions.tsv",
            "--bam_out", "predictions.bam",
        ])

        self.assertEqual(args.pod5_dir, "reads.pod5")
        self.assertEqual(args.bam_dir, "reads.bam")
        self.assertEqual(args.model_dir, "model.pt")
        self.assertEqual(args.out_dir, "predictions.tsv")
        self.assertEqual(args.tsv_out_dir, "predictions.tsv")
        self.assertEqual(args.bam_out_dir, "predictions.bam")

    def test_slow5_aliases_use_separate_destination(self):
        parser = create_argument_parser("inference")

        args = parser.parse_args([
            "--slow5", "reads.blow5",
            "--bam", "reads.bam",
        ])

        self.assertIsNone(args.pod5_dir)
        self.assertEqual(args.slow5_dir, "reads.blow5")

    def test_training_help_omits_slow5_options(self):
        for mode in ["pretrain", "finetune", "calibration"]:
            with self.subTest(mode=mode):
                help_text = create_argument_parser(mode).format_help()

                self.assertNotIn("--slow5", help_text)
                self.assertNotIn("--slow5_dir", help_text)

    def test_slow5_normalizes_to_signal_input(self):
        parser = create_argument_parser("inference")
        args = parser.parse_args(["--slow5", "reads.blow5"])

        normalize_signal_input(args, parser)

        self.assertEqual(args.signal_dir, "reads.blow5")
        self.assertEqual(args.signal_format, "SLOW5/BLOW5")
        self.assertEqual(args.signal_suffixes, SLOW5_SUFFIXES)
        self.assertEqual(args.signal_label, "SLOW5/BLOW5")
        self.assertEqual(args.pod5_dir, "reads.blow5")

    def test_pod5_normalizes_to_signal_input(self):
        parser = create_argument_parser("inference")
        args = parser.parse_args(["--pod5", "reads.pod5"])

        normalize_signal_input(args, parser)

        self.assertEqual(args.signal_dir, "reads.pod5")
        self.assertEqual(args.signal_format, "POD5")
        self.assertEqual(args.signal_suffixes, POD5_SUFFIXES)
        self.assertEqual(args.signal_label, "POD5")
        self.assertEqual(args.pod5_dir, "reads.pod5")

    def test_pod5_and_slow5_are_mutually_exclusive(self):
        parser = create_argument_parser("inference")
        stderr = StringIO()
        args = parser.parse_args(["--pod5", "reads.pod5", "--slow5", "reads.blow5"])

        with self.assertRaises(SystemExit) as ctx, redirect_stderr(stderr):
            normalize_signal_input(args, parser)

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("Use only one raw signal input option", stderr.getvalue())

    def test_legacy_path_options_still_work(self):
        parser = create_argument_parser("inference")

        args = parser.parse_args([
            "--pod5_dir", "reads.pod5",
            "--bam_dir", "reads.bam",
            "--model_dir", "model.pt",
            "--out_dir", "predictions.tsv",
        ])

        self.assertEqual(args.pod5_dir, "reads.pod5")
        self.assertEqual(args.bam_dir, "reads.bam")
        self.assertEqual(args.model_dir, "model.pt")
        self.assertEqual(args.out_dir, "predictions.tsv")

    def test_inference_gzip_flag_defaults_to_false_and_can_be_enabled(self):
        parser = create_argument_parser("inference")

        self.assertFalse(parser.parse_args([]).gzip)
        self.assertTrue(parser.parse_args(["--gzip"]).gzip)

    def test_inference_mapq_sets_mapq_threshold(self):
        parser = create_argument_parser("inference")

        self.assertEqual(parser.parse_args([]).mapq_thres, 0)
        self.assertEqual(parser.parse_args(["--mapq", "20"]).mapq_thres, 20)

    def test_inference_keep_mv_flag_defaults_to_false_and_can_be_enabled(self):
        parser = create_argument_parser("inference")

        self.assertFalse(parser.parse_args([]).keep_mv)
        self.assertTrue(parser.parse_args(["--keep_mv"]).keep_mv)

    def test_inference_help_omits_training_pod5_options(self):
        help_text = create_argument_parser("inference").format_help()

        self.assertNotIn("--train_pod5_dir", help_text)
        self.assertNotIn("--train_pod5", help_text)
        self.assertNotIn("--val_pod5_dir", help_text)
        self.assertNotIn("--val_pod5", help_text)
        self.assertNotIn("--version", help_text)

    def test_training_pod5_aliases_stay_available_for_training_modes(self):
        for mode in ["pretrain", "finetune", "calibration"]:
            with self.subTest(mode=mode):
                parser = create_argument_parser(mode)
                args = parser.parse_args([
                    "--train_pod5", "train.pod5",
                    "--val_pod5", "val.pod5",
                ])

                self.assertEqual(args.train_pod5_dir, "train.pod5")
                self.assertEqual(args.val_pod5_dir, "val.pod5")

    def test_version_comes_from_project_metadata(self):
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as handle:
            project = tomllib.load(handle)["project"]

        self.assertEqual(__version__, project["version"])

    def test_slow5_dependency_is_optional(self):
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as handle:
            pyproject = tomllib.load(handle)

        dependencies = pyproject["project"]["dependencies"]
        optional_dependencies = pyproject["project"]["optional-dependencies"]

        self.assertNotIn("pyslow5>=1.4.0", dependencies)
        self.assertIn("pyslow5>=1.4.0", optional_dependencies["slow5"])

    def test_inference_parser_rejects_version(self):
        parser = create_argument_parser("inference")
        parser.prog = "unimeth-infer"
        stderr = StringIO()

        with self.assertRaises(SystemExit) as ctx, redirect_stderr(stderr):
            parser.parse_args(["--version"])

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("unrecognized arguments: --version", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
