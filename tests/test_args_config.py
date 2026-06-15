import unittest
from contextlib import redirect_stdout
from io import StringIO

from unimeth import __version__
from unimeth.config.args_config import create_argument_parser


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

    def test_inference_help_omits_training_pod5_options(self):
        help_text = create_argument_parser("inference").format_help()

        self.assertNotIn("--train_pod5_dir", help_text)
        self.assertNotIn("--train_pod5", help_text)
        self.assertNotIn("--val_pod5_dir", help_text)
        self.assertNotIn("--val_pod5", help_text)

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
        self.assertEqual(__version__, "0.2.0")

    def test_inference_parser_prints_version(self):
        parser = create_argument_parser("inference")
        stdout = StringIO()

        with self.assertRaises(SystemExit) as ctx, redirect_stdout(stdout):
            parser.parse_args(["--version"])

        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), "unimeth-infer 0.2.0")


if __name__ == "__main__":
    unittest.main()
