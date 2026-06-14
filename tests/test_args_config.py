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
