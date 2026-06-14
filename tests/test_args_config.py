import unittest

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


if __name__ == "__main__":
    unittest.main()
