"""
Annotation module for creating read-level calibration labels.

This module redistributes site-level bisulfite labels to individual reads
based on model predictions, creating read-level training data for calibration.

Example usage:
    python -m scripts.calibration.annotator \
        --pred_result <prediction_result.txt> \
        --bam_file_dir <input_bam> \
        --bam_write_dir <output_folder> \
        --mapq_thres 10 \
        --cpg 1
"""
import argparse
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

from unimeth.ioutils.reader import TSVReader
from unimeth.ioutils.writer import LabelBAMWriter


def create_read_labels(
    site_label: Dict[str, int],
    site_result: Dict[str, List],
    min_reads: int = 5
) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Redistribute site-level labels to reads based on prediction ranking.

    For each site, sorts reads by predicted methylation probability and
    assigns labels to match the bisulfite methylation ratio.

    Args:
        site_label: Dictionary mapping chr_pos to bisulfite score (0-100)
        site_result: Dictionary mapping chr_pos to list of [read_id, pred]
        min_reads: Minimum number of reads required at a site

    Returns:
        Dictionary mapping read_id to list of (chr, pos, label) tuples
    """
    print(f'Number of sites: {len(site_result)}')

    read_labels: Dict[str, List[Tuple[str, int, int]]] = {}
    valid_ref_pos = 0

    for chr_ref_pos in tqdm(site_result, desc='Processing sites'):
        if len(site_result[chr_ref_pos]) <= min_reads:
            continue

        valid_ref_pos += 1

        # Parse chr and pos from chr_ref_pos (format: chr_pos)
        parts = chr_ref_pos.rsplit('_', 1)
        if len(parts) != 2:
            continue
        chr_name, pos_str = parts
        try:
            pos = int(pos_str)
        except ValueError:
            continue

        # Sort reads by prediction score (descending)
        sorted_result = sorted(site_result[chr_ref_pos], key=lambda x: x[1], reverse=True)
        read_ids = [x[0] for x in sorted_result]
        preds = np.array([x[1] for x in sorted_result])

        # Calculate number of methylated reads based on bisulfite ratio
        bis_label = site_label[chr_ref_pos] / 100
        bis_pos = int(round(len(preds) * bis_label))

        # Assign labels
        for i, read_id in enumerate(read_ids):
            if read_id not in read_labels:
                read_labels[read_id] = []

            if i < bis_pos:
                read_label = 100  # Methylated
            elif i >= bis_pos:
                read_label = 0    # Unmethylated
            else:
                continue

            read_labels[read_id].append((chr_name, pos, read_label))

    print(f'Valid reference positions: {valid_ref_pos}')
    print(f'Reads with new labels: {len(read_labels)}')
    return read_labels


class ReadAnnotator:
    """Main class for read-level annotation."""

    def __init__(self, args):
        """
        Initialize annotator.

        Args:
            args: Arguments with pred_result, bam_file_dir, bam_write_dir, etc.
        """
        self.args = args

    def annotate(self) -> Dict[str, List[Tuple[str, int, int]]]:
        """
        Run annotation process.

        Returns:
            Dictionary mapping read_id to list of (chr, pos, label) tuples
        """
        print(f'annotation.py: {self.args}')

        tsv_reader = TSVReader(self.args.pred_result)
        site_label, site_result = tsv_reader.load_site_results()
        read_labels = create_read_labels(site_label, site_result)

        bam_writer = LabelBAMWriter(
            bam_dir=self.args.bam_file_dir,
            out_folder=self.args.bam_write_dir,
            mapq_thres=self.args.mapq_thres
        )
        bam_writer.write_multi_process(read_labels)

        print(f'Annotation complete. Written to {self.args.bam_write_dir}')
        return read_labels


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for annotation."""
    parser = argparse.ArgumentParser(
        description='Create read-level labels from site-level bisulfite data'
    )
    parser.add_argument(
        '--pred_result',
        type=str,
        required=True,
        help='Path to prediction result file (TSV format)'
    )
    parser.add_argument(
        '--bam_file_dir',
        type=str,
        required=True,
        help='Path to input BAM file'
    )
    parser.add_argument(
        '--bam_write_dir',
        type=str,
        required=True,
        help='Output folder for annotated BAM files'
    )
    parser.add_argument(
        '--mapq_thres',
        type=int,
        default=10,
        help='Minimum mapping quality threshold (default: 10)'
    )
    parser.add_argument(
        '--cpg',
        type=int,
        default=0,
        help='Enable CpG detection (1=yes)'
    )
    parser.add_argument(
        '--chg',
        type=int,
        default=0,
        help='Enable CHG detection (1=yes)'
    )
    parser.add_argument(
        '--chh',
        type=int,
        default=0,
        help='Enable CHH detection (1=yes)'
    )
    parser.add_argument(
        '--m6A',
        type=int,
        default=0,
        help='Enable m6A detection (1=yes)'
    )
    parser.add_argument(
        '--pore_type',
        type=str,
        help='Nanopore type (R9.4.1 or R10.4.1)'
    )
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    annotator = ReadAnnotator(args)
    annotator.annotate()


if __name__ == '__main__':
    main()
