"""
Convert BED format bisulfite labels to BAM format annotations.

Example usage:
    python -m scripts.calibration.bed_to_bam \
        --bed_file_dir <wgbs_bed_file> \
        --bam_file_dir <aligned_bam> \
        --mapq_thres 10 \
        --coverage_thres 5 \
        --chr <chromosome_filter> \
        --out_folder <output_dir>
"""
import argparse
from typing import Dict, Tuple

from unimeth.ioutils.reader import BEDReader
from unimeth.ioutils.writer import LabelBAMWriter


def convert_bed_to_site_labels(bed_labels: Dict[str, float]) -> Dict[Tuple[str, int], float]:
    """
    Convert BED labels from {chr_pos: score} to {(chr, pos): score} format.
    
    Args:
        bed_labels: Dictionary mapping chr_pos to methylation score
        
    Returns:
        Dictionary mapping (chr, pos) to methylation score
    """
    site_labels: Dict[Tuple[str, int], float] = {}
    for chr_pos, score in bed_labels.items():
        # Parse chr_pos (format: chr_pos)
        parts = chr_pos.rsplit('_', 1)
        if len(parts) != 2:
            continue
        chr_name, pos_str = parts
        try:
            pos = int(pos_str)
            site_labels[(chr_name, pos)] = score
        except ValueError:
            continue
    return site_labels


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for bed_to_bam."""
    parser = argparse.ArgumentParser(
        description='Convert BED format bisulfite labels to BAM annotations'
    )
    parser.add_argument(
        '--bed_file_dir',
        type=str,
        required=True,
        help='Path to BED file with bisulfite labels'
    )
    parser.add_argument(
        '--bam_file_dir',
        type=str,
        required=True,
        help='Path to input BAM file'
    )
    parser.add_argument(
        '--coverage_thres',
        type=int,
        default=5,
        help='Minimum coverage threshold (default: 5)'
    )
    parser.add_argument(
        '--mapq_thres',
        type=int,
        default=10,
        help='Minimum mapping quality threshold (default: 10)'
    )
    parser.add_argument(
        '--chr',
        type=str,
        default='|',
        help='Chromosome filter (e.g., "|" for all, "|Chr1,Chr2" for include only)'
    )
    parser.add_argument(
        '--out_folder',
        type=str,
        required=True,
        help='Output folder for annotated BAM files'
    )
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    print(f'bed_to_bam: {args}')

    bam_writer = LabelBAMWriter(
        bam_dir=args.bam_file_dir,
        out_folder=args.out_folder,
        mapq_thres=args.mapq_thres
    )

    bed_labels = BEDReader(args.bed_file_dir).load_labels(
        coverage_thres=args.coverage_thres,
        chr_str=args.chr
    )

    if bed_labels:
        # Convert to site-level format
        site_labels = convert_bed_to_site_labels(bed_labels)
        bam_writer.write_multi_process(site_labels)
        print(f'Written {len(site_labels)} labeled sites to {args.out_folder}')
    else:
        print('No bisulfite labels found.')


if __name__ == '__main__':
    main()
