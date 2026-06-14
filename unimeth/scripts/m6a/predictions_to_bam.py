"""
Convert prediction results (TSV) to BAM format with MM/ML tags.

Supports CpG, CHG, CHH, and m6A methylation types.

Example usage:
    python -m scripts.m6a.predictions_to_bam \
        --result_dir <predictions.txt> \
        --input_bam_dir <input.bam> \
        --out_bam_dir <output.bam> \
        --threshold 0.5
"""
import argparse

from tqdm import tqdm
import numpy as np
import pysam
from array import array

from unimeth.ioutils.reader import BamReader
from unimeth.ioutils.reader import TSVReader
from unimeth.utils.bam_tags import get_MM_ML


def convert_results_to_bam(
    result_dir: str,
    input_bam_dir: str,
    out_bam_dir: str,
    threshold: float = 0.5
):
    """
    Convert prediction results to BAM with MM/ML tags.
    
    Args:
        result_dir: Path to prediction TSV file
        input_bam_dir: Path to input BAM file
        out_bam_dir: Path to output BAM file
        threshold: Methylation probability threshold
    """
    # Load predictions using the shared reader
    read_results = TSVReader(result_dir).load_read_results()
    print(f'Loaded {len(read_results)} reads from {result_dir}')
    
    bam = BamReader(input_bam_dir, force_rebuild_index=True)
    
    cnt_read, cnt_pred, cnt_A = 0, 0, 0
    
    with pysam.AlignmentFile(out_bam_dir, "wb", header=bam.bam_file.header, add_sam_header=False) as out_bam:
        for read_id in tqdm(read_results, desc='Processing reads'):
            bam_reads = bam.get_read_by_id(read_id)
            if len(bam_reads) == 0:
                continue
            
            bam_read = bam_reads[0]
            seq = bam_read.get_forward_sequence().upper()
            a_pos = np.array([idx for idx, c in enumerate(seq) if c == 'A'])
            
            # Verify prediction matches sequence
            if len(read_results[read_id]) != len(a_pos):
                continue
            
            # Build positions and scores for MM/ML tags
            positions = []
            scores = []
            for pos in a_pos:
                if pos in read_results[read_id] and read_results[read_id][pos] > threshold:
                    positions.append(pos)
                    scores.append(read_results[read_id][pos] * 100)
            
            if len(positions) == 0:
                continue
            
            MM_tag, ML_tag = get_MM_ML(positions, scores, mod_name='A+a,')
            
            if len(ML_tag) == 0:
                continue
            
            bam_read.set_tag(tag="MM", value=MM_tag, value_type="Z")
            bam_read.set_tag(tag="ML", value=array("B", ML_tag))
            out_bam.write(bam_read)
            
            cnt_read += 1
            cnt_pred += len([p for p in a_pos if p in read_results[read_id] and read_results[read_id][p] > threshold])
            cnt_A += len(a_pos)
    
    methylation_ratio = cnt_pred / cnt_A if cnt_A > 0 else 0
    print(f'Processed {cnt_read} reads. Predicted {cnt_pred} m6A sites, '
          f'methylation ratio={methylation_ratio:.4f}')


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for result conversion."""
    parser = argparse.ArgumentParser(
        description='Convert m6A prediction results to BAM format'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='Path to prediction result TSV file'
    )
    parser.add_argument(
        '--input_bam_dir',
        type=str,
        required=True,
        help='Path to input BAM file'
    )
    parser.add_argument(
        '--out_bam_dir',
        type=str,
        default='output.bam',
        help='Path to output BAM file (default: output.bam)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Methylation probability threshold (default: 0.5)'
    )
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    convert_results_to_bam(
        args.result_dir,
        args.input_bam_dir,
        args.out_bam_dir,
        args.threshold
    )


if __name__ == '__main__':
    main()
