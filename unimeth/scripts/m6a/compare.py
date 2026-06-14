"""
Compare predictions between two methods (e.g., UniMeth vs Dorado).

Analyzes and compares methylation patterns between two prediction sources.

Example usage:
    python -m scripts.m6a.compare \
        --result_tsv <predictions.txt> \
        --baseline_bam <baseline.bam> \
        --mod_type A+a
"""
import argparse
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

from unimeth.ioutils.reader import BamReader
from unimeth.ioutils.reader import TSVReader
from unimeth.utils.bam_tags import get_mod_config


def compare_predictions(
    result_tsv: str,
    baseline_bam: str,
    mod_type: str = 'A+a',
    threshold: float = 0.5
) -> Tuple[Dict, Dict]:
    """
    Compare predictions with baseline.
    
    Args:
        result_tsv: Path to prediction TSV file
        baseline_bam: Path to baseline BAM file (with MM/ML tags)
        mod_type: Modification type string (e.g., 'A+a', 'C+m')
        threshold: Prediction threshold
        
    Returns:
        Tuple of (our_stats, baseline_stats) dictionaries
    """
    # Load predictions
    result_dict = TSVReader(result_tsv).load_read_results()
    print(f'Loaded {len(result_dict)} reads from {result_tsv}')
    
    bam = BamReader(baseline_bam, force_rebuild_index=False)
    
    # Get mod configuration
    _, mod_key = get_mod_config(mod_type)
    
    cnt_read, cnt_pred, cnt_baseline, cnt_base = 0, 0, 0, 0
    
    for read_id in tqdm(result_dict, desc='Processing reads'):
        bam_reads = bam.get_read_by_id(read_id)
        if len(bam_reads) == 0:
            continue
        
        bam_read = bam_reads[0]
        seq = bam_read.get_forward_sequence().upper()
        base_char = mod_type[0]  # 'A' or 'C'
        base_pos = np.array([idx for idx, c in enumerate(seq) if c == base_char])
        
        if len(result_dict[read_id]) != len(base_pos):
            continue
        
        # Get our predictions
        pred_result = [result_dict[read_id][p] for p in base_pos]
        methy_idx = np.where(np.array(pred_result) > threshold)[0]
        methy_pos_pred = base_pos[methy_idx]
        
        # Get baseline predictions
        mod = bam_read.modified_bases_forward
        mod_baseline = mod.get(mod_key, [])
        methy_pos_baseline = np.array([x for x, y in mod_baseline if y > 127])
        
        cnt_read += 1
        cnt_pred += len(methy_pos_pred)
        cnt_baseline += len(methy_pos_baseline)
        cnt_base += len(base_pos)
    
    ratio_pred = cnt_pred / cnt_base if cnt_base > 0 else 0
    ratio_baseline = cnt_baseline / cnt_base if cnt_base > 0 else 0
    
    our_stats = {
        'reads': cnt_read,
        'methylated_sites': cnt_pred,
        'total_sites': cnt_base,
        'ratio': ratio_pred
    }
    baseline_stats = {
        'reads': cnt_read,
        'methylated_sites': cnt_baseline,
        'total_sites': cnt_base,
        'ratio': ratio_baseline
    }
    
    print(f'\nProcessed {cnt_read} reads.')
    print(f'Our predictions: {cnt_pred} (ratio={ratio_pred:.4f})')
    print(f'Baseline predictions: {cnt_baseline} (ratio={ratio_baseline:.4f})')
    
    return our_stats, baseline_stats


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for comparison."""
    parser = argparse.ArgumentParser(
        description='Compare predictions with baseline'
    )
    parser.add_argument(
        '--result_tsv',
        type=str,
        required=True,
        help='Path to prediction TSV file'
    )
    parser.add_argument(
        '--baseline_bam',
        type=str,
        required=True,
        help='Path to baseline BAM with MM/ML tags'
    )
    parser.add_argument(
        '--mod_type',
        type=str,
        default='A+a',
        help='Modification type (default: A+a for m6A)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Prediction threshold (default: 0.5)'
    )
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    compare_predictions(
        args.result_tsv,
        args.baseline_bam,
        args.mod_type,
        args.threshold
    )


if __name__ == '__main__':
    main()
