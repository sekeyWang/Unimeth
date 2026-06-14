"""
Bisulfite evaluation script for comparing predictions against ground truth.

Supports both TSV and BAM format predictions.

Example usage:
    python -m scripts.evaluate \
        --tsv_dir <predictions.txt> \
        --CpG_bed_dir <cpg_labels.bed>

    python -m scripts.evaluate \
        --bam_dir <predictions.bam> \
        --CpG_bed_dir <cpg_labels.bed>
"""
import argparse
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from unimeth.ioutils.reader import TSVReader, BEDReader, BamReader
from unimeth.eval.metrics import compute_correlation_metrics, compute_classification_metrics


def evaluate_predictions(
    pred_dir: str,
    bisulfite: Dict[str, Optional[Dict]],
    mod_type: str = 'm'
):
    """
    Compare predictions against bisulfite ground truth.
    
    Args:
        pred_dir: Path to prediction file (TSV or BAM)
        bisulfite: Dictionary mapping methylation types to label dictionaries
        mod_type: Modification type for BAM reading ('m' for CpG, 'a' for m6A)
    """
    is_bam = pred_dir.endswith('.bam')
    if is_bam:
        bam_reader = BamReader(pred_dir)
        site_result_all = bam_reader.load_site_predictions(mod_type=mod_type)
        tsv_all_results = None
    else:
        # Load all contexts in a single pass to avoid reading the file 3 times
        tsv_all_results = {'[CpG]': {}, '[CHG]': {}, '[CHH]': {}}
        tsv_reader = TSVReader(pred_dir)
        for record in tsv_reader.iter_records():
            if record.ref_pos == -1:
                continue
            if record.methy_type not in tsv_all_results:
                continue
            name = f'{record.chrom}_{record.ref_pos}'
            tsv_all_results[record.methy_type].setdefault(name, []).append(record.prob_methylated)

    # Evaluate each methylation type
    for methy_type, type_label in bisulfite.items():
        if type_label is None:
            print(f'No ground truth labels for {methy_type}')
            continue

        print(f'\nEvaluating {methy_type}')

        if is_bam:
            site_result = site_result_all
        else:
            site_result = tsv_all_results[methy_type]
        
        # Collect site-level predictions
        site_preds, site_truth = [], []
        read_true, read_pred = [], []
        
        for name in tqdm(site_result, desc='Comparing sites'):
            if name not in type_label:
                continue
            
            preds = site_result[name]
            site_pred_ratio = np.sum(np.array(preds) > 0.5) / len(preds)
            site_truth_ratio = type_label[name] / 100
            
            # Site-level (coverage >= 5)
            if len(preds) >= 5:
                site_preds.append(site_pred_ratio)
                site_truth.append(site_truth_ratio)
            
            # Read-level (binary labels only)
            if site_truth_ratio == 0 or site_truth_ratio == 1:
                read_true.extend([int(site_truth_ratio)] * len(preds))
                read_pred.extend(preds)
        
        site_truth = np.array(site_truth)
        site_preds = np.array(site_preds)
        read_true = np.array(read_true, dtype=int)
        read_pred = np.array(read_pred, dtype=float)
        
        # Compute and print metrics
        print('\nRead-level metrics:')
        print(compute_classification_metrics(read_true, read_pred))
        
        print('\nSite-level metrics:')
        print(compute_correlation_metrics(site_truth, site_preds))


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation."""
    parser = argparse.ArgumentParser(
        description='Compare predictions against bisulfite ground truth'
    )
    parser.add_argument(
        '--bam_dir',
        type=str,
        default=None,
        help='Path to prediction BAM file'
    )
    parser.add_argument(
        '--tsv_dir',
        type=str,
        default=None,
        help='Path to prediction TSV file'
    )
    parser.add_argument(
        '--CpG_bed_dir',
        type=str,
        default=None,
        help='Path to CpG bisulfite BED file'
    )
    parser.add_argument(
        '--CHG_bed_dir',
        type=str,
        default=None,
        help='Path to CHG bisulfite BED file'
    )
    parser.add_argument(
        '--CHH_bed_dir',
        type=str,
        default=None,
        help='Path to CHH bisulfite BED file'
    )
    parser.add_argument(
        '--mod_type',
        type=str,
        default='m',
        choices=['m', 'a'],
        help="Modification type in BAM ('m' for CpG/CHG/CHH, 'a' for m6A)"
    )
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Determine prediction source
    if args.bam_dir is not None:
        pred_dir = args.bam_dir
    elif args.tsv_dir is not None:
        pred_dir = args.tsv_dir
    else:
        parser.error('Either --bam_dir or --tsv_dir must be provided')
    
    # Load bisulfite labels
    bis_CpG = BEDReader(args.CpG_bed_dir).load_labels() if args.CpG_bed_dir else None
    bis_CHG = BEDReader(args.CHG_bed_dir).load_labels() if args.CHG_bed_dir else None
    bis_CHH = BEDReader(args.CHH_bed_dir).load_labels() if args.CHH_bed_dir else None
    
    bisulfite = {
        '[CpG]': bis_CpG,
        '[CHG]': bis_CHG,
        '[CHH]': bis_CHH
    }
    
    evaluate_predictions(pred_dir, bisulfite, mod_type=args.mod_type)


if __name__ == '__main__':
    main()
