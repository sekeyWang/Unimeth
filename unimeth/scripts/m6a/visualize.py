"""
Visualization module for methylation and nucleosome analysis.

Compares predictions between different methods and visualizes patterns.

Example usage:
    python -m scripts.m6a.visualize \
        --pred_dir <predictions.bam> \
        --baseline_dir <baseline.bam> \
        --fiberseq_dir <fiberseq.csv> \
        --fig_folder <output_folder>
"""
import os
import argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator

from unimeth.ioutils.reader import BamReader

plt.rcParams.update({'font.size': 6})


def extract_fiberseq_data(
    fiberseq_dir: str,
    nrows: int = 10000,
    min_freq: float = 0.02
) -> Tuple[List[int], np.ndarray]:
    """
    Extract nucleosome and m6A data from Fiber-seq CSV.
    
    Args:
        fiberseq_dir: Path to Fiber-seq CSV file
        nrows: Maximum number of rows to read
        min_freq: Minimum m6A frequency threshold
        
    Returns:
        Tuple of (nucleosome_lengths, fiber_data)
    """
    tmp = pd.read_csv(fiberseq_dir, sep="\t", nrows=nrows)
    
    # Filter for full-length fibers
    tmp = tmp[
        (tmp['en'] - tmp['st'] > 0.5 * tmp['fiber_length']) |
        ((tmp['en'] == 0) & (tmp['st'] == 0))
    ]
    
    if 'sam_flag' in tmp.columns:
        tmp = tmp[tmp['sam_flag'] <= 16]
    
    all_nl, fiber_data = [], []
    cnt_all_methy, cnt_pos_methy = 0, 0
    cnt_m6a_reads, cnt_nuclesome = 0, 0
    
    for _, row in tqdm(tmp.iterrows(), total=len(tmp), desc='Processing Fiber-seq'):
        if ',' not in row['m6a'] or ',' not in row['nuc_starts']:
            continue
        
        m6a = np.array(row['m6a'][:-1].split(','), dtype=int)
        m6a_qual = np.array(row['m6a_qual'][:-1].split(','), dtype=int)
        assert len(m6a) == len(m6a_qual)
        
        max_pos = max(m6a) if len(m6a) > 0 else 0
        count_per_fiber = len(m6a)
        
        if max_pos > 0 and count_per_fiber / max_pos > min_freq:
            index = np.zeros(max_pos + 1)
            index[m6a] = 1
            fiber_data.extend(index)
            
            seq = row['fiber_sequence']
            num_a = seq.count('A')
            cnt_all_methy += num_a
            cnt_pos_methy += len(m6a)
            cnt_m6a_reads += 1
            
            nuc_starts = np.array(row['nuc_starts'][:-1].split(','), dtype=int)
            nuc_lengths = np.array(row['nuc_lengths'][:-1].split(','), dtype=int)
            assert len(nuc_starts) == len(nuc_lengths)
            
            all_nl.extend(list(nuc_lengths))
            cnt_nuclesome += 1
    
    if cnt_all_methy > 0:
        print(f'Fiber-seq: {cnt_m6a_reads} reads, {cnt_pos_methy}/{cnt_all_methy} m6A '
              f'(ratio={cnt_pos_methy/cnt_all_methy:.4f})')
    else:
        print('Fiber-seq: No m6A data')
    print(f'Fiber-seq: {cnt_nuclesome} reads, {len(all_nl)} nucleosomes')
    
    fiber_data = np.array(fiber_data, dtype=bool)
    return all_nl, fiber_data


def plot_distance_distribution(
    ax,
    distances: List[int],
    name: str,
    linestyle: Optional[str] = None
):
    """Plot nucleosome distance distribution."""
    bandwidth = 2
    x_vals = np.linspace(0, 500, 1000)
    
    if len(distances) == 0:
        return
    
    kde = gaussian_kde(distances, bw_method=bandwidth / np.std(distances))
    y = kde(x_vals)
    
    ax.plot(x_vals, y, label=name, linestyle=linestyle, linewidth=1)
    peak = x_vals[np.argmax(y)]
    ax.text(peak, y.max(), f"{int(peak)}", ha="center", va="bottom")


def extract_bam_data(
    bam_dir: str,
    read_ids: Optional[List[str]] = None,
    min_freq: float = 0.0
) -> Tuple[List[int], int, int, int]:
    """
    Extract m6A and nucleosome data from BAM file.
    
    Returns:
        Tuple of (nucleosome_lengths, m6a_reads, pos_methy, all_methy)
    """
    bam = BamReader(bam_dir, force_rebuild_index=False)
    
    if read_ids is None:
        read_ids = list(bam.bam_index.keys())
    
    all_nl = []
    cnt_m6a_reads, cnt_pos_methy, cnt_all_methy = 0, 0, 0
    
    for read_id in tqdm(read_ids, desc=f'Processing {os.path.basename(bam_dir)}'):
        bam_reads = bam.get_read_by_id(read_id)
        if len(bam_reads) == 0:
            continue
        
        bam_read = bam_reads[0]
        seq = bam_read.get_forward_sequence().upper()
        
        # Extract m6A
        mod = bam_read.modified_bases_forward
        mod_6ma = mod.get(('A', 0, 'a'), [])
        methy_pos = np.array([x for x, y in mod_6ma if y >= 243])
        
        if len(methy_pos) > 0:
            max_pos = max(methy_pos)
            if max_pos > 0 and len(methy_pos) / max_pos > min_freq:
                cnt_m6a_reads += 1
                cnt_pos_methy += len(methy_pos)
                cnt_all_methy += len(mod_6ma)
        
        # Extract nucleosomes
        if bam_read.has_tag('ns') and bam_read.has_tag('nl'):
            nl = bam_read.get_tag('nl')
            all_nl.extend(list(nl))
    
    return all_nl, cnt_m6a_reads, cnt_pos_methy, cnt_all_methy


def draw_distance_comparison(
    bam_dirs: List[str],
    names: List[str],
    fig_folder: str,
    fiberseq_dir: str,
    min_freq: float = 0.0
):
    """
    Draw distance distribution comparison between methods.
    
    Args:
        bam_dirs: List of BAM file paths
        names: List of method names
        fig_folder: Output folder for figures
        fiberseq_dir: Fiber-seq CSV file path
        min_freq: Minimum m6A frequency threshold
    """
    fig1, ax1 = plt.subplots(figsize=(8/2.54, 4.5/2.54))
    
    # Load Fiber-seq data
    all_nl_fiber, _ = extract_fiberseq_data(fiberseq_dir, nrows=10000, min_freq=0.02)
    plot_distance_distribution(ax1, all_nl_fiber, 'Fibertools (PacBio)')
    
    # Find common reads between first two BAMs
    if len(bam_dirs) >= 2:
        bam0 = BamReader(bam_dirs[0], force_rebuild_index=False)
        bam1 = BamReader(bam_dirs[1], force_rebuild_index=False)
        read_ids0 = set(bam0.bam_index.keys())
        read_ids1 = set(bam1.bam_index.keys())
        common_reads = sorted(list(read_ids0 & read_ids1))[:10000]
        print(f'Common reads: {len(common_reads)}')
    else:
        common_reads = None
    
    # Process each BAM
    for bam_dir, name in zip(bam_dirs, names):
        all_nl, cnt_reads, cnt_pos, cnt_all = extract_bam_data(
            bam_dir, common_reads, min_freq
        )
        
        if cnt_all > 0:
            ratio = cnt_pos / cnt_all
            print(f'{name}: {cnt_reads} reads, {cnt_pos}/{cnt_all} m6A (ratio={ratio:.4f})')
        print(f'{name}: {len(all_nl)} nucleosomes')
        
        plot_distance_distribution(ax1, all_nl, name)
    
    # Save figure
    os.makedirs(fig_folder, exist_ok=True)
    
    ax1.legend()
    ax1.axvline(147, linestyle="--", color="black", alpha=0.5)
    ax1.set_xlim(0, 500)
    ax1.set_xlabel("Distance between adjacent m6A")
    ax1.set_ylabel("Density")
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig1.tight_layout()
    
    dist_path = os.path.join(fig_folder, 'dist.pdf')
    fig1.savefig(dist_path)
    print(f'Distance figure saved to {dist_path}')


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize m6A and nucleosome patterns'
    )
    parser.add_argument(
        '--pred_dir',
        type=str,
        required=True,
        help='Path to prediction BAM file'
    )
    parser.add_argument(
        '--dorado_dir',
        type=str,
        required=True,
        help='Path to Dorado BAM file'
    )
    parser.add_argument(
        '--fig_folder',
        type=str,
        required=True,
        help='Output folder for figures'
    )
    parser.add_argument(
        '--fiberseq_dir',
        type=str,
        required=True,
        help='Path to Fiber-seq CSV file'
    )
    parser.add_argument(
        '--min_freq',
        type=float,
        default=0.0,
        help='Minimum m6A frequency threshold (default: 0.0)'
    )
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    draw_distance_comparison(
        [args.pred_dir, args.dorado_dir],
        ['Unimeth (ONT)', 'Dorado (ONT)'],
        args.fig_folder,
        args.fiberseq_dir,
        args.min_freq
    )


if __name__ == '__main__':
    main()
