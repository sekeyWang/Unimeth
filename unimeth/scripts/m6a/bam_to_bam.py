"""
Convert BAM files with methylation tags to training-ready BAM format.

Supports extracting labels from existing MM/ML tags in BAM files
and creating training data with proper labels.

Example usage:
    python -m scripts.m6a.bam_to_bam \
        --input_bam <input.bam> \
        --output_folder <output_folder> \
        --mod_type m6A \
        --min_confidence 0.9
"""
import os
import argparse
import multiprocessing
from typing import Set, Dict
import numpy as np
import pysam
from array import array


from unimeth.utils.bam_tags import get_MM_ML, get_mod_config, get_target_positions, get_modifications



def get_modification_positions(bam_read, mod_type: str) -> tuple:
    """
    Get positions of target bases and existing modifications.
    
    Args:
        bam_read: pysam AlignedSegment
        mod_type: Modification type ('CpG', 'CHG', 'CHH', 'm6A')
        
    Returns:
        Tuple of (target_positions, mod_dict, mod_key)
    """
    seq = bam_read.get_forward_sequence()
    if seq is None:
        return [], {}, None
    
    base, mod_key = get_mod_config(mod_type)
    target_pos = get_target_positions(seq, base)
    mod_dict = get_modifications(bam_read, mod_key)
    
    return target_pos, mod_dict, mod_key


def create_training_labels(
    target_pos: list,
    mod_dict: dict,
    min_confidence: float = 0.9,
    use_nucleosome: bool = False,
    bam_read = None
) -> Dict[str, int]:
    """
    Create training labels from modification data.
    
    Args:
        target_pos: List of target base positions
        mod_dict: Dictionary of {pos: modification_score}
        min_confidence: Minimum confidence threshold (0-1)
        use_nucleosome: Whether to use nucleosome positioning for labeling
        bam_read: BAM read object (required if use_nucleosome=True)
        
    Returns:
        Dictionary mapping position_key to label (0 or 100)
    """
    label = {}
    
    if use_nucleosome and bam_read is not None:
        # Use nucleosome positioning (for m6A training data)
        if not bam_read.has_tag('ns') or not bam_read.has_tag('nl'):
            return {}
        
        ns = np.array(bam_read.get_tag('ns'))
        nl = np.array(bam_read.get_tag('nl'))
        
        # Label positions near nucleosome boundaries
        left = ns - 1
        right = ns + nl
        union: Set[int] = set(left).union(set(right))
        
        for pos in target_pos:
            name = f'pos_{pos}'
            if pos in union:
                label[name] = 100  # Methylated
            elif pos in mod_dict and mod_dict[pos] == 0:
                label[name] = 0    # Unmethylated (explicitly unmodified)
    else:
        # Standard threshold-based labeling
        threshold = int(min_confidence * 255)
        
        for pos in target_pos:
            name = f'pos_{pos}'
            if pos in mod_dict:
                if mod_dict[pos] >= threshold:
                    label[name] = 100  # Methylated
                elif mod_dict[pos] == 0:
                    label[name] = 0    # Unmethylated
            # Skip ambiguous positions (0 < score < threshold)
    
    return label


def process_bam_worker(
    input_bam: str,
    output_folder: str,
    pid: int,
    num_process: int,
    mod_type: str,
    min_confidence: float,
    use_nucleosome: bool
):
    """
    Process BAM file to create training data (worker function).
    
    Args:
        input_bam: Input BAM file path
        output_folder: Output folder path
        pid: Process ID
        num_process: Total number of processes
        mod_type: Modification type
        min_confidence: Minimum confidence threshold
        use_nucleosome: Whether to use nucleosome positioning
    """
    out_path = os.path.join(output_folder, f'{pid}.bam')
    
    with pysam.AlignmentFile(input_bam, "rb") as in_bam, \
         pysam.AlignmentFile(out_path, "wb", header=in_bam.header, add_sam_header=False) as out_bam:
        
        for idx, bam_read in enumerate(in_bam):
            if idx % num_process != pid:
                continue
            
            # Skip low-quality alignments
            if bam_read.is_supplementary or bam_read.is_secondary:
                continue
            if bam_read.get_forward_sequence() is None:
                continue
            
            # Get modification positions
            target_pos, mod_dict, _ = get_modification_positions(bam_read, mod_type)
            
            if len(target_pos) == 0:
                continue
            
            # Create labels
            label = create_training_labels(
                target_pos, mod_dict, min_confidence,
                use_nucleosome=use_nucleosome,
                bam_read=bam_read
            )
            
            if len(label) == 0:
                continue
            
            # Build index-in-A-array and scores from label {pos_{pos}: score}
            # MM tag skip counts are relative to the target base array, not absolute positions
            positions = []
            scores = []
            for i, pos in enumerate(target_pos):
                key = f'pos_{pos}'
                if key in label:
                    positions.append(i)
                    scores.append(label[key])
            
            if len(positions) == 0:
                continue
            
            # Generate MM/ML tags
            mod_name = 'A+a,' if mod_type == 'm6A' else 'C+m,'
            MM_tag, ML_tag = get_MM_ML(positions, scores, mod_name=mod_name)
            
            bam_read.set_tag(tag="MM", value=MM_tag, value_type="Z")
            bam_read.set_tag(tag="ML", value=array("B", ML_tag))
            out_bam.write(bam_read)


def convert_bam_to_training_data(
    input_bam: str,
    output_folder: str,
    mod_type: str = 'm6A',
    min_confidence: float = 0.9,
    use_nucleosome: bool = False,
    num_process: int = None
):
    """
    Convert BAM with methylation tags to training-ready BAM.
    
    Args:
        input_bam: Input BAM file path
        output_folder: Output folder path
        mod_type: Modification type ('CpG', 'CHG', 'CHH', 'm6A')
        min_confidence: Minimum confidence threshold for positive labels
        use_nucleosome: Use nucleosome positioning (for m6A)
        num_process: Number of parallel processes (default: CPU count)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if num_process is None:
        num_process = multiprocessing.cpu_count()
    
    print(f'Converting {input_bam} to training data...')
    print(f'Modification type: {mod_type}')
    print(f'Using {num_process} processes')
    
    processes = []
    for pid in range(num_process):
        p = multiprocessing.Process(
            target=process_bam_worker,
            args=(input_bam, output_folder, pid, num_process,
                  mod_type, min_confidence, use_nucleosome)
        )
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print(f'Training data written to {output_folder}')


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for BAM to BAM conversion."""
    parser = argparse.ArgumentParser(
        description='Convert BAM with methylation tags to training-ready BAM'
    )
    parser.add_argument(
        '--input_bam',
        type=str,
        required=True,
        help='Input BAM file with methylation tags'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Output folder for training BAM files'
    )
    parser.add_argument(
        '--mod_type',
        type=str,
        choices=['CpG', 'CHG', 'CHH', 'm6A'],
        default='m6A',
        help='Modification type (default: m6A)'
    )
    parser.add_argument(
        '--min_confidence',
        type=float,
        default=0.9,
        help='Minimum confidence threshold for positive labels (default: 0.9)'
    )
    parser.add_argument(
        '--use_nucleosome',
        action='store_true',
        help='Use nucleosome positioning for labeling (m6A only)'
    )
    parser.add_argument(
        '--num_process',
        type=int,
        default=None,
        help='Number of parallel processes (default: CPU count)'
    )
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    convert_bam_to_training_data(
        input_bam=args.input_bam,
        output_folder=args.output_folder,
        mod_type=args.mod_type,
        min_confidence=args.min_confidence,
        use_nucleosome=args.use_nucleosome,
        num_process=args.num_process
    )


if __name__ == '__main__':
    main()
