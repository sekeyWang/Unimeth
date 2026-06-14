"""
BAM tag utilities for methylation data.

This module provides functions for generating MM/ML tags used in BAM files
to store methylation modification information.
"""
import numpy as np
from array import array
from typing import Dict, List, Tuple, Optional

# Modification type to (base, mod_key) mapping
MOD_TYPE_CONFIG = {
    'm6A': ('A', ('A', 0, 'a')),
    'm': ('C', ('C', 0, 'm')),
    'CpG': ('C', ('C', 0, 'm')),
    'CHG': ('C', ('C', 0, 'm')),
    'CHH': ('C', ('C', 0, 'm')),
    'a': ('A', ('A', 0, 'a')),
}


def get_mod_config(mod_type: str) -> Tuple[str, Tuple]:
    """
    Get base character and mod_key for a modification type.
    
    Args:
        mod_type: Modification type ('m6A', 'CpG', 'CHG', 'CHH', 'm', 'a')
        
    Returns:
        Tuple of (base_char, mod_key)
        - base_char: 'A' or 'C'
        - mod_key: Tuple (base, strand, modification_code) for pysam
        
    Example:
        >>> get_mod_config('m6A')
        ('A', ('A', 0, 'a'))
        >>> get_mod_config('CpG')
        ('C', ('C', 0, 'm'))
    """
    if mod_type not in MOD_TYPE_CONFIG:
        raise ValueError(f"Unknown mod_type: {mod_type}. Supported: {list(MOD_TYPE_CONFIG.keys())}")
    return MOD_TYPE_CONFIG[mod_type]


def get_target_positions(seq: str, base: str) -> List[int]:
    """
    Find all positions of a base in a sequence.
    
    Args:
        seq: DNA sequence string
        base: Base to find ('A' or 'C')
        
    Returns:
        List of positions (0-indexed) where base occurs
    """
    return [i for i, c in enumerate(seq.upper()) if c == base]


def get_modifications(bam_read, mod_key: Tuple) -> Dict[int, int]:
    """
    Get modification dictionary from BAM read.
    
    Args:
        bam_read: pysam AlignedSegment
        mod_key: Modification key tuple (base, strand, mod_code)
        
    Returns:
        Dictionary mapping position to modification score (0-255)
    """
    mod = bam_read.modified_bases_forward
    mod_list = mod.get(mod_key, [])
    return {pos: score for pos, score in mod_list}


def get_MM_ML(
    positions: List[int],
    scores: List[float],
    mod_name: str = 'C+m,'
) -> Tuple[str, array]:
    """
    Generate MM and ML tags for a read.

    Args:
        positions: List of positions (0-indexed) with modification signal
        scores: List of methylation scores (0-100) corresponding to positions
        mod_name: Modification name for MM tag (default: 'C+m,')

    Returns:
        Tuple of (MM_tag_string, ML_tag_array)

    Example:
        >>> positions = [100, 150]
        >>> scores = [100, 0]
        >>> mm, ml = get_MM_ML(positions, scores)
    """
    if len(positions) == 0 or len(scores) == 0:
        return '', array('B')
    
    if len(positions) != len(scores):
        raise ValueError(f"positions and scores must have same length: {len(positions)} vs {len(scores)}")
    
    positions = np.array(positions)
    scores = np.array(scores)
    
    # positions are indices within all same-type bases in the read.
    # Insert dummy -1 at beginning for computing per-modification skip counts.
    valid_idx = np.insert(positions, 0, -1)
    sep = valid_idx[1:] - valid_idx[:-1] - 1
    str_sep = [str(x) for x in sep]
    MM_tag = mod_name + ','.join(str_sep) + ';'
    ML_tag = np.array(np.round(scores * 255 / 100), dtype=int)

    return MM_tag, array('B', ML_tag)


def write_mm_ml_tags(bam_read, positions: List[int], scores: List[float],
                     mod_name: str = 'C+m,') -> bool:
    """
    Generate and set MM/ML tags on a BAM read.

    Args:
        bam_read: pysam AlignedSegment to modify
        positions: 0-based indices of modified bases within all same-type bases in the read
        scores: Methylation scores (0-100) for each modified base
        mod_name: Modification name for MM tag

    Returns:
        True if tags were set successfully, False otherwise
    """
    if len(positions) == 0 or len(scores) == 0:
        return False
    
    MM_tag, ML_tag = get_MM_ML(positions, scores, mod_name)
    if len(ML_tag) == 0:
        return False
    
    bam_read.set_tag(tag="MM", value=MM_tag, value_type="Z")
    bam_read.set_tag(tag="ML", value=array("B", ML_tag))
    return True
