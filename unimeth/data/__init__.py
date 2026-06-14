"""
Data processing module for feature extraction and sequence utilities.

Includes:
- Chromosome filtering and coordinate transformations
- DNA sequence manipulation
- Methylation site detection
- Signal extraction from raw nanopore data
- Sequence patching for model input
- Read processing pipelines
"""
from .coords import (
    parse_chromosome_filter,
    complement_seq,
    get_ref_pos,
    align_to_ref,
)
from .sites import find_methylation_sites, get_methy_type
from .extract import SignalFeatureExtractor
from .patcher import patch_sequence
from .pipeline import get_datasets

__all__ = [
    # coords.py
    'parse_chromosome_filter',
    'complement_seq',
    'get_ref_pos',
    'align_to_ref',
    # sites.py
    'find_methylation_sites',
    'get_methy_type',
    # extract.py
    'SignalFeatureExtractor',
    # patcher.py
    'patch_sequence',
    # pipeline.py
    'get_datasets',
]
