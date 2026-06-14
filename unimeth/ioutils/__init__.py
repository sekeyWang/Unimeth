"""
IO module for reading and writing various file formats.

Organized into two submodules:
- io.reader: File reading utilities
- io.writer: File writing utilities
"""

# Reader imports
from .reader import BamReader, SignalReader, TSVReader, BEDReader, PredictionRecord

# Writer imports
from .writer import LabelBAMWriter, AggregationBAMWriter, TSVWriter

__all__ = [
    # Readers
    'BamReader',
    'SignalReader',
    'TSVReader',
    'BEDReader',
    'PredictionRecord',
    # Writers
    'LabelBAMWriter',
    'AggregationBAMWriter',
    'TSVWriter',
]
