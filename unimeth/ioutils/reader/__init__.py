"""
IO readers for UniMeth.

Provides readers for various input formats:
- BamReader: BAM file with indexing and prediction loading
- SignalReader: POD5 + BAM feature extraction  
- TSVReader: TSV prediction result reader
- BEDReader: BED bisulfite label reader
"""
from .bam import BamReader
from .signal import SignalReader
from .tsv import TSVReader, PredictionRecord
from .bed import BEDReader

__all__ = [
    'BamReader',
    'SignalReader',
    'TSVReader',
    'BEDReader',
    'PredictionRecord',
]
