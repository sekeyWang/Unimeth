"""
Output writers for UniMeth.

Provides writers for different output formats:
- TSVWriter: TSV writer with background-thread I/O for inference results
- LabelBAMWriter: BAM writer for calibration annotation (pre-computed labels)
- AggregationBAMWriter: BAM writer for inference (streaming patch aggregation)
"""
from .tsv import TSVWriter
from .bam_label import LabelBAMWriter
from .bam_aggregation import AggregationBAMWriter

__all__ = [
    'TSVWriter',
    'LabelBAMWriter',
    'AggregationBAMWriter',
]
