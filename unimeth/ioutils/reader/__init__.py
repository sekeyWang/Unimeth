"""
IO readers for UniMeth.

Provides readers for various input formats:
- BamReader: BAM file with indexing and prediction loading
- SignalReader: POD5 + BAM feature extraction
- TSVReader: TSV prediction result reader
- BEDReader: BED bisulfite label reader
"""
from .bam import BamReader
from .bam_stream import (
    BamOffsetReader,
    BamStreamItem,
    BamStreamReader,
    BamStreamStats,
    SerializedBamStreamItem,
    calculate_alignment_identity,
    get_signal_read_id,
    get_signal_source_hint,
    resolve_bam_mode,
    resolve_bam_mode_from_path,
)
from .bam_signal_stream import (
    BamSignalFeatureStream,
    BamSignalFeatureStreamStats,
)
from .signal_index import (
    DuplicateSignalReadIdError,
    SignalIndexProgress,
    SignalIndexStats,
    SignalReadRouter,
    SignalRouteIndex,
    SignalRoutingPlan,
    build_signal_route_index,
    prepare_signal_routing,
)
from .signal_lookup import (
    SignalBatchLookup,
    SignalLookupBatch,
    SignalLookupRequest,
    SignalSourceHintError,
)
from .signal import SignalReader
from .tsv import TSVReader, PredictionRecord
from .bed import BEDReader

__all__ = [
    'BamReader',
    'BamOffsetReader',
    'BamStreamItem',
    'BamStreamReader',
    'BamStreamStats',
    'SerializedBamStreamItem',
    'BamSignalFeatureStream',
    'BamSignalFeatureStreamStats',
    'DuplicateSignalReadIdError',
    'SignalBatchLookup',
    'SignalIndexProgress',
    'SignalIndexStats',
    'SignalLookupBatch',
    'SignalLookupRequest',
    'SignalReadRouter',
    'SignalRouteIndex',
    'SignalRoutingPlan',
    'SignalSourceHintError',
    'SignalReader',
    'TSVReader',
    'BEDReader',
    'PredictionRecord',
    'calculate_alignment_identity',
    'build_signal_route_index',
    'get_signal_read_id',
    'get_signal_source_hint',
    'prepare_signal_routing',
    'resolve_bam_mode',
    'resolve_bam_mode_from_path',
]
