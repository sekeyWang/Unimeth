"""Public reader exports loaded lazily by their consumers."""

from importlib import import_module

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
    'SignalIndexWriteError',
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
    'resolve_signal_index_path',
    'resolve_bam_mode',
    'resolve_bam_mode_from_path',
]


_EXPORTS = {
    'BamReader': ('.bam', 'BamReader'),
    'BamOffsetReader': ('.bam_stream', 'BamOffsetReader'),
    'BamStreamItem': ('.bam_stream', 'BamStreamItem'),
    'BamStreamReader': ('.bam_stream', 'BamStreamReader'),
    'BamStreamStats': ('.bam_stream', 'BamStreamStats'),
    'SerializedBamStreamItem': ('.bam_stream', 'SerializedBamStreamItem'),
    'BamSignalFeatureStream': ('.bam_signal_stream', 'BamSignalFeatureStream'),
    'BamSignalFeatureStreamStats': (
        '.bam_signal_stream',
        'BamSignalFeatureStreamStats',
    ),
    'DuplicateSignalReadIdError': (
        '.signal_index',
        'DuplicateSignalReadIdError',
    ),
    'SignalBatchLookup': ('.signal_lookup', 'SignalBatchLookup'),
    'SignalIndexProgress': ('.signal_index', 'SignalIndexProgress'),
    'SignalIndexStats': ('.signal_index', 'SignalIndexStats'),
    'SignalIndexWriteError': ('.signal_index', 'SignalIndexWriteError'),
    'SignalLookupBatch': ('.signal_lookup', 'SignalLookupBatch'),
    'SignalLookupRequest': ('.signal_lookup', 'SignalLookupRequest'),
    'SignalReadRouter': ('.signal_index', 'SignalReadRouter'),
    'SignalRouteIndex': ('.signal_index', 'SignalRouteIndex'),
    'SignalRoutingPlan': ('.signal_index', 'SignalRoutingPlan'),
    'SignalSourceHintError': ('.signal_lookup', 'SignalSourceHintError'),
    'SignalReader': ('.signal', 'SignalReader'),
    'TSVReader': ('.tsv', 'TSVReader'),
    'BEDReader': ('.bed', 'BEDReader'),
    'PredictionRecord': ('.tsv', 'PredictionRecord'),
    'calculate_alignment_identity': (
        '.bam_stream',
        'calculate_alignment_identity',
    ),
    'build_signal_route_index': ('.signal_index', 'build_signal_route_index'),
    'get_signal_read_id': ('.bam_stream', 'get_signal_read_id'),
    'get_signal_source_hint': ('.bam_stream', 'get_signal_source_hint'),
    'prepare_signal_routing': ('.signal_index', 'prepare_signal_routing'),
    'resolve_signal_index_path': ('.signal_index', 'resolve_signal_index_path'),
    'resolve_bam_mode': ('.bam_stream', 'resolve_bam_mode'),
    'resolve_bam_mode_from_path': ('.bam_stream', 'resolve_bam_mode_from_path'),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
