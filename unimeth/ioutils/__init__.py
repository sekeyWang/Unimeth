"""Public I/O exports loaded on demand to avoid unrelated reader side effects."""

from importlib import import_module

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


_EXPORTS = {
    'BamReader': ('.reader.bam', 'BamReader'),
    'SignalReader': ('.reader.signal', 'SignalReader'),
    'TSVReader': ('.reader.tsv', 'TSVReader'),
    'BEDReader': ('.reader.bed', 'BEDReader'),
    'PredictionRecord': ('.reader.tsv', 'PredictionRecord'),
    'LabelBAMWriter': ('.writer.bam_label', 'LabelBAMWriter'),
    'AggregationBAMWriter': ('.writer.bam_aggregation', 'AggregationBAMWriter'),
    'TSVWriter': ('.writer.tsv', 'TSVWriter'),
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
