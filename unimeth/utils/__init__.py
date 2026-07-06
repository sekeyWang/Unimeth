"""
Utilities module for UniMeth.
"""
from .common import (
    local_print,
    token2seq,
    get_cpu_count,
    make_output_path,
    merge_rank_files,
)
from .bam_tags import (
    get_MM_ML,
    get_mod_config,
    get_target_positions,
    get_modifications,
)

__all__ = [
    # common
    'local_print',
    'token2seq',
    'get_cpu_count',
    'make_output_path',
    'merge_rank_files',
    # bam_tags
    'get_MM_ML',
    'get_mod_config',
    'get_target_positions',
    'get_modifications',
    # callbacks
    'MetricsCallback',
    'EarlyStoppingCallback',
]


def __getattr__(name):
    if name in {'MetricsCallback', 'EarlyStoppingCallback'}:
        from .callbacks import MetricsCallback, EarlyStoppingCallback
        globals().update({
            'MetricsCallback': MetricsCallback,
            'EarlyStoppingCallback': EarlyStoppingCallback,
        })
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
