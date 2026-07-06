"""
Inference module for UniMeth.
"""

__all__ = [
    'InferenceEngine',
    'load_model',
]


def __getattr__(name):
    if name == 'InferenceEngine':
        from .engine import InferenceEngine
        return InferenceEngine
    if name == 'load_model':
        from unimeth.model.loader import load_model
        return load_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
