"""
Inference module for UniMeth.
"""
from .engine import InferenceEngine
from unimeth.model.loader import load_model

__all__ = [
    'InferenceEngine',
    'load_model',
]
