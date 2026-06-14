"""
Configuration module for UniMeth.

This module provides configuration classes and constants for the UniMeth model.
All backward compatibility aliases are defined here to keep definitions clean.
"""
from dataclasses import asdict

# Import base definitions from submodules
from .model_config import (
    ModelConfig,           # Model architecture config (dataclass)
    DataConfig,            # Data/training config (dataclass)
    VOCAB,                 # Vocabulary list
    TOKENIZER,             # Token to ID mapping
    METHYLATION_TOKENS,    # Methylation token IDs
    METHYLATION_LABELS,    # Label token IDs (+/-)
)
from .args_config import create_argument_parser, merge_with_default_config

# =============================================================================
# Default instances and backward compatibility aliases
# =============================================================================

# DataConfig default instance (for creating defaultconfig dict)
_default_data_config = DataConfig()

# Backward compatibility: defaultconfig as dict (use DataConfig for new code)
defaultconfig = asdict(_default_data_config)
defaultconfig['vocab'] = VOCAB  # Add vocab for compatibility

# Tokenizer aliases
tokenizer = TOKENIZER
vocab = VOCAB  # Backward compatibility
methylation_token_map = METHYLATION_TOKENS
methylation_labels = METHYLATION_LABELS

# Methylation types list (for backward compatibility)
methy_types = list(METHYLATION_TOKENS.keys())

# Model config dicts (for backward compatibility only - will be removed)
modelconfig = ModelConfig.from_name("default").to_dict()
default_modelconfig = modelconfig
distilled_modelconfig = ModelConfig.from_name("distilled").to_dict()


def get_total_stride(model_type: str = "default") -> int:
    """Get total CNN stride for a given model type.
    
    Args:
        model_type: Model type name ('default', 'distilled', or path to JSON)
    
    Returns:
        Total stride (product of all CNN strides)
    """
    if model_type.endswith('.json'):
        config = ModelConfig.from_json(model_type)
    else:
        config = ModelConfig.from_name(model_type)
    return config.total_stride


# =============================================================================
# Public exports
# =============================================================================

__all__ = [
    # Dataclass-based configs (recommended for new code)
    'ModelConfig',
    'DataConfig',
    
    # Backward compatibility (dict-based, deprecated but maintained)
    'defaultconfig',
    'modelconfig',
    'default_modelconfig',
    'distilled_modelconfig',
    
    # Constants
    'VOCAB',
    'vocab',  # Backward compatibility alias
    'tokenizer',
    'methy_types',
    'methylation_token_map',
    'methylation_labels',
    
    # Functions
    'create_argument_parser',
    'merge_with_default_config',
    'get_total_stride',
]
