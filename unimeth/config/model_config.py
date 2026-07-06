"""
Model and training configuration definitions.
"""
import os
import time
import json
from pathlib import Path
from importlib import resources
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

# Set timezone
os.environ['TZ'] = 'Asia/Shanghai'
if hasattr(time, 'tzset'):
    time.tzset()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# Vocabulary (fixed)
VOCAB = ['[PAD]', '[START]', '[END]', 'A', 'G', 'T', 'C', 
         '[CpG]', '[CHG]', '[CHH]', '+', '-', '[MASK]', '[m6A]', 
         '[R10]', '[4khz]', '[5khz]']

# Token mappings (fixed)
TOKENIZER = {v: i for i, v in enumerate(VOCAB)}
METHYLATION_TOKENS = {
    '[CpG]': TOKENIZER['[CpG]'],
    '[CHG]': TOKENIZER['[CHG]'],
    '[CHH]': TOKENIZER['[CHH]'],
    '[m6A]': TOKENIZER['[m6A]'],
}
METHYLATION_LABELS = {'+': TOKENIZER['+'], '-': TOKENIZER['-']}


def get_config_dir() -> Path:
    """Get the packaged model configs directory."""
    config_dir = resources.files("unimeth.configs")
    if isinstance(config_dir, Path):
        return config_dir
    raise RuntimeError(
        "Packaged model configs are not available as a filesystem path; "
        "use ModelConfig.from_name() to load named configs."
    )


@dataclass(frozen=True)
class ModelConfig:
    """UniMeth model configuration (variable parameters only)."""
    
    # Model architecture (variable)
    d_model: int = 384
    d_ff: int = 2048
    d_kv: int = 64
    num_heads: int = 8
    num_layers: int = 12
    dropout_rate: float = 0.1
    
    # Position encoding (variable)
    max_position_embeddings: int = 4096
    
    # CNN Processor (variable)
    num_cnn_layers: int = 2
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])
    strides: List[int] = field(default_factory=lambda: [2, 2])
    
    def __post_init__(self):
        """Validate config after initialization."""
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert len(self.kernel_sizes) == self.num_cnn_layers, \
            "kernel_sizes length must match num_cnn_layers"
        assert len(self.strides) == self.num_cnn_layers, \
            "strides length must match num_cnn_layers"
    
    @property
    def total_stride(self) -> int:
        """Calculate total CNN stride."""
        import math
        return math.prod(self.strides)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary for model initialization."""
        result = asdict(self)
        # Add fixed parameters
        result.update({
            'pad_token_id': TOKENIZER['[PAD]'],
            'eos_token_id': TOKENIZER['[END]'],
            'decoder_start_token_id': TOKENIZER['[START]'],
            'vocab_size': len(VOCAB),
            'is_encoder_decoder': True,
            'feed_forward_proj': 'gated-gelu',
            'methylation_tokens': list(METHYLATION_TOKENS.values()),
            'methylation_labels': list(METHYLATION_LABELS.values()),
            # BART layer counts
            'encoder_layers': self.num_layers,
            'decoder_layers': self.num_layers,
        })
        return result
    
    def to_json(self, path: str):
        """Save variable config to JSON file."""
        data = asdict(self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_name(cls, name: str = "default"):
        """Load config from packaged configs/{name}.json."""
        config_path = resources.files("unimeth.configs").joinpath(f"{name}.json")
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass(frozen=True)
class DataConfig:
    """Data loading and training configuration (runtime parameters).
    
    This contains parameters for data loading, batching, and training behavior,
    as opposed to ModelConfig which contains model architecture parameters.
    """
    # Data loading
    num_workers: int = 8
    batch_size: int = 512
    
    # Binning for efficient batching
    num_bins: int = 15
    bin_size: int = 500
    max_bin_length: int = 512
    
    # Sequence patching/chunking
    chunk_size: int = 256
    overlap: int = 16
    
    # Methylation detection flags (0=off, 1=on)
    cpg: int = 0
    chg: int = 0
    chh: int = 0
    m6A: int = 0
    
    # Platform parameters
    pore_type: str = 'R9.4.1'
    frequency: str = '4khz'
    dorado_version: float = 0.71
    
    # Filtering thresholds
    mapq_thres: int = 0
    negative_thres: int = 0
    positive_thres: int = 100
    
    # Vocabulary reference (fixed constant, not configurable)
    @property
    def vocab(self) -> List[str]:
        return VOCAB
