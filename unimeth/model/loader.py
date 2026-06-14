"""
Model loading utilities for UniMeth.
"""
import torch
import warnings

from unimeth.model.unimeth import UniMeth
from unimeth.config.model_config import ModelConfig


def load_model(
    config=None,
    model_path=None,
    mode='inference',
    device=None,
    **model_kwargs
):
    """
    Load and prepare UniMeth model.

    Args:
        config: ModelConfig object, config name (str), or path to JSON file.
                If None, uses default config from configs/default.json.
        model_path: Path to checkpoint file (optional)
        mode: Model mode ('inference', 'finetune', 'pretrain', etc.)
        device: Target device (optional)
        **model_kwargs: Additional arguments passed to model constructor (e.g., plant=True)
    
    Returns:
        Prepared model
    
    Examples:
        # Use default config (configs/default.json)
        model = load_model()
        
        # Use config by name (configs/distilled.json)
        model = load_model(config="distilled")
        
        # Use config from specific JSON file
        model = load_model(config="path/to/my_config.json")
        
        # Use ModelConfig object directly
        model = load_model(config=ModelConfig.from_name("distilled"))
        
        # Load with checkpoint
        model = load_model(config="default", model_path="checkpoint.bin")
    """
    # Resolve config
    if config is None:
        config = ModelConfig.from_name("default")
    elif isinstance(config, str):
        if config.endswith('.json'):
            # Path to JSON file
            config = ModelConfig.from_json(config)
        else:
            # Config name (e.g., "default", "distilled")
            config = ModelConfig.from_name(config)
    elif not isinstance(config, ModelConfig):
        raise TypeError(f"config must be ModelConfig, str, or None, got {type(config)}")
    
    # Create model
    model = UniMeth(mode=mode, config=config.to_dict(), **model_kwargs)
    
    # Load weights if provided
    if model_path is not None:
        model = _load_checkpoint(model, model_path)
    
    # Enable SDPA (built into torch>=2.0, works on all hardware)
    model = _enable_sdpa(model)

    # Move to device
    if device is not None:
        model = model.to(device)
    
    return model

def _load_checkpoint(model, checkpoint_path):
    """Load checkpoint weights into model."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    return model


def _enable_sdpa(model):
    """Enable PyTorch SDPA attention (uses Flash Attention kernel on compatible GPUs, math fallback elsewhere).

    Uses _attn_implementation="sdpa" instead of "flash_attention_2" because HuggingFace's
    Flash Attention 2 unpadding does not correctly handle encoder padding masks when
    inputs_embeds is used, causing incorrect cross-attention in the decoder.
    """
    model.encoder_decoder.config._attn_implementation = "sdpa"
    return model
