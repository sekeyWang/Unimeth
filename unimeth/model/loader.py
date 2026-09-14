"""
Model loading utilities for UniMeth.
"""
import torch

from unimeth.model.unimeth import UniMeth
from unimeth.config.model_config import ModelConfig


def load_model(
    config=None,
    model_path=None,
    mode='inference',
    device=None,
    attention_backend=None,
    **model_kwargs
):
    """
    Load and prepare UniMeth model.

    Args:
        config: ModelConfig object, config name (str), or path to JSON file.
                If None, uses packaged default config.
        model_path: Path to checkpoint file (optional)
        mode: Model mode ('inference', 'finetune', 'pretrain', etc.)
        device: Target device (optional)
        attention_backend: Optional attention implementation selected before
                           constructing the underlying BART model.
        **model_kwargs: Additional arguments passed to model constructor (e.g., plant=True)
    
    Returns:
        Prepared model
    
    Examples:
        # Use packaged default config
        model = load_model()
        
        # Use packaged config by name
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
    model = UniMeth(
        mode=mode,
        config=config.to_dict(),
        attention_backend=attention_backend,
        **model_kwargs,
    )
    
    # Load weights if provided
    if model_path is not None:
        model = _load_checkpoint(model, model_path)
    
    # Move to device
    if device is not None:
        model = model.to(device)
    
    return model

def _load_checkpoint(model, checkpoint_path):
    """Load checkpoint weights into model."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    return model
