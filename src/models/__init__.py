"""
Model registry for easy instantiation and switching.
"""

from typing import Dict, Type
from .lstm_model import LSTMModel
from .hybrid_models import CNNLSTMHybridV1, CNNLSTMHybridV2, CNNLSTMHybridV3
from .transformer_model import TransformerModel
from .multihead_model import CNNLSTMMultiHead


MODEL_REGISTRY: Dict[str, Type] = {
    "lstm": LSTMModel,
    "hybrid_v1": CNNLSTMHybridV1,
    "hybrid_v2": CNNLSTMHybridV2,
    "hybrid_v3": CNNLSTMHybridV3,
    "multihead": CNNLSTMMultiHead,
    "transformer": TransformerModel,
}


def get_model(model_name: str, **kwargs):
    """
    Get model by name from registry.
    
    Args:
        model_name: Name of model to instantiate
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model name not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def list_available_models():
    """Return list of available models."""
    return list(MODEL_REGISTRY.keys())
