from typing import Literal, Any
from .mlp import MLP
from .moe import MoE

__all__ = ["MLP", "MoE"]

MODEL_REGISTRY = {
    "mlp": MLP,
    "moe": MoE,
}


def get_model_instance(type: Literal['mlp', 'moe'], **kwargs: Any) -> MLP | MoE:
    """
    Returns an instance of the specified model type with the given parameters.

    :param model_type: The type of model ('mlp' or 'moe').
    :param kwargs: Additional keyword arguments for the model's constructor.
    :return: An instance of the requested model.
    """
    model_class = MODEL_REGISTRY.get(type.lower())
    if model_class is None:
        raise ValueError(f"Invalid model type: {type}. Supported types: {list(MODEL_REGISTRY.keys())}")
    return model_class(**kwargs)
