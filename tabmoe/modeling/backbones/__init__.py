from typing import Literal
from .mlp import MLP
from .moe import MoE

MODEL_REGISTRY = {
    "mlp": MLP,
    "moe": MoE,
}


# for case-insensitive import
def get_model_class(model_type: Literal['mlp', 'moe']):
    return MODEL_REGISTRY.get(model_type.lower(), None)
