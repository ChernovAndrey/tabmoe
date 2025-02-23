import torch
from torch import nn
from torch import Tensor

from tabmoe.models.embeddings import PiecewiseLinearEmbeddings  # TODO: could be a local import?
from tabmoe.models.moe import MoE  # TODO: could be a local import?


class Model(nn.Module):
    """MLP & MoE."""

    def __init__(
            self,
            *,
            n_num_features: int,
            cat_cardinalities: list[int],
            n_classes: None | int,
            backbone: dict,
            bins: None | list[Tensor],  # For piecewise-linear encoding/embeddings.
            num_embeddings: None | dict = None,  # # # Embedding type
    ) -> None:

        # >>> Validate arguments.
        assert n_num_features >= 0
        # assert n_num_features or cat_cardinalities # TODO: what if only binary features
        super().__init__()

        # >>> numeric features

        if n_num_features == 0:
            assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            assert bins is None
            self.num_module = None
            d_num = n_num_features

        else:
            self.num_module = PiecewiseLinearEmbeddings(**num_embeddings, bins=bins)
            d_num = n_num_features * num_embeddings['d_embedding']

        # >>> Categorical features

        d_cat = sum(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat  # TODO: add binaries features
        d_out = 1 if n_classes is None else n_classes

        self.backbone = MoE(d_in=d_flat, **backbone, d_out=d_out)

    def forward(
            self, x_num: None | Tensor = None, x_cat: None | Tensor = None,
            num_samples: None | int = None, return_average: None | bool = None,
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is not None:
            x.append(x_cat)  # TODO: convert to float here: x.append(x_cat.float()) or in encoder?

        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])

        if (return_average is not None) and (num_samples is not None):
            x = self.backbone(x, num_samples=num_samples, return_average=return_average)
        else:
            x = self.backbone(x)

        return x
