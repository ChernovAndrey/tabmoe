import torch
from torch import nn
from torch import Tensor
from typing import Literal
import rtdl_num_embeddings

# from tabmoe.modeling.embeddings import PiecewiseLinearEmbeddings  # TODO: could be a local import?
from .embeddings import PiecewiseLinearEmbeddings
from tabmoe.preprocessing.dataset import Dataset
from tabmoe.enums.utils import validate_enum
from tabmoe.enums.model import GatingType, EmbeddingPolicy
from .backbones.moe import MoE
from .backbones.mlp import MLP
from typing import Type


class Model(nn.Module):
    """MLP & MoE."""

    def __init__(
            self,
            *,
            dataset: Dataset,

            backbone: dict,
            num_embeddings: None | dict = None,  # Embedding type
            gating_type: Literal['standard', 'gumbel'] = 'gumbel'
    ) -> None:
        """

        :param dataset:
        :param backbone: must have a key 'type' + model parameters
        :param num_embeddings: if not None, must have the following keys: 'type', 'n_bins', 'd_embedding'
        :param embedding_policy:
        :param gating_type:
        """

        assert backbone.get('type', None), "Backbone dictionary must have a key 'type'"

        super().__init__()
        self.dataset = dataset
        if num_embeddings is not None:
            self.num_embedding_policy = validate_enum(EmbeddingPolicy, num_embeddings.get('type', None))
        else:
            self.num_embedding_policy = None

        self.gating_type = validate_enum(GatingType, gating_type)

        n_num_features = self.dataset.n_num_features
        if n_num_features == 0:
            self.num_module = None
            self.d_num = 0
        elif num_embeddings is None:
            self.num_module = None
            self.d_num = n_num_features
        else:
            if self.num_embedding_policy == EmbeddingPolicy.PIECEWISE_LINEAR_EMBEDDINGS:
                self.n_bins = num_embeddings.get('n_bins', None)
                if self.n_bins is None:
                    raise ValueError(
                        "Number of bins (`n_bins`) must be specified in num_embeddings when using PiecewiseLinearEmbeddings.")
                self.bin_edges = rtdl_num_embeddings.compute_bins(
                    torch.tensor(self.dataset.X_train_num, dtype=torch.float32), self.n_bins)

                self.num_module = PiecewiseLinearEmbeddings(d_embedding=num_embeddings['d_embedding'], bins=self.bin_edges)
                self.d_num = n_num_features * num_embeddings['d_embedding']
            else:
                assert False, "num_embeddings must be None or contains a valid 'type'"

        self.d_cat = sum(self.dataset.cat_cardinalities)
        self.d_bin = self.dataset.n_bin_features
        self.d_total = self.d_num + self.d_cat + self.d_bin
        self.n_classes = self.dataset.n_classes

        assert self.d_total > 0, 'All d_num, d_cat and d_bin are zero, at least one should be postive'

        d_out = 1 if self.n_classes is None else self.n_classes

        self.backbone = MoE(d_in=self.d_total, **backbone, d_out=d_out) #TODO: fix it,

    def forward(
            self, x_num: None | Tensor = None, x_cat: None | Tensor = None, x_bin: None | Tensor = None,
            num_samples: None | int = None, return_average: None | bool = None,
    ) -> Tensor: #TODO: pass the whole tensor as one?
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is not None:
            x.append(x_cat)  # TODO: convert to float here: x.append(x_cat.float()) or in encoder?
        if x_bin is not None:
            x.append(x_bin)

        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])

        if (return_average is not None) and (num_samples is not None):
            x = self.backbone(x, num_samples=num_samples, return_average=return_average)
        else:
            x = self.backbone(x)

        return x
