import torch
from torch import nn
from torch import Tensor
from typing import Literal
import rtdl_num_embeddings

from tabmoe.preprocessing.dataset import Dataset
from tabmoe.enums.utils import validate_enum
from tabmoe.enums.model import EmbeddingPolicy
from tabmoe.utils.model import is_dataparallel_available

from .backbones import get_model_instance
from .embeddings import PiecewiseLinearEmbeddings
from tabmoe.utils.hyperparam_logger import HyperparamLogger


class Model(nn.Module):
    """MLP & MoE."""

    def __init__(
            self,
            *,
            dataset: Dataset,

            backbone_parameters: dict,
            num_embeddings: None | dict = None,  # Embedding type
            amp: bool = False,
            input_logger: None | HyperparamLogger = None
    ) -> None:
        """

        :param dataset:
        :param backbone: must have a key 'type' + model parameters
        :param num_embeddings: if not None, must have the following keys: 'type', 'n_bins', 'd_embedding'
        :param embedding_policy:
        :param gating_type:
        """

        assert backbone_parameters.get('type', None), "backbone_parameters dictionary must have a 'type' key"

        super().__init__()
        if input_logger is not None:
            input_logger.log('model', backbone=backbone_parameters, num_embeddings=num_embeddings, amp=amp)
        self.dataset = dataset

        # Use it with caution; there is no need to use AMP on small datasets.
        self.amp_dtype = (
            torch.bfloat16
            if amp
               and torch.cuda.is_available()
               and torch.cuda.is_bf16_supported()
            else None
        )
        self.amp_enabled = self.amp_dtype is not None

        # For FP16, the gradient scaler must be used.
        print(f'AMP enabled: {self.amp_enabled}')

        if num_embeddings is not None:
            self.num_embedding_policy = validate_enum(EmbeddingPolicy, num_embeddings.get('type', None))
        else:
            self.num_embedding_policy = None

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
                    torch.as_tensor(self.dataset.X_train_num, dtype=torch.float32), self.n_bins)

                self.num_module = PiecewiseLinearEmbeddings(d_embedding=num_embeddings['d_embedding'],
                                                            bins=self.bin_edges)
                self.d_num = n_num_features * num_embeddings['d_embedding']
            else:
                assert False, "num_embeddings must be None or contains a valid 'type'"

        self.d_cat = self.dataset.n_encoded_cat_features
        self.d_bin = self.dataset.n_bin_features
        self.d_in = self.d_num + self.d_cat + self.d_bin
        self.d_out = 1 if self.dataset.is_regression or self.dataset.is_binary else self.n_classes

        print(f'input dimension to a network: {self.d_in}')
        print(f'output dimension: {self.d_out}')
        print(f'numeric dimension: {self.d_num}')
        print(f'categorical dimension: {self.d_cat}')
        print(f'binary dimension: {self.d_bin}')

        assert self.d_in > 0, 'All d_num, d_cat and d_bin are zero, at least one should be positive'

        self.backbone = get_model_instance(**backbone_parameters, d_in=self.d_in, d_out=self.d_out)

        if is_dataparallel_available():
            self.to(self.dataset.device)  # TODO: it was never tested, but it should work :)
        else:
            self.to(self.dataset.device)

    def run(self, x: torch.Tensor = None, num_samples: None | int = None, return_average: bool = True, ) -> Tensor:
        with torch.autocast(str(self.dataset.device), enabled=self.amp_enabled, dtype=self.amp_dtype):
            return self(x, num_samples, return_average) \
                .squeeze(-1).float()  # Remove the last dimension for regression predictions.

    # def forward(
    #         self, x_num: None | Tensor = None, x_cat: None | Tensor = None, x_bin: None | Tensor = None,
    #         num_samples: None | int = None, return_average: bool = True,
    # ) -> Tensor:  # TODO: pass the whole tensor as one or leave as it is?
    #     x = []
    #     if x_num is not None:
    #         x.append(x_num if self.num_module is None else self.num_module(x_num))
    #     if x_cat is not None:
    #         x.append(x_cat)
    #     if x_bin is not None:
    #         x.append(x_bin)
    #
    #     x = torch.column_stack([x_.flatten(1, -1) for x_ in x])
    #     if (return_average is not None) and (num_samples is not None):
    #         x = self.backbone(x, num_samples=num_samples, return_average=return_average)
    #     else:
    #         x = self.backbone(x)
    #
    #     return x

    def forward(
            self, x: Tensor, num_samples: None | int = None, return_average: bool = True
    ) -> Tensor:
        """
        Splits concatenated input into numerical and other features.
        - Processes numerical features via `num_module` if applicable.
        - Concatenates all features again before passing to `backbone`.
        """
        n_num_features = self.dataset.n_num_features
        # Extract numerical features (first `n_numerical_features` columns)
        x_num = x[:, :n_num_features] if n_num_features > 0 else None
        x_other = x[:, n_num_features:] if x.shape[1] > n_num_features else None

        # Apply num_module if available
        if x_num is not None and self.num_module is not None:
            x_num = self.num_module(x_num).flatten(1, -1)

        # Concatenate processed numerical features with others
        x_processed = torch.cat([x_num, x_other], dim=1) if x_other is not None else x_num
        # Pass to backbone
        if num_samples is not None and return_average is not None:
            x_output = self.backbone(x_processed, num_samples=num_samples, return_average=return_average)
        else:
            x_output = self.backbone(x_processed)

        return x_output
