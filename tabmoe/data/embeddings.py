# embeddings is essentially borrowed from the https://github.com/yandex-research/tabm
import rtdl_num_embeddings
# from typing import Any, Literal, cast
from typing import Literal


class PiecewiseLinearEmbeddings(rtdl_num_embeddings.PiecewiseLinearEmbeddings):
    """
    This class simply adds the default values for `activation` and `version`.
    """

    def __init__(
            self,
            *args,
            activation: bool = False,
            version: None | Literal['A', 'B'] = 'B',
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs, activation=activation, version=version)
