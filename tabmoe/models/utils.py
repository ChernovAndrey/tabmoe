from torch import Tensor, nn


def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d ** -0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)
