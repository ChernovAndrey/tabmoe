import itertools
from collections.abc import Callable
from typing import Any, Literal, cast

import delu
import rtdl_num_embeddings
import torch
import torch.nn as nn
from torch.nn import Parameter


def get_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    Optimizer = getattr(torch.optim, type)
    return Optimizer(**kwargs)


# folowing two functions was taken from: https://github.com/yandex-research/tabm/blob/main/lib/deep.py
def default_zero_weight_decay_condition(
        module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        | nn.LayerNorm
        | nn.InstanceNorm1d
        | rtdl_num_embeddings.LinearEmbeddings
        | rtdl_num_embeddings.LinearReLUEmbeddings
        | _Periodic,
    )


def make_parameter_groups(
        module: nn.Module,
        zero_weight_decay_condition=default_zero_weight_decay_condition,
        custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]
