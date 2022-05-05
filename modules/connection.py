#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: connection.py
@time:2022/05/04
@description:
"""
import math
import torch
from torch import nn
from functools import partial
from .utils import register
from . import Linear

registry = {}
register = partial(register, registry=registry)


@register('none')
class NullConnection(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, _, __):
        return x


@register("residual")
class Residual(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.embedding_dim, args.hidden_size)

    def forward(self, x: torch.Tensor, res: torch.Tensor, index: int):
        if index == 1:
            res = self.linear(res)
        return (x + res) * math.sqrt(0.5)


@register('aug')
class AugmentedResidual(nn.Module):

    def __init__(self, _):
        super().__init__()

    def forward(self, x: torch.Tensor, res: torch.Tensor, index: int):
        if index == 1:
            return torch.cat([x, res], dim=-1)  # res is embedding
        hidden_size = x.size(-1)
        x = (res[:, :, : hidden_size] + x) * math.sqrt(0.5)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)  # latter half of res is embedding
