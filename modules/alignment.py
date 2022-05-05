#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: alignment.py
@time:2022/05/04
@description:
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from .utils import register
from . import Linear

registry = {}
# 将register中registry参数值固定为registry
register = partial(register, registry=registry)


@register("identity")
class Alignment(nn.Module):

    def __init__(self, args, _):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(args.hidden_size)))

    def _attention(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, a: torch.Tensor, b: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor):
        attention = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float())
        mask = mask.bool()
        attention.masked_fill_(~mask, -1e4)
        attention_a = F.softmax(attention, dim=1)
        attention_b = F.softmax(attention, dim=2)
        feature_a = torch.matmul(attention_b, b)
        feature_b = torch.matmul(attention_a, a)
        return feature_a, feature_b


@register("linear")
class MappedAlignment(Alignment):

    def __init__(self, args, input_size):
        super().__init__(args, input_size)
        self.projection = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(input_size, args.hidden_size, activations=True)
        )

    def _attention(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)
