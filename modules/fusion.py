#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: fusion.py
@time:2022/05/04
@description:
"""
import torch
from torch import nn
from functools import partial
from .utils import register
from . import Linear
import torch.nn.functional as F

registry = {}
register = partial(register, registry=registry)


@register('simple')
class Fusion(nn.Module):

    def __init__(self, args, input_size):
        super().__init__()
        self.fusion = Linear(input_size * 2, args.hidden_size, activations=True)

    def forward(self, x, align):
        return self.fusion(torch.cat([x, align], dim=-1))


@register('full')
class FulFusion(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.fusion1 = Linear(input_size*2, args.hidden_size, activations=True)
        self.fusion2 = Linear(input_size*2, args.hidden_size, activations=True)
        self.fusion3 = Linear(input_size*2, args.hidden_size, activations=True)
        self.fusion = Linear(args.hidden_size * 3, args.hidden_size, activations=True)

    def forward(self, x: torch.Tensor, align: torch.Tensor):
        g1 = self.fusion1(torch.cat([x, align], dim=-1))
        g2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        g3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        g = F.dropout(torch.cat([g1, g2, g3], dim=-1), self.dropout, self.training)
        return self.fusion(g)
