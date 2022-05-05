#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: prediction.py
@time:2022/05/04
@description:
"""

import torch
from torch import nn
from functools import partial
from . import Linear
from .utils import register

registry = {}
register = partial(register, registry=registry)


@register('simple')
class Prediction(nn.Module):

    def __init__(self, args, input_features=2):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(args.hidden_size * input_features, args.hidden_size, activations=True),
            nn.Dropout(args.dropout),
            Linear(args.hidden_size, args.num_classes),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return self.dense(torch.cat([a, b], dim=-1))


@register('full')
class AdvancedPrediction(Prediction):

    def __init__(self, args):
        super().__init__(args, input_features=4)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return self.dense(torch.cat([a, b, a - b, a * b], dim=-1))


@register('symmetric')
class SymmetricPrediction(AdvancedPrediction):

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return self.dense(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))
