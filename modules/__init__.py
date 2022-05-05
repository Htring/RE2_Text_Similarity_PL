#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: __init__.py.py
@time:2022/05/04
@description:
"""
from typing import Collection

import torch
import math
from torch import nn


class GeLU(nn.Module):
    __doc__ = """ gelu激活函数 """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))


class Linear(nn.Module):
    __doc__ = """ 改写的Linear层 """

    def __init__(self, in_features:int, out_features:int, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Conv1d(nn.Module):
    __doc__ = """ 改写的一维卷积 """

    def __init__(self, in_channels, out_channels, kernel_sizes: Collection[int]):
        super().__init__()
        assert all(k % 2 == 1 for k in kernel_sizes), 'only support odd kernel sizes'
        assert out_channels % len(kernel_sizes) == 0, 'out channels must be dividable by kernels'
        out_channels = out_channels // len(kernel_sizes)
        convs = []
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(in_channels,
                             out_channels,
                             kernel_size,
                             padding=(kernel_size - 1) // 2)
            nn.init.normal_(conv.weight, std=math.sqrt(2. / (in_channels * kernel_size)))
            nn.init.zeros_(conv.bias)
            convs.append(nn.Sequential(nn.utils.weight_norm(conv), GeLU()))
        self.model = nn.ModuleList(convs)

    def forward(self, x):
        return torch.cat([encoder(x) for encoder in self.model], dim=-1)