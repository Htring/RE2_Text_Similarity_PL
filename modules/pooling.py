#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: pooling.py
@time:2022/05/04
@description:
"""
from torch import nn
import torch

class Pooling(nn.Module):

    def forward(self, x:torch.Tensor, mask: torch.Tensor):
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]
