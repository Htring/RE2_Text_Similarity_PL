#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: encoder.py
@time:2022/05/04
@description:
"""
import torch
from torch import nn
import torch.nn.functional as F
from . import Conv1d


class Encoder(nn.Module):

    __doc__ = """ 编码器 """

    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.encoders = nn.ModuleList(
            [
                Conv1d(in_channels=input_size if i == 0 else args.hidden_size,
                       out_channels=args.hidden_size,
                       kernel_sizes=args.kernel_sizes) for i in range(args.enc_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x.transpose(1, 2)  # BxCxL
        mask = mask.transpose(1, 2)
        for i, encoder in enumerate(self.encoders):
            x.masked_fill_(~mask, 0.)
            if i > 0:
                x = F.dropout(x, self.dropout, self.training)
            x = encoder(x)
        x = F.dropout(x, self.dropout, self.training)
        return x.transpose(1, 2)  # BxLxC
