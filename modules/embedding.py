#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: embedding.py
@time:2022/05/04
@description:
"""
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


class Embedding(nn.Module):

    __doc__ = """ 改写的embedding """

    def __init__(self, args):
        super().__init__()
        self.fix_embeddings = args.fix_embeddings
        self.embedding = nn.Embedding(args.num_vocab, args.embedding_dim, padding_idx=0)
        self.dropout = args.dropout

    def set_(self, value):
        self.embedding.weight.requires_grad = not self.fix_embeddings
        self.embedding.load_state_dict(OrderedDict({'weight': torch.tensor(value)}))

    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, self.dropout, self.training)
        return x
