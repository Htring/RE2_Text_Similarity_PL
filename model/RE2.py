#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: RE2.py
@time:2022/05/04
@description:
"""
import torch
from torch import nn
from modules.encoder import Encoder
from modules.alignment import registry as alignment
from modules.fusion import registry as fusion
from modules.connection import registry as connection
from modules.prediction import registry as prediction
from modules.pooling import Pooling
from modules.embedding import Embedding
from argparse import ArgumentParser, Namespace
import torch.nn.functional as F


class RE2(nn.Module):

    @staticmethod
    def add_argparse_args() -> ArgumentParser:
        parser = ArgumentParser(description='re2', add_help=False)
        parser.add_argument('-dropout', type=float, default=0.2, help='dropout层参数， default 0.2')
        parser.add_argument('-hidden_size', type=int, default=150, help='编码层中输出通道数， default 150')
        parser.add_argument('-kernel_sizes', type=list, default=[3], help='编码层卷积核大小， default 3')
        parser.add_argument('-blocks', type=int, default=2, help='block数量， default 3')
        parser.add_argument('-enc_layers', type=int, default=2, help='编码层cnn层数， default 2')
        parser.add_argument('-alignment', type=str, default="linear", help='对齐方式， default linear')
        parser.add_argument('-connection', type=str, default="aug", help='连接方式， default aug')
        parser.add_argument('-fusion', type=str, default="full", help='融合方式， default full')
        parser.add_argument('-fix_embeddings', type=bool, default=True, help='是否固定embedding， default True')
        parser.add_argument('-prediction', type=str, default="full", help='预测方式， default full')
        return parser

    def __init__(self, args: Namespace):
        super().__init__()
        self.dropout = args.dropout
        self.embedding = Embedding(args)
        input_emb_size = args.embedding_dim if args.connection == 'aug' else 0
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "encoder":Encoder(args, args.embedding_dim if i == 0 else input_emb_size + args.hidden_size),
                "alignment": alignment[args.alignment](args,
                                                       args.embedding_dim + args.hidden_size if i == 0 else input_emb_size +args.hidden_size * 2),
                "fusion": fusion[args.fusion](args,
                                              args.embedding_dim + args.hidden_size if i == 0 else input_emb_size + args.hidden_size * 2)
            }) for i in range(args.blocks)
        ])
        self.connection = connection[args.connection](args)
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

    def forward(self, a:torch.Tensor, b: torch.Tensor):
        mask_a = torch.ne(a, 0).unsqueeze(2)  # batch_size, seq_len, 1
        mask_b = torch.ne(b, 0).unsqueeze(2)
        a = self.embedding(a)
        b = self.embedding(b)
        res_a, res_b = a, b
        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)    # NxLx(hidden_size + embedding_dim)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)  # NxLx(hidden_size)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        logits = self.prediction(a, b)
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities
