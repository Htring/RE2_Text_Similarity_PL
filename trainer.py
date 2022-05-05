#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: trainer.py
@time:2022/04/19
@description:
"""
import json
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from model.RE2_PL import RE2_PL, RE2
from dataloader import NLIDataModule, char_cut

pl.seed_everything(2022)


def training(param):
    nli_dm = NLIDataModule(data_dir=param.data_dir,
                           batch_size=param.batch_size,
                           max_length=param.max_length)
    checkpoint_callback = ModelCheckpoint(monitor='f1_score',
                                          filename="re2-{epoch:03d}-{val_loss:.3f}-{f1_score:.3f}",
                                          dirpath=param.save_dir,
                                          save_top_k=3,
                                          mode="max")
    os.makedirs(param.save_dir, exist_ok=True)
    nli_dm.save_dict(param.save_dir)
    param.num_vocab = len(nli_dm.char2idx)
    param.num_classes = len(nli_dm.tag2idx)
    model = RE2_PL(param)
    if param.load_pre:
        model = model.load_from_checkpoint(param.pre_ckpt_path, param=param)
    logger = TensorBoardLogger("log_dir", name="re2")

    trainer = pl.Trainer(logger=logger, gpus=1,
                         callbacks=[checkpoint_callback],
                         max_epochs=param.epoch,
                         precision=16,
                         gradient_clip_val=0.5,
                         # limit_train_batches=0.5,
                         # limit_val_batches=0.5
                         )
    if param.train:
        trainer.fit(model=model, datamodule=nli_dm)
        nli_dm.save_dict(param.save_dir)
    if param.test:
        trainer.test(model, nli_dm)


def model_use(param):
    def _load_dict(dir_name):
        with open(os.path.join(dir_name, 'token2index.txt'), 'r', encoding='utf8') as reader:
            token2index = json.load(reader)

        with open(os.path.join(dir_name, 'index2tag.txt'), 'r', encoding='utf8') as reader:
            index2tag = json.load(reader)

        return token2index, index2tag

    def _number_data(content):
        number_data = []
        for char in char_cut(content):
            number_data.append(token2index.get(char, token2index.get("<unk>")))
        return torch.tensor([number_data], dtype=torch.long), torch.tensor([len(number_data)], dtype=torch.long)

    token2index, index2tag = _load_dict(param.save_dir)
    param.vocab_size = len(token2index)
    param.num_classes = len(index2tag)
    model = RE2_PL.load_from_checkpoint(param.pre_ckpt_path, param=param)
    test_data = {"sentence1": "杭州哪里好玩", "sentence2": "杭州哪里好玩点"}
    result_index = \
    model.forward(*_number_data(test_data["sentence1"]), *_number_data(test_data["sentence2"]))[1].argmax(dim=-1)[0].item()
    print(index2tag.get(str(result_index)))  # 1


if __name__ == '__main__':
    model_parser = RE2.add_argparse_args()
    parser = ArgumentParser(parents=[model_parser])
    parser.add_argument('-lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('-weight_decay', type=float, default=0, help='权重衰减')
    parser.add_argument('-data_dir', type=str, default="corpus/LCQMC_S", help='训练语料地址')
    parser.add_argument('-batch_size', type=int, default=300, help='批次数据大小')
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-train', type=bool, default=False)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-embedding_dim', type=int, default=200, help='词向量的维度')
    parser.add_argument('-max_length', type=int, default=50, help='取序列的最大长度')
    parser.add_argument('-save_dir', type=str, default="model_save/re2", help='模型存储位置')
    parser.add_argument('-load_pre', type=bool, default=True, help='是否加载已经训练好的ckpt')
    parser.add_argument('-pre_ckpt_path', type=str,
                        default="model_save/re2/re2-epoch=022-val_loss=0.483-f1_score=0.857.ckpt",
                        help='是否加载已经训练好的ckpt')

    args = parser.parse_args()
    training(args)
    # model_use(args)
