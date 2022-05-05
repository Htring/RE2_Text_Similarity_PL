#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: RE2_PL.py
@time:2022/05/01
@description:
"""
from typing import Any, Optional

import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from sklearn.metrics import classification_report
from argparse import Namespace
import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn import functional as F
from model.RE2 import RE2


class RE2_PL(pl.LightningModule):

    def __init__(self, param: Namespace):
        super().__init__()
        print(param)
        self.re2 = RE2(param)
        self.lr = param.lr
        self.weight_decay = param.weight_decay

    def forward(self, premises, hypotheses) -> Any:
        return self.re2(premises, hypotheses)

    def training_step(self, batch) -> STEP_OUTPUT:
        x1, x2, y = batch
        out = self.forward(x1, x2)
        loss = F.cross_entropy(out[0], y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x1, x2, y = batch
        pred = self.forward(x1, x2)
        loss = F.cross_entropy(pred[0], y)
        pred_index = pred[0].argmax(dim=-1)
        return {"true": y, "pred": pred_index, "loss": loss}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._test_unit(outputs)

    def _test_unit(self, outputs):
        print()
        pred_lst, true_lst = [], []
        loss_sum = 0
        for batch_result in outputs:
            pred_lst.extend(batch_result["pred"].cpu().tolist())
            true_lst.extend(batch_result['true'].cpu().tolist())
            loss_sum += batch_result['loss'].item()
        report = classification_report(true_lst, pred_lst)
        f1_score = torchmetrics.functional.f1(torch.tensor(pred_lst), torch.tensor(true_lst), average="micro")
        accuracy = torchmetrics.functional.accuracy(torch.tensor(pred_lst), torch.tensor(true_lst), average="micro")
        recall = torchmetrics.functional.recall(torch.tensor(pred_lst), torch.tensor(true_lst), average="micro")
        self.log("val_loss", loss_sum / len(outputs))
        self.log("f1_score", f1_score)
        self.log("recall", recall)
        self.log("accuracy", accuracy)
        print(report)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x1, x2, y = batch
        pred = self.forward(x1, x2)
        loss = F.cross_entropy(pred[0], y)
        pred_index = pred[0].argmax(dim=-1)
        return {"true": y, "pred": pred_index, "loss": loss}

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._test_unit(outputs)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

