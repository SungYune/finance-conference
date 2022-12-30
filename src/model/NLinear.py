from typing import Any
from icecream import ic

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len, pred_len, channels):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Conv = nn.Conv2d(in_channels=1,
                              out_channels=self.pred_len,
                              kernel_size=(self.seq_len, self.channels),
                              bias=False
                              )

    def forward(self, input_conv):
        # x: [Batch, Input length, Channel]
        input_linear = input_conv[:, :, 0].detach()  # input_linear.shape: torch.Size([1, 730])
        seq_last = input_linear[:, -1].unsqueeze(1).detach()  # seq_last.shape: torch.Size([1, 1])
        input_linear = input_linear - seq_last
        output_linear = self.Linear(input_linear.unsqueeze(2).permute(0, 2, 1)).permute(0, 2, 1).squeeze(2)
        output_linear = output_linear + seq_last  # [Batch, Output length]

        output_conv = self.Conv(input_conv.unsqueeze(1)).view(-1, self.pred_len)
        output = output_linear + output_conv
        return output  # [Batch, Output length, Channel]


class LitNLinear(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        X, y = batch['X'], batch['y']
        output = self.model(X)
        loss = F.l1_loss(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, y = batch['X'], batch['y']
        output = self.model(X)
        val_loss = F.l1_loss(output, y)
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        X, y = batch['X'], batch['y']
        output = self.model(X)
        test_loss = F.l1_loss(output, y)
        self.log("test_loss", test_loss)
        return test_loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        X, y = batch['X'], batch['y']
        output = self.model(X)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        return optimizer

