from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.nn.modules.rnn import LSTM
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, predict_length, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, predict_length)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

class LitLSTM(pl.LightningModule):
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