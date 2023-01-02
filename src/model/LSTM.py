from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from icecream import ic

from torch.nn.modules.rnn import LSTM
from torch import optim
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size

        self.dropout = args.dropout
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.lstm = nn.LSTM(
            input_size=self.seq_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bias=True,
        )
        self.linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.pred_len
        )

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        x = x.reshape(x.size(0), 1, -1)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out.unsqueeze(2)  # [Batch, Output length, Channel]
