""" NLinear Model """
import torch.nn as nn


class NLinear(nn.Module):
    def __init__(self, args):
        super(NLinear, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
