from torch import nn


class ARIMABlock(nn.Module):
    def __init__(self, args):
        super(ARIMABlock, self).__init__()
        self.p = args.p
        self.q = args.q
        self.ar = nn.Linear(self.p, 1)
        self.ma = nn.Linear(self.q, 1)

    def forward(self, x):
        ar_output = self.ar(x[:, :self.p])
        ma_output = self.ma(x[:, self.p:])
        return ar_output + ma_output


class ARIMA(nn.Module):
    def __init__(self, args):
        super(ARIMA, self).__init__()
        self.model = nn.Sequential(
            ARIMABlock(args),
            nn.Linear(1, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x
