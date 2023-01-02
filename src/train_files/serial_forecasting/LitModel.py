import pytorch_lightning as pl
import torch.nn.functional as F
from icecream import ic
from torch import optim


class LitModel(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.scale = args.scale

    def training_step(self, batch, batch_idx):
        X, y = batch['X'], batch['y']
        output = self.model(X)
        output = output[:, -self.args.pred_len:, :]
        y = y[:, -self.args.pred_len:, :]
        loss = F.l1_loss(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, y = batch['X'], batch['y']
        output = self.model(X)
        output = output[:, -self.args.pred_len:, :]
        y = y[:, -self.args.pred_len:, :]
        val_loss = F.l1_loss(output, y)
        # TODO : add more metrics and log
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_basic.html
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        X, y = batch['X'], batch['y']
        output = self.model(X)
        output = output[:, -self.args.pred_len:, :]
        y = y[:, -self.args.pred_len:, :]

        test_loss = F.l1_loss(output, y)
        self.log("test_loss", test_loss)
        return test_loss

    def predict_step(self, batch, batch_idx):
        X, y, X_mark, y_mark = batch['X'], batch['y'], batch['X_mark'], batch['y_mark']
        output = self.model(X)
        # stack the output and y detach
        return output.cpu().numpy(), y.cpu().numpy(), y_mark.cpu().numpy()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer
