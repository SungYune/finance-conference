#%%

from icecream import ic
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.dataloader.dataloader import data_provider
from src.model.NLinear import NLinear
from src.utils.logger import logger_provider

class LitNLinear(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

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
        test_loss = F.l1_loss(output, y)
        self.log("test_loss", test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer


def train(args):
    train_loader = data_provider(args, flag='train')
    val_loader   = data_provider(args, flag='val')
    test_loader  = data_provider(args, flag='test')

    model = LitNLinear(args, NLinear(args))
    logger = logger_provider(args)

    # train model
    trainer = pl.Trainer(
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.epoch,
        accelerator=args.accelerator,
        auto_lr_find=args.auto_lr_find,
        callbacks=[EarlyStopping(monitor="val_loss")],
        default_root_dir=args.exp_dir,
    )

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                )

