#%%
from icecream import ic
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from src.dataloader.dataloader import data_provider
from src.model.NLinear import NLinear, LitNLinear

def train(args):
    train_loader = data_provider(args, 'train')
    val_loader = data_provider(args, 'val')
    test_loader = data_provider(args, 'test')

    model = LitNLinear(NLinear(args))

    # train model
    trainer = pl.Trainer(
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.epoch,
        accelerator=args.accelerator,
        auto_lr_find=args.auto_lr_find,
    )

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                )
