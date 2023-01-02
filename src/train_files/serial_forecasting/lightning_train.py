import pytorch_lightning as pl

from src.dataloader.dataloader import data_provider
from src.model.NLinear import NLinear
from src.model.ARIMA import ARIMA
from src.model.LSTM import LSTM
from src.train_files.serial_forecasting.LitModel import LitModel

from src.utils.logger import logger_provider
from src.utils.visualizer import visualizer


def build_model(args):
    models = {
        'nlinear': NLinear,
        'arima': ARIMA,
        'lstm': LSTM
    }
    model = models[args.model_type](args)
    lightning_model = LitModel(args, model)
    return lightning_model


def train(args):
    train_loader = data_provider(args, flag='train')
    val_loader   = data_provider(args, flag='val')
    test_loader  = data_provider(args, flag='test')

    model = build_model(args)
    logger = logger_provider(args)

    # train model
    trainer = pl.Trainer(
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.epoch,
        accelerator=args.accelerator,
        auto_lr_find=args.auto_lr_find,
        default_root_dir=args.exp_dir,
        logger=logger
    )

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                )

    pred = trainer.predict(dataloaders=test_loader)
    visualizer(args, pred)
