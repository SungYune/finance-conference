import os
from pytorch_lightning.loggers import TensorBoardLogger


# default logger used by trainer
def logger_provider(args):
    logger = TensorBoardLogger(
        save_dir=args.exp_dir,
        name=args.model_arch_desc
    )
    return logger, logger.version
