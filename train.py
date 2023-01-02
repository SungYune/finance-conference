import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.train_files.serial_forecasting.lightning_train import train


def parse():
    random.seed(fix_seed := 2023)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # basic config
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model_arch_desc', type=str, required=True, default='basic',
                        help='brief description of model architecture')
    parser.add_argument('--run_tensorboard', type=bool, default=False, help='rather to use tensorboard')

    # data loader
    parser.add_argument('--data_path', type=str, default='data/forex/nasdaq.csv', help='data file')
    parser.add_argument('--accelerator', type=str, default='cpu', help=' "mps" "cpu", "gpu", "tpu", "ipu", "auto" ')
    parser.add_argument('--vis_col_row', type=int, default=3, help='visualize row col')

    parser.add_argument('--target', type=str, default='value', help='target feature in S or MS task')
    parser.add_argument('--scale', type=bool, default=False, help='whether to scale the data')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle the data')
    parser.add_argument('--log_every_n_steps', type=int, default=300, help='log every n step')
    parser.add_argument('--cols', type=any, default=None, help='cols to use, default: None')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S]; '
                             'M:multivariate predict multivariate, '
                             'S:univariate predict univariate')

    # model
    parser.add_argument('--model_type', type=str, default="nlinear", help='"nlinear" "arima" "lstm"')

    # forecasting task ARIMA
    parser.add_argument('--p', type=int, default=2, help='p in ARIMA')
    parser.add_argument('--q', type=int, default=1, help='q in ARIMA')

    # forecasting task NLinear
    parser.add_argument('--seq_len', type=int, default=96 * 4, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # forecasting task LSTM
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='num layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

    # optimization
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='optimizer weight decay')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--auto_lr_find', type=bool, default=True, help='whether to use auto lr finder of lighrning')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse()
    args.exp_dir = f'src/lightning_logs/{args.model_id}'
    if not os.path.exists(Path(args.exp_dir)):
        os.makedirs(Path(args.exp_dir))
    train(args)

    if args.run_tensorboard:
        os.system(f"tensorboard --logdir=src/lightning_logs/{args.model_id}/{args.model_arch_desc}")
