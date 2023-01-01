import argparse
import os
import sys
from pathlib import Path

import torch
import random
import numpy as np

from src.train_files.serial_forecasting.train_serial import train




def parse():
    random.seed(fix_seed := 2023)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # basic config
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model_arch_desc', type=str, required=True, default='basic',
                        help='brief decription of model architecture')

    # data loader
    parser.add_argument('--data_path', type=str, default='data/forex/nasdaq.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S]; '
                             'M:multivariate predict multivariate, '
                             'S:univariate predict univariate')
    parser.add_argument('--target', type=str, default='value', help='target feature in S or MS task')
    parser.add_argument('--scale', type=bool, default=False, help='whether to scale the data')
    parser.add_argument('--cols', type=any, default=None, help='cols to use, default: None')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle the data')
    parser.add_argument('--log_every_n_steps', type=int, default=300, help='log every n step')
    parser.add_argument('--accelerator', type=str, default='cpu', help=' "mps" "cpu", "gpu", "tpu", "ipu", "auto" ')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96*4, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # optimization
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='optimizer weight decay')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--auto_lr_find', type=bool, default=True, help='whether to use auto lr finder of lighrning')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    os.chdir("../../")
    print(os.getcwd())

    args = parse()
    args.exp_dir = f'src/lightning_logs/{args.model_id}/{args.model_arch_desc}'
    if not os.path.exists(exp_path := Path(args.exp_dir + '/lightning_logs')):
        os.makedirs(exp_path)
    train(args)
    #
    # if args.is_training:
    #     for ii in range(args.itr):
    #         # setting record of experiments
    #         setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
    #             args.model_id,
    #             args.model,
    #             args.data,
    #             args.features,
    #             args.seq_len,
    #             args.label_len,
    #             args.pred_len,
    #             args.d_model,
    #             args.n_heads,
    #             args.e_layers,
    #             args.d_layers,
    #             args.d_ff,
    #             args.factor,
    #             args.embed,
    #             args.distil,
    #             args.des, ii)
    #
    #         exp = Exp(args)  # set experiments
    #         print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    #         exp.train(setting)
    #
    #         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #         exp.test(setting)
    #
    #         if args.do_predict:
    #             print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #             exp.predict(setting, True)
    #
    #         torch.cuda.empty_cache()
    # else:
    #     ii = 0
    #     setting = f'{args.model_id}_{args.model}_{args.data}' \
    #               f'ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
    #               f'dm{args.d_model}_nh{ args.n_heads}_el{args.e_layers}_dl{args.d_layers,}' \
    #               f'df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}'
    #
    #     exp = train(args)  # set experiments
    #     print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.test(setting, test=1)
    #     torch.cuda.empty_cache()