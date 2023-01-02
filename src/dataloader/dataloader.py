"""dataloader.py"""

import icecream as ic
from src.dataloader.dataset import TimeSeries, TimeSeriesPred
from torch.utils.data import DataLoader


def data_provider(args, flag):
    """Create Dataloader"""
    if flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = TimeSeriesPred
    elif flag in ['train', 'val']:
        shuffle_flag = args.shuffle
        drop_last = True
        batch_size = args.batch_size
        Data = TimeSeries
    else:  # flag == 'test'
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        Data = TimeSeries

    data_set = Data(
        data_path=args.data_path,
        flag=flag,
        features=args.features,
        target=args.target,
        scale=args.scale,
        size=[args.seq_len, args.label_len, args.pred_len],
        cols=args.cols
    )

    ic.ic(flag, len(data_set))

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_loader
