# Modified https://github.com/cure-lab/LTSF-Linear/blob/main/data_provider/data_loader.py
"""Data Loader for the dataset."""

import warnings
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


class TimeSeries(Dataset):
    def __init__(self,
                 data_path: str = '../../data/crypto/btc_data_5min.csv',
                 flag: str = 'train',
                 features: str = 'S',
                 target: str = 'Low',
                 scale: bool = True,
                 size=None,
                 cols=None
                 ):
        # size [seq_len, label_len, pred_len]
        if cols is None:
            cols = []
        if size is None:
            self.seq_len   = 28
            self.label_len = 7  # Decoder Input for Transformers (Decoder Attention)
            self.pred_len  = 7
        else:
            self.seq_len   = size[0]
            self.label_len = size[1]
            self.pred_len  = size[2]

        assert features in ['S', 'M']  # single variable (ex. exchange rate) multi variable (ex. stocks, crypto )
        assert flag in ['train', 'test', 'val']

        self.data_path = data_path
        self.cols = cols
        purpose_map = {'train': 0, 'val': 1, 'test': 2}
        self.dataset_purpose = purpose_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.__read_data__()

    def __read_data__(self):
        # Scaler
        self.scaler = StandardScaler()

        # Read data
        df_raw = pd.read_csv(self.data_path) # df_raw.columns: ['date', ...(other features), target feature]

        # Rearrange Columns with Intended Order
        if self.cols:
            assert type(self.cols) is list
            df_raw = df_raw[self.cols]

        assert df_raw is not None, 'No data found'

        if self.cols:
            df_raw = df_raw[self.cols]
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')

        df_raw = df_raw[['date'] + cols + [self.target]]

        # Get how many data in train, val, test
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # EX. border_marks_start : start [0 : start of train, 7000 : start of val, 9000 : start of test]
        # EX. border_marks_end   : end   [7000 : end of train, 9000 : end of val, 10000 : end of test]
        # EX. border_start, border = ( 0, 7000 if train ) ( 7000, 9000 if val ) ( 9000, 10000 if test )
        border_marks_start = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border_marks_end   = [num_train, num_train + num_vali, len(df_raw)]
        border_start = border_marks_start[self.dataset_purpose]
        border_end   = border_marks_end[self.dataset_purpose]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]  # Exclude 'date'
            df_data = df_raw[cols_data]
        else:  # self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border_marks_start[0]:border_marks_end[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # making data_stamp : Not to Forget Date Information
        df_stamp = df_raw[['date']][border_start:border_end]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['year']    = df_stamp.date.apply(lambda row: row.year, 1)
        df_stamp['month']   = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day']     = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour']    = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute']  = df_stamp.date.apply(lambda row: row.minute, 1)

        data_stamp = df_stamp.drop(['date'], axis=1).values
        self.data_stamp = data_stamp

        self.data_x = data[border_start:border_end]
        self.data_y = data[border_start:border_end]

    def __getitem__(self, index):
        # indexes
        seq_begin = index
        seq_end   = seq_begin + self.seq_len
        lap_begin = seq_end   - self.label_len  # Overlapped part Especially for Transformers
        lap_end   = lap_begin + self.label_len + self.pred_len

        # data
        seq_x = self.data_x[seq_begin:seq_end]
        seq_y = self.data_y[lap_begin:lap_end]
        seq_x_mark = self.data_stamp[seq_begin:seq_end]  # Date Information
        seq_y_mark = self.data_stamp[lap_begin:lap_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class TimeSeriesPred(Dataset):
    def __init__(self,
                 data_path: str = 'data/crypto/btc_data_5min.csv',
                 flag: str = 'train',
                 features: str = 'S',
                 target: str = 'Low',
                 scale: bool = True,
                 size=None,
                 cols: List = []
                 ):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 28
            self.label_len = 7  # Decoder Input for Transformers (Decoder Attention)
            self.pred_len = 7
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert features in ['S', 'M']  # single variable (ex. exchange rate) multi variable (ex. stocks, crypto )
        assert flag in ['pred']

        self.data_path = data_path
        self.cols = cols

        purpose_map = {'train': 0, 'val': 1, 'test': 2}
        self.dataset_purpose = purpose_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.__read_data__()

    def __read_data__(self):
        # Scaler
        self.scaler = StandardScaler()

        # Read data
        df_raw = pd.read_csv(self.data_path) # df_raw.columns: ['date', ...(other features), target feature]

        if self.cols:
            assert type(self.cols) is list
            df_raw = df_raw[self.cols]

        assert df_raw is not None, 'No data found'

        if self.cols:
            df_raw = df_raw[self.cols]
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')

        df_raw = df_raw[['date'] + cols + [self.target]]

        # borders
        border_start = len(df_raw) - self.seq_len
        border_end   = len(df_raw)

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:  # self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border_start:border_end]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])

        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)

        data_stamp = df_stamp.drop(['date'], 1).values

        self.data_x = data[border_start:border_end]
        self.data_y = data[border_start:border_end]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        seq_begin = index
        seq_end = seq_begin + self.seq_len
        lap_begin = seq_end - self.label_len  # Overlapped part Especially for Transformers
        lap_end = lap_begin + self.label_len + self.pred_len

        seq_x = self.data_x[seq_begin:seq_end]
        seq_y = self.data_y[lap_begin:lap_begin + self.label_len]
        seq_x_mark = self.data_stamp[seq_begin:seq_end]
        seq_y_mark = self.data_stamp[lap_begin:lap_end]

        batch = {'X': from_numpy(seq_x).float(),
                 'y': from_numpy(seq_y).float(),
                 'X_mark': seq_x_mark,
                 'y_mark': seq_y_mark}

        return batch

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
