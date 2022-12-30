#%% import libraries
import warnings
from typing import List
from icecream import ic
import torch
from torch import from_numpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

#%%

def sliding_windows(data, seq_length, pred_length):
    x = []
    y = []

    for i in range(0, len(data) - seq_length - pred_length, 7):
        x_patch = data[i:(i + seq_length), :]
        y_patch = data[i + seq_length:i + seq_length + pred_length, 0]
        x.append(x_patch)
        y.append(y_patch)

    return np.array(x), np.array(y)

class MyDataset(Dataset):
    def __init__(self,
                     root: str,
                     columns_to_use: List[str],
                     target_column: str = 'target',
                     seq_length: int = 30,
                     pred_length: int = 1,
                     scaling: bool = True,
                     phase: str = 'train'):
        self.data_frame = pd.read_csv(root).set_index('일시')
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.scaling = scaling
        self.phase = phase
        assert self.phase in ['train', 'val', 'test'], 'phase must be train, val or test'

        self.target_column = target_column
        self.columns_to_use = columns_to_use
        if self.scaling:
            column_names_to_not_normalize = ['target',
                                             '강수일수비율',
                                             '평균습도(%rh)',
                                             '최저습도(%rh)',
                                             '일조율(%)',
                                             'holiday',
                                             'sin',
                                             'cos',
                                             't',
                                             'month',
                                             'year'
                                             ]
            self.column_names_to_normalize = [x for x in list(self.data_frame) if x not in column_names_to_not_normalize]
            self.scaler = StandardScaler()
            self.data_frame[self.column_names_to_normalize] = pd.DataFrame(
                self.scaler.fit_transform(self.data_frame[self.column_names_to_normalize].values),
                columns=self.column_names_to_normalize,
                index=self.data_frame.index)
            self.scaled = pd.DataFrame(
                self.scaler.fit_transform(self.data_frame[self.column_names_to_normalize].values),
                columns=self.column_names_to_normalize,
                index=self.data_frame.index
            )
            self.data_frame = pd.concat([self.data_frame[column_names_to_not_normalize], self.scaled], axis=1)
            self.means_scales = dict(zip(self.column_names_to_normalize,
                                     list(zip(self.scaler.mean_, self.scaler.scale_))))

        self.data_frame = self.data_frame[[self.target_column, *self.columns_to_use]]
        self.X, self.y = sliding_windows(self.data_frame.values, self.seq_length, self.pred_length)

        self.train_set_size = int(len(self.X) * 0.8)

        if self.phase == 'train': # backward => train set is the first 80% of the data
            self.X = self.X[:self.train_set_size, ...]
            self.y = self.y[:self.train_set_size, ...]
            assert self.X.shape == (self.train_set_size, self.seq_length, len(self.columns_to_use) + 1)
        elif self.phase == 'val':
            self.X = self.X[self.train_set_size:, ...]
            self.y = self.y[self.train_set_size:, ...]
        elif self.phase == 'test':
            self.X = np.expand_dims(self.data_frame.values[-self.seq_length:, :], 0)
            self.y = np.zeros_like(self.X[..., 0])
        else:
            raise ValueError('phase must be train, val or test')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        batch = {'X' : from_numpy(self.X[i]).float(), 'y' : from_numpy(self.y[i]).float()}
        return batch


def create_data_loader(root: str,
                       columns_to_use: List[str],
                       target_column: str = 'target',
                       seq_length: int = 30,
                       pred_length: int = 1,
                       scaling: bool = True,
                       phase: str = 'train',
                       batch_size: int = 32,
                       shuffle: bool = True,
                       num_workers: int = 0,
                       pin_memory: bool = False,
                       drop_last: bool = False,
                       ):
    dataset = MyDataset(root=root,
                        columns_to_use=columns_to_use,
                        target_column=target_column,
                        seq_length=seq_length,
                        pred_length=pred_length,
                        scaling=scaling,
                        phase=phase)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=drop_last)

    return data_loader

#%%
if __name__=='__main__':
    from orange.src.model.NLinear import NLinear, LitNLinear
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    pd.read_csv('orange/data/G_final.csv').columns
    model = NLinear(seq_len=365, pred_len=365)
    #%%
    data_set = MyDataset('orange/data/G_final.csv',
                     seq_length=365,
                     pred_length=365,
                     target_column='smooth_6_month',
                     columns_to_use=['sd_m_trend',
'sd_m_seasonal', 'sd_m_residual', '강수일수비율', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)',
'평균습도(%rh)', '최저습도(%rh)',
'holiday', 'sin', 'cos', 't', 'month', 'year'],
                     phase='train')
    loader = DataLoader(data_set, batch_size=32, shuffle=True)
    #%%
    X, y = next(iter(loader))['X'], next(iter(loader))['y']
    df = data_set.data_frame
    ref = pd.read_csv('orange/data/G_final.csv')
    ref['최고기온(℃)'].mean()
    ref['0.99_3'].mean()
    ref['7'].mean()
    plt.plot(ref['sd_m_residual'])
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(ref['target'])
    # plt.plot(ref['sd_m_trend']+ref['sd_m_seasonal'])
    plt.plot(ref['sd_m_trend']*ref['sd_m_seasonal'])
    plt.plot(ref['sd_a_trend'])
    plt.show()

    mean_absolute_error(ref['target'], ref['sd_a_trend']+ref['sd_a_seasonal'])
# #%%
    from scipy.signal import savgol_filter
    original = ref['target']
    yhat = savgol_filter(original, 365, 3)
    #
    # # plot original and yhat using matplotlib subplot to compare
    plt.figure(figsize=(20, 10))
    plt.plot(original, color='blue', label='original')
    plt.plot(yhat, color='red', label='savgol')
    plt.legend(loc='best')
    plt.show()

    #%%

    loader_y = loader.dataset.X[0][:, 0]
    ref_y = ref['smooth_6_month'].values[:365]
    mean, scale = loader.dataset.means_scales['smooth_6_month']
    for i in range(len(loader_y)):
        print(loader_y[i]*scale+mean, ref_y[i])
        print('...')