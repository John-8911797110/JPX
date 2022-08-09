import pandas as pd
import torch
import random
import numpy as np

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
EPS = 1e-12

class DataProcessor:
    """
    :argument
        df : whole dataframe without process
        cols_need: cols needed, other cols will be dropped. default: ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']
        process_type: process type like: RobustZscoreNorm
    :returns
        df : whole dateframe with process
        df_part : df of cols_need and labels
    """

    def __init__(self):
        self.df = None
        self.cols_need = None
        self.mean, self.std = None, None

    def process(self, df, process_type, cols_need=None, label_col=None, clip=True):
        print('---------------------Starting processing---------------------')
        if cols_need is None:
            self.cols_need = ['Open', 'High', 'Low', 'Close', 'Volume']
        else:
            self.cols_need = cols_need

        self.df = df
        x = self.df[self.cols_need].values
        if process_type == 'RobustZscoreNorm':
            self.mean = np.nanmedian(x, axis=0)
            self.std = np.nanmedian(np.abs(x - self.mean), axis=0)
            self.std += EPS
            self.std *= 1.4826
        else:
            self.mean, self.std = np.mean(x, axis=0), np.std(x, axis=0)

        x = self.df[self.cols_need]
        x -= self.mean
        x /= self.std
        print(f'features {process_type} has finished')
        # fillna with 0.
        select_nan = np.isnan(x.values)
        x.values[select_nan] = 0.
        print('features fillna with 0.')
        if clip:
            x = x.clip(-3, 3)

        self.df[self.cols_need] = x

        self.df.dropna(subset=self.cols_need, inplace=True)
        print('features dropna has finished')
        t = self.df[label_col].groupby('Date').rank(pct=True)
        t -= 0.5
        t *= 3.46
        self.df[label_col] = t
        print('labels CSRankNorm has finished')

        self.df.dropna(subset=cols_need+[label_col], inplace=True)
        print('labels dropna has finished')

        return self.df, self.df[self.cols_need + [label_col]]

    def init(self):
        cols = [
            ('feature', 'Open'),
            ('feature', 'High'),
            ('feature', 'Low'),
            ('feature', 'Close'),
            ('feature', 'Volume'),
            ('label', 'Target'),
        ]
        self.df.rename_axis(['datetime', 'instrument'], axis=0, inplace=True)
        self.df = self.df.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
        self.df.columns = pd.MultiIndex.from_tuples(cols)

    def prepare(self, segments, col_set):

        if segments:
            segs = 0.8
        else:
            segs = 1

        seg_position = int(segs*self.df.shape[0])
        seg1_df = self.df[: seg_position]
        seg2_df = self.df[seg_position: ]

        return seg1_df[col_set], seg2_df[col_set]



class DataLoader:
    """
    :argument
        df : whole dataframe
    :return
        df daily dataframe if split is False
        (features, labels)  if split is True
    """
    def __init__(self, df, batch_size=None):

        self.df = df
        self.batch_size = batch_size
        self.num_sample = len(self.df)
        self.dt_index = list(self.df.groupby('Date').size().index)

    def iter(self):
        random.seed(123)
        random.shuffle(self.dt_index)
        for timestamp in self.dt_index:
            yield timestamp

    def get_items(self, timestamp, split=True):

        if split:
            out = self.df.iloc[:, :-1].groupby('Date').get_group(timestamp), self.df.iloc[:, -1].groupby('Date').get_group(timestamp)
        else:
            out = self.df.groupby('Date').get_group(timestamp)

        return out

