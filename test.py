from DataLoader import DataProcessor, DataLoader
import numpy as np
import pandas as pd
from ADD1 import ADD
import qlib
from tqdm import tqdm

def df_length_fix(df, indexes, date):
    indexes = np.setdiff1d(indexes, df.index)
    df_tmp = pd.DataFrame(data=np.full((len(indexes), len(df.columns)), np.nan), index=indexes,
                       columns=df.columns)
    df = df.append(df_tmp).sort_index()
    df.insert(0, 'Date', date)
    df.index.rename('SecuritiesCode')
    df.reset_index()
    df = df.groupby('Date').get_group(date)
    return df


if __name__ == '__main__':
    # qlib.init()
    df = pd.read_csv('train_files/stock_prices.csv')
    # processor = DataProcessor()
    # add = ADD(d_feat=5, lr=0.00005, early_stop=2000)
    # cols_need = ['Open', 'High', 'Low', 'Close', 'Volume']
    # df = df.set_index(['Date', 'SecuritiesCode'])
    # _, df = processor.process(df, process_type='RobustZscoreNorm', cols_need=cols_need, label_col='Target', clip=True)
    # processor.init()

    # add.fit(processor)
    cols_need = ['SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target']
    df_train = df.set_index(['Date'])[cols_need]
    df_train_feat = df_train.loc['2017-01-04', :]
    stocks_all = df_train.loc['2021-12-03', :].index
    for date, _ in tqdm(df_train.groupby('Date')):

        df_train.loc[date, :] = df_length_fix(df_train.loc[date, :], stocks_all, date)

    pass
