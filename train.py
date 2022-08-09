import numpy as np
import pandas as pd
import os
import argparse
import torch

from qlib.contrib.model.pytorch_add import ADDModel
from loss import ADDLoss
from ADD import ADDModel
from DataLoader import DataProcessor, DataLoader

def create_dataloader(args):
    df = pd.read_csv(args.data_file)
    processor = DataProcessor()
    cols_need = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.set_index(['Date', 'SecuritiesCode'])
    df = processor.process(df, process_type='RobustZscoreNorm', cols_need=cols_need, label_col='Target', clip=True)
    features, labels = df[cols_need], df['Target']
    train_dataloader = DataLoader(features, labels)
    return train_dataloader

def train_epoch(model, args):
    loss = ADDLoss()
    model = ADDModel(d_feat=5)
    train_loader = create_dataloader(args)
    for timestamp in train_loader.iter():
        X, y = train_loader.get_items(timestamp)
        X, y = torch.tensor(X.values, dtype=torch.float), torch.tensor(y.values, dtype=torch.float)
        pred = model(X)

    return
def main():
    args = parser_args()





def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-file', default='supplemental_files/stock_prices.csv')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()