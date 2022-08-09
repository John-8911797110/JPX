import numpy as np
import pandas as pd


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


if __name__ == '__main__':
    df = pd.read_csv('example_test_files/sample_submission.csv')
    df2 = pd.read_csv('supplemental_files/stock_prices.csv')
    df3 = pd.concat([df, df2['Target']], axis=1)
    df3gb = df3.groupby('Date')
    df4 = pd.DataFrame()
    for date, data in df3gb:
        data.sort_values(by='Target', ascending=False, inplace=True)
        data['Rank'] = np.arange(data.shape[0])
        df4 = pd.concat([df4, data], axis=0)
    # df3 = pd.DataFrame(df3gb)
    # df3.reset_index(inplace=True)
    # df3 = df3.sort_values(by='Target')
    sr = calc_spread_return_sharpe(df4)

    print(sr)