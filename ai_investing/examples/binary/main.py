# -*- coding: utf-8 -*-
from os import getenv
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np
from finta import TA
from tvDatafeed import TvDatafeed, Interval

from common.Args import get_argparse
from common.utils import reload_module  # noqa # pylint: disable=unused-import
from project_configs.experiment_dir import ExperimentDir

TRAIN_DATE_START = '2019-01-01'
TRAIN_DATE_END = '2020-01-01'
TEST_DATE_START = '2020-01-01'
TEST_DATE_END = '2021-01-01'
DATASET_PATH = 'out/dataset.csv'
DEBUG = getenv('DEBUG', False)


def calculate_binary_profit(trades_amount: int, payout_rate: float, success_rate: float):
    # Trading stats
    profit_trades = success_rate * trades_amount
    loss_trades = (1 - success_rate) * trades_amount

    # Calculate profit and loss
    profit = profit_trades * (1 + payout_rate)
    loss = loss_trades * 1

    # Results
    print(f"{profit=}")
    print(f"{loss=}")
    print(f"{profit-loss=}")


class ForexDataset:
    def __init__(self):
        # TODO: load data for all tickers
        self.tickers = None
        self.indicators = None
        self.dataset = None

        # TODO: preprocess data

        # TODO: save dataset

    def preprocess(self) -> pd.DataFrame:
        """Return dataset"""
        #
        df = self.download("EURUSD", exchange='OANDA', interval=Interval.in_1_minute, n_bars=1000)
        df = self.add_ta_features(df)
        df = self.add_candlestick_features(df)

        # save dataset
        df.to_csv(DATASET_PATH, index=False)
        self.dataset = df
        return df

    def add_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features to dataset"""
        # Add features
        df["macd"] = TA.MACD(df).SIGNAL
        df["boll_ub"] = TA.BBANDS(df).BB_UPPER
        df["boll_lb"] = TA.BBANDS(df).BB_LOWER
        df["rsi_30"] = TA.RSI(df, period=30)
        df["dx_30"] = TA.ADX(df, period=30)
        # df["close_30_sma"] = TA.SMA(df, period=30) # Unnecessary correlated with boll_lb
        # df["close_60_sma"] = TA.SMA(df, period=60) # Unnecessary correlated with close_30_sma
        df = df.fillna(0)  # Nan will be replaced with 0
        return df

    def add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick features to dataset"""
        # get candlestick features
        candle_size = abs(df["high"] - df["low"])  # noqa
        body_size = abs(df["close"] - df["open"])  # noqa

        # get upper and lower shadow
        upper_shadow = np.where((df["close"] > df["open"]), df["high"] - df["close"], df["high"] - df["open"])
        lower_shadow = np.where((df["close"] > df["open"]), df["open"] - df["low"], df["close"] - df["low"])

        df["candle_size"] = candle_size
        df["body_size"] = body_size / candle_size  # pertentage of body size in candle size
        df["candle_upper_shadow"] = upper_shadow / candle_size  # pertentage of upper shadow in candle size
        df["candle_lower_shadow"] = lower_shadow / candle_size  # pertentage of lower shadow in candle size
        df["candle_direction"] = np.where((df["close"] - df["open"]) > 0, 1, 0)  # 1 - up, 0 - down
        return df

    def feature_correlation_drop(self, df: pd.DataFrame, threshold: float = 0.6,
                                 method: Literal['pearson', 'kendall', 'spearman'] = 'pearson') -> pd.DataFrame:
        """Drop features with correlation higher than threshold"""
        corr = df.corr(method=method)
        triu = pd.DataFrame(np.triu(corr.T).T, corr.columns, corr.columns)
        threshhold = triu[(triu > threshold) & (triu < 1)]
        return threshhold

    # TODO: Create function for downloading data using yfinance and another one for TradingView
    def download(self, ticker,
                 exchange: str,
                 interval: Interval | str,
                 period: str = '',  # FIXME: not used
                 n_bars: int = 10) -> pd.DataFrame:
        """Return raw data"""
        # df = yf.download(tickers=tickers, period=period, interval=interval) # BUG: yfinance bad candlestick data

        tv = TvDatafeed()
        df = tv.get_hist(symbol=ticker, exchange=exchange, interval=interval, n_bars=n_bars)
        df = df.fillna(0)  # Nan will be replaced with 0

        return df

    def save(self) -> None:
        """Save dataset"""

    def load(self) -> pd.DataFrame:
        """Load dataset"""


class Train:
    def __init__(self):
        # TODO: load dataset
        self.dataset: ForexDataset = None
        self.model = None

        # TODO: train

        # TODO: save model

    def train(self) -> None:
        pass

    def save_model(self) -> None:
        pass

    def load_model(self) -> None:
        pass


class Test:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.results = None

    def test(self) -> None:
        pass

    def save_results(self) -> None:
        pass

    def load_results(self) -> None:
        pass

    def plot_results(self) -> None:
        pass


def t1():
    forex_dataset = ForexDataset()
    data = forex_dataset.preprocess()
    corr_pearson = data.corr(method='pearson')
    corr_kendall = data.corr(method='kendall')
    corr_spearman = data.corr(method='spearman')
    pearson_triu = pd.DataFrame(np.triu(corr_pearson.T).T, corr_spearman.columns, corr_pearson.columns)
    pearson_threshhold = pearson_triu[(pearson_triu > 0.7) & (pearson_triu < 1)]

    return {
        'data': data,
        'pearson': corr_pearson,
        'kendall': corr_kendall,
        'spearman': corr_spearman,
        'pearson_triu': pearson_triu,
        'pearson_threshold': pearson_threshhold,
    }


def t2():
    forex_dataset = ForexDataset()
    data = forex_dataset.download("EURUSD", exchange='OANDA', interval=Interval.in_1_minute, n_bars=1000)
    return {
        'data': data,
    }


def t3():
    forex_dataset = ForexDataset()
    df = forex_dataset.preprocess()
    return {
        'data': df,
    }


if __name__ == "__main__":
    args_vars, args = get_argparse()
    experiment_dir = ExperimentDir(Path(__file__).parent)
    experiment_dir.create_dirs()
    # TODO: Create all unnesessary folders

    if DEBUG:
        forex_dataset = ForexDataset()
        df = forex_dataset.preprocess()
    if not DEBUG:
        if args.dataset:
            forex_dataset = ForexDataset()
            forex_dataset.preprocess()
            forex_dataset.save()
        if args.train:
            train = Train()
            train.train()
        if args.test:
            test = Test()
            test.test()
