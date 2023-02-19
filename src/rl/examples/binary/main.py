# -*- coding: utf-8 -*-

import pandas as pd
import yfinance as yf

from common.Args import get_argparse

TRAIN_DATE_START = '2019-01-01'
TRAIN_DATE_END = '2020-01-01'
TEST_DATE_START = '2020-01-01'
TEST_DATE_END = '2021-01-01'
DATASET_PATH = 'out/dataset.csv'


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
        # load
        data = self.download("EURUSD=X", "1w", "1m")  # noqa

        # ADD features
        dataset = None  # noqa

        # save

    def download(self, tickers, period, interval) -> pd.DataFrame:
        """Return raw data"""
        return yf.download(tickers=tickers, period=period, interval=interval)

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


if __name__ == "__main__":
    args_vars, args = get_argparse()
    # TODO: Create all unnesessary folders

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
