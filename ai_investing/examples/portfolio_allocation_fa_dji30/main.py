# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path
from typing import Dict, Literal

import pandas as pd
import numpy as np
from finta import TA
from tvDatafeed import TvDatafeed, Interval
from meta.config_tickers import DOW_30_TICKER

from utils.project import get_argparse
from project_configs.experiment_dir import ExperimentDir
from project_configs.project_dir import ProjectDir
from project_configs.program import Program
from data.train.company_info import CompanyInfo


# from model_config.utils import load_all_initial_symbol_data


class StockDataset:
    def __init__(self, program: Program):
        TICKERS = deepcopy(DOW_30_TICKER)
        TICKERS.remove("DOW")  # TODO: "DOW" is not DJI30 or what?
        #
        self.program: Program = program
        self.tickers = TICKERS
        self.base_cols = ["date", "tic", "open", "high", "low", "close", "volume"]
        # Technical analysis indicators
        self.ta_indicators = None
        # Fundamental analysis indicators
        self.fa_indicators = [
            "operatingProfitMargin",
            "netProfitMargin",
            "returnOnAssets",
            "returnOnEquity",
            "currentRatio",
            "quickRatio",
            "cashRatio",
            "inventoryTurnover",
            "receivablesTurnover",
            "payablesTurnover",
            "debtRatio",
            "debtEquityRatio",
            "priceEarningsRatio",
            "priceBookValueRatio",
            "dividendYield",
        ]

        # Final dataset for training and testing
        self.dataset = None

    def data_split(self):
        """Split dataset into train and test"""

    def preprocess(self) -> pd.DataFrame:
        """Return dataset"""
        #
        # df = self.download("EURUSD", exchange='OANDA', interval=Interval.in_1_minute, n_bars=1000)
        df = self.load_raw(self.tickers, )
        # df = self.add_ta_features(df)
        df = self.add_fa_features(df)
        # df = self.add_candlestick_features(df)

        # save dataset
        self.dataset = df
        self.save()
        return df

    def add_fa_features(self, tickers_data: Dict[str, CompanyInfo]) -> pd.DataFrame:
        """
        Add fundamental analysis features to dataset
        Merge tickers information into one pd.Dataframe
        """
        df = pd.DataFrame()
        # for k, v in [("DIS", tickers_data["DIS"])]:
        for k, v in tickers_data.items():
            # Prices
            data = v.data_detailed[self.base_cols]
            data.insert(0, "tic", k)

            # Fill before or forward
            data = data.fillna(method="bfill")
            data = data.fillna(method="ffill")

            # Ratios
            ratios = v.financial_ratios.loc[self.fa_indicators].transpose()

            # Fill 0, where Nan/np.inf
            ratios = ratios.fillna(0)
            ratios = ratios.replace(np.inf, 0)

            #
            merge = pd.merge(data, ratios, how="outer", left_index=True, right_index=True)
            filled = merge.fillna(method="bfill")
            filled = filled.fillna(method="ffill")
            clean = filled.drop(filled[~filled.index.str.contains("\d{4}-\d{2}-\d{2}")].index)
            df = pd.concat([clean, df])

        df.insert(0, "date", df.index)
        assert df.isna().any().any() is False  # Can't be any Nan/np.inf values
        return df

    def clean_dataset_from_missing_stock_in_some_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Take only those dates where we have data for all stock in each day
        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        df: pd.DataFrame
        """
        # max_size = df.groupby("date").size().unique().max()
        _d = df.groupby("date").size()
        binary = _d.values == 29
        latest_date = _d[binary].index[0]
        df = df[df["date"] > latest_date]
        return df

    def give_index_to_each_day(self, df: pd.DataFrame) -> pd.DataFrame:
        dataset = df.sort_values(by="date")
        dataset.index = dataset["date"].factorize()[0]
        assert dataset.groupby("date").size().unique().size == 1
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
        self.dataset.to_csv(self.program.dataset_path, index=False)

    def load_raw(self,
                 tickers: list,
                 directory: Path
                 ) -> Dict[str, CompanyInfo]:
        """Check if folders with tickers exists and load all data from them into CompanyInfo class"""
        tickers_data: Dict[str, CompanyInfo] = {}
        for tic in tickers:
            data = {"symbol": tic}
            files = deepcopy(CompanyInfo.Names.list())
            files.remove("symbol")
            for f in files:
                tic_file = directory.joinpath(tic).joinpath(f + ".csv")
                if tic_file.exists():
                    data[f] = pd.read_csv(tic_file, index_col=0)
                else:
                    raise FileExistsError(f"File not exists: {tic_file}")
            tickers_data[tic] = CompanyInfo(**data)

        return tickers_data

    def load_dataset(self) -> None:
        """Load dataset"""
        self.dataset = pd.read_csv(self.program.dataset_path)


class Train:
    def __init__(self):
        # TODO: load dataset
        self.dataset: StockDataset = None
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


def initilisation() -> Program:
    prj_dir = ProjectDir(root=Path("/Users/zlapik/my-drive-zlapik/0-todo/ai-investing"))
    _, args = get_argparse()
    experiment_dir = ExperimentDir(Path(__file__).parent)
    experiment_dir.create_dirs()
    return Program(
        project_dir=prj_dir,
        experiment_dir=experiment_dir,
        args=args,
        train_date_start='2019-01-01',
        train_date_end='2020-01-01',
        test_date_start='2020-01-01',
        test_date_end='2021-01-01',
        dataset_path='out/dataset.csv',
    )


def t1():
    program = initilisation()
    d = StockDataset(program)
    return {
        "d": d,
    }


if __name__ == "__main__":
    program = initilisation()

    if program.DEBUG is not None:
        forex_dataset = StockDataset(program)
        df = forex_dataset.preprocess()
    if program.DEBUG is None:
        if program.args.dataset:
            forex_dataset = StockDataset(program)
            forex_dataset.preprocess()
            forex_dataset.save()
        if program.args.train:
            train = Train()
            train.train()
        if program.args.test:
            test = Test()
            test.test()
