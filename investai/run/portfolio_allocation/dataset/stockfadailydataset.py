# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from finta import TA
from run.shared.tickers import DOW_30_TICKER
from tqdm import tqdm
from tvDatafeed import Interval, TvDatafeed

from raw_data.train.company_info import CompanyInfo
from shared.program import Program
from run.shared.dataset.dataengineer import DataEngineer as DE
from run.shared.dataset.candlestickengineer import CandlestickEngineer as CSE

# For Debugging
from shared.utils import reload_module  # noqa
from IPython.display import display  # noqa


class StockFaDailyDataset:
    def __init__(self, program: Program, tickers: List[str], dataset_split_coef: float):
        TICKERS = deepcopy(tickers)
        TICKERS.remove("DOW")  # TODO: Fixme: "DOW" is not in DJI30 or what?
        #
        self.program: Program = program
        self.tickers = TICKERS
        self.unique_columns = ["date", "tic"]
        self.base_columns = ["open", "high", "low", "close", "volume", "changePercent"]
        # Technical analysis indicators
        self.ta_indicators = []
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
        self.dataset_split_coef = dataset_split_coef

    @property
    def train_dataset(self) -> pd.DataFrame:
        """Split dataset into train and test"""
        if self.program.args.project_verbose > 0:
            print("Train dataset from", self.dataset["date"].min(), "to",
                  DE.get_split_date(self.dataset, self.dataset_split_coef))
        df = self.dataset[self.dataset["date"] < DE.get_split_date(self.dataset, self.dataset_split_coef)]
        return df

    @property
    def test_dataset(self) -> pd.DataFrame:
        """Split dataset into train and test"""
        df = self.dataset[self.dataset["date"] >= DE.get_split_date(self.dataset, self.dataset_split_coef)]
        df.index = df["date"].factorize()[0]
        return df

    def get_features(self):
        """Return features for training and testing"""
        return self.fa_indicators + self.ta_indicators + self.base_columns

    def preprocess(self) -> pd.DataFrame:
        """Return dataset"""
        self.dataset = self.get_stock_dataset()
        return self.dataset

    def get_stock_dataset(self) -> pd.DataFrame:
        df = pd.DataFrame()

        iterable = tqdm(self.tickers) if self.program.args.project_verbose > 0 else self.tickers
        for tic in iterable:
            if type(iterable) is tqdm: iterable.set_description(f"Processing {tic}")
            raw_data: CompanyInfo = self.load_raw_data(tic)  # Load tickers raw_data
            feature_data = self.add_fa_features(raw_data)  # Add features
            df = pd.concat([feature_data, df])  # Add ticker to dataset

        df.insert(0, "date", df.index)
        df = df.sort_values(by=['tic'])
        df = DE.clean_dataset_from_missing_tickers_by_date(df)
        df = df.sort_values(by=self.unique_columns)
        df.index = df["date"].factorize()[0]
        DE.check_dataset_correctness_assert(df)
        return df

    def add_fa_features(self, ticker_raw_data: CompanyInfo) -> pd.DataFrame:
        """
        Add fundamental analysis features to dataset
        Merge tickers information into one pd.Dataframe
        """
        # Prices
        prices = ticker_raw_data.data_detailed[self.base_columns]
        prices.insert(0, "tic", ticker_raw_data.symbol)

        # Fill before or forward
        prices = prices.fillna(method="bfill")
        prices = prices.fillna(method="ffill")

        # Ratios
        ratios = ticker_raw_data.financial_ratios.loc[self.fa_indicators].transpose()

        # Fill 0, where Nan/np.inf
        ratios = ratios.fillna(0)
        ratios = ratios.replace(np.inf, 0)

        merge = pd.merge(prices, ratios,
                         how="outer", left_index=True, right_index=True)
        filled = merge.fillna(method="bfill")
        filled = filled.fillna(method="ffill")
        clean = filled.drop(filled[~filled.index.str.contains(r"\d{4}-\d{2}-\d{2}")].index)  # Remove non-dates
        return clean

    @staticmethod
    def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick features to dataset"""
        candle_size = CSE.candlestick_size(df)
        body_size = CSE.body_size(df)
        upper_shadow = CSE.candlestick_up_shadow(df)
        down_shadow = CSE.candlestick_down_shadow(df)

        # Add features
        df["shadow_size_pct"] = (upper_shadow + down_shadow) / candle_size
        df["body_size_pct"] = body_size / candle_size
        df["candle_up_shadow_pct"] = upper_shadow / candle_size
        df["candle_down_shadow_pct"] = down_shadow / candle_size
        df["candle_direction"] = CSE.candlestick_direction(df)
        return df

    @staticmethod
    def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features to dataset"""
        # Add features
        df["macd"] = TA.MACD(df).SIGNAL
        df["boll_ub"] = TA.BBANDS(df).BB_UPPER
        df["boll_lb"] = TA.BBANDS(df).BB_LOWER
        df["rsi_30"] = TA.RSI(df, period=30)
        df["adx_30"] = TA.ADX(df, period=30)
        # df["close_30_sma"] = TA.SMA(df, period=30) # Unnecessary correlated with boll_lb
        # df["close_60_sma"] = TA.SMA(df, period=60) # Unnecessary correlated with close_30_sma
        df = df.fillna(0)  # Nan will be replaced with 0
        return df

    # TODO: Create function for downloading raw_data using yfinance and another one for TradingView
    def download(self, ticker,
                 exchange: str,
                 interval: Interval | str,
                 period: str = '',  # FIXME: not used
                 n_bars: int = 10) -> pd.DataFrame:
        """Return raw raw_data"""
        # df = yf.download(tickers=tickers, period=period, interval=interval) # BUG: yfinance bad candlestick raw_data

        tv = TvDatafeed()
        df = tv.get_hist(symbol=ticker, exchange=exchange, interval=interval, n_bars=n_bars)
        df = df.fillna(0)  # Nan will be replaced with 0

        return df

    def save_dataset(self, file_path) -> None:
        """Save dataset
        :param file_path:
        """

        if self.program.args.project_verbose > 0:
            print(f"Saving dataset to: {file_path}")
        self.dataset.to_csv(file_path, index=True)

    def load_raw_data(self, tic) -> CompanyInfo:
        """Check if folders with ticker exists and load all raw_data from them into CompanyInfo class"""
        data = {"symbol": tic}
        files = deepcopy(CompanyInfo.Names.list())
        files.remove("symbol")
        for f in files:
            tic_file = self.program.project_structure.data.tickers.joinpath(tic).joinpath(f + ".csv")
            if tic_file.exists():
                data[f] = pd.read_csv(tic_file, index_col=0)
            else:
                raise FileExistsError(f"File not exists: {tic_file}")
        return CompanyInfo(**data)

    def load_dataset(self, file_path: str) -> None:
        """Load dataset"""
        if self.program.args.project_verbose > 0:
            print(f"Loading dataset from: {file_path}")
        self.dataset = pd.read_csv(file_path, index_col=0)


def t1():
    from dotenv import load_dotenv

    program = Program()
    program.args.project_verbose = 1
    program.args.debug = True
    load_dotenv(dotenv_path=program.project_structure.root.as_posix())
    dataset = StockFaDailyDataset(
        program,
        tickers=DOW_30_TICKER,
        dataset_split_coef=program.args.dataset_split_coef
    )
    return {
        "d": dataset.get_stock_dataset(),
    }


def main():
    from dotenv import load_dotenv

    program = Program()
    load_dotenv(dotenv_path=program.project_structure.root.as_posix())
    dataset = StockFaDailyDataset(
        program,
        tickers=DOW_30_TICKER,
        dataset_split_coef=program.args.dataset_split_coef)
    dataset.preprocess()
    dataset.save_dataset((program.project_structure.datasets.joinpath(dataset.__class__.__name__.lower() + ".csv"))
                         if program.args.dataset_path is None
                         else program.args.dataset_path)


if __name__ == "__main__":
    main()
