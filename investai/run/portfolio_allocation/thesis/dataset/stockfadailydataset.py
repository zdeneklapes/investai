# -*- coding: utf-8 -*-
"""
Stock fundamental analysis dataset
"""
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd
from finta import TA
from IPython.display import display  # noqa
from tqdm import tqdm

from run.shared.dataset.candlestickengineer import CandlestickEngineer as CSE
from run.shared.dataset.dataengineer import DataEngineer as DE
from run.shared.tickers import DOW_30_TICKER
from run.shared.memory import Memory
from shared.program import Program
from shared.reload import reload_module  # noqa
from shared.utils import log_artifact
from raw_data.tvdatafeed import Interval, TvDatafeed
from extra.math.finance.ticker.ticker import Ticker


class StockFaDailyDataset(Memory):
    def __init__(self, program: Program, tickers: List[str], split_coef: float, df: pd.DataFrame = pd.DataFrame()):
        TICKERS = deepcopy(tickers)
        TICKERS.remove("DOW")  # TODO: Fixme: "DOW" is not in DJI30 or what?
        #
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
        self.dataset_split_coef = split_coef
        super().__init__(program, df)

    @property
    def train_dataset(self) -> pd.DataFrame:
        """Split dataset into train and test"""
        if "i" in self.program.args.project_verbose:
            self.program.log.info(
                f"Train dataset from {self.df['date'].iloc[0]} to {DE.get_split_date(self.df, self.dataset_split_coef)}"
                # noqa
            )
        df: pd.DataFrame = self.df[self.df["date"] < DE.get_split_date(self.df, self.dataset_split_coef)]
        return df

    @property
    def test_dataset(self) -> pd.DataFrame:
        """Split dataset into train and test"""
        df: pd.DataFrame = self.df[
            self.df["date"] >= DE.get_split_date(self.df, self.dataset_split_coef)
        ]
        df.index = df["date"].factorize()[0]
        return df

    def get_features(self):
        """Return features for training and testing"""
        return self.fa_indicators + self.ta_indicators + self.base_columns

    def preprocess(self) -> pd.DataFrame:
        """Return dataset"""
        self.df = self.get_stock_dataset()
        return self.df

    def get_stock_dataset(self) -> pd.DataFrame:
        df = pd.DataFrame()

        iterable = tqdm(self.tickers) if "i" in self.program.args.project_verbose else self.tickers
        for tic in iterable:
            if isinstance(iterable, tqdm):
                iterable.set_description(f"Processing {tic}")
            raw_data: Ticker = self.load_raw_data(tic)  # Load tickers raw_data
            feature_data = self.add_fa_features(raw_data)  # Add features
            df = pd.concat([feature_data, df])  # Add ticker to dataset

        df.insert(0, "date", df.index)
        df = df.sort_values(by=["tic"])
        df = DE.clean_dataset_from_missing_tickers_by_date(df)
        df = df.sort_values(by=self.unique_columns)
        df.index = df["date"].factorize()[0]
        DE.check_dataset_correctness_assert(df)
        return df

    def add_fa_features(self, ticker_raw_data: Ticker) -> pd.DataFrame:
        """
        Add fundamental analysis features to dataset
        Merge tickers information into one pd.Dataframe
        """
        # Prices
        prices = ticker_raw_data.data_detailed[self.base_columns]
        prices.insert(0, "tic", ticker_raw_data.ticker)

        # Fill before or forward
        prices = prices.fillna(method="bfill")
        prices = prices.fillna(method="ffill")

        # Ratios
        ratios = ticker_raw_data.financial_ratios.loc[self.fa_indicators].transpose()

        # Fill 0, where Nan/np.inf
        ratios = ratios.fillna(0)
        ratios = ratios.replace(np.inf, 0)

        merge = pd.merge(prices, ratios, how="outer", left_index=True, right_index=True)
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
    def download(
        self, ticker, exchange: str, interval: Interval | str, period: str = "", n_bars: int = 10  # FIXME: not used
    ) -> pd.DataFrame:
        """Return raw raw_data"""
        # df = yf.download(tickers=tickers, period=period, interval=interval) # BUG: yfinance bad candlestick raw_data

        tv = TvDatafeed()
        df = tv.get_hist(symbol=ticker, exchange=exchange, interval=interval, n_bars=n_bars)
        df = df.fillna(0)  # Nan will be replaced with 0

        return df

    def load_raw_data(self, tic) -> Ticker:
        """Check if folders with ticker exists and load all raw_data from them into Ticker class"""
        data = {"ticker": tic}
        files = deepcopy(Ticker.Names.list())
        files.remove("symbol")
        for f in files:
            tic_file = self.program.args.folder_ticker.joinpath(tic).joinpath(f + ".csv")
            if tic_file.exists():
                data[f] = pd.read_csv(tic_file, index_col=0)
            else:
                raise FileExistsError(f"File not exists: {tic_file}")
        return Ticker(**data)


def t1() -> Dict:
    """
    :return: dict
    """
    from dotenv import load_dotenv

    program = Program()
    program.args.project_verbose = 1
    program.args.debug = True
    load_dotenv(dotenv_path=program.args.folder_root.as_posix())

    dataset = StockFaDailyDataset(program, tickers=DOW_30_TICKER, split_coef=program.args.dataset_split_coef)
    return {"dataset": dataset, "d": dataset.get_stock_dataset()}


def main():
    program = Program()
    dataset = StockFaDailyDataset(program, tickers=DOW_30_TICKER, split_coef=program.args.dataset_split_coef)
    dataset.preprocess()
    file_path = program.args.dataset_paths[0]
    dataset.save_csv(file_path)

    # Save to wandb
    if program.is_wandb_enabled(check_init=False):
        log_artifact(program.args, file_path, file_path.name.split('.')[0], "dataset", {"path": file_path.as_posix()})


if __name__ == "__main__":
    main()
