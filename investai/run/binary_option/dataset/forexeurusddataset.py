# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
import pandas as pd
from finta import TA
from run.binary_option.main import DATASET_PATH
from tvDatafeed import Interval, TvDatafeed


class ForexDataset:
    def __init__(self):
        # TODO: load raw_data for all tickers
        self.tickers = None
        self.indicators = None
        self.dataset = None

        # TODO: preprocess raw_data

        # TODO: save dataset

    def preprocess(self) -> pd.DataFrame:
        """Return dataset"""
        #
        df = self.download("EURUSD", exchange="OANDA", interval=Interval.in_1_minute, n_bars=1000)
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

    def feature_correlation_drop(
        self, df: pd.DataFrame, threshold: float = 0.6, method: Literal["pearson", "kendall", "spearman"] = "pearson"
    ) -> pd.DataFrame:
        """Drop features with correlation higher than threshold"""
        corr = df.corr(method=method)
        triu = pd.DataFrame(np.triu(corr.T).T, corr.columns, corr.columns)
        threshhold = triu[(triu > threshold) & (triu < 1)]
        return threshhold

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

    def save(self) -> None:
        """Save dataset"""

    def load(self) -> pd.DataFrame:
        """Load dataset"""


def main():
    # TODO
    pass


if __name__ == "__main__":
    main()
