# -*- coding: utf-8 -*-
"""Candlestick engineer"""
from functools import wraps

import numpy as np
import pandas as pd
from numpy import ndarray


# TODO: Add df_name and get the arg from **kwargs
def check_dataframe_columns(required_columns=["open", "high", "low", "close"]):
    """Check if dataframe has the required columns"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[0], pd.DataFrame):
                raise ValueError("First argument must be a pandas dataframe")

            if len(args) != 1:
                raise ValueError("Only one argument is allowed")

            for column in required_columns:
                if column not in args[0].columns:
                    raise ValueError(f"Missing column {column} in dataframe")

            return func(*args, **kwargs)

        return wrapper

    return decorator


class CandlestickEngineer:
    """Candlestick feature"""

    @staticmethod
    @check_dataframe_columns(required_columns=["high", "low"])
    def candlestick_size(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get candle size"""
        return abs(dataframe["high"] - dataframe["low"])

    @staticmethod
    @check_dataframe_columns(required_columns=["open", "close"])
    def body_size(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get body size"""
        return abs(dataframe["close"] - dataframe["open"])

    @staticmethod
    @check_dataframe_columns(required_columns=["open", "close", "high"])
    def candlestick_up_shadow(dataframe: pd.DataFrame) -> ndarray:
        """Get upper shadow"""
        return np.where(
            (dataframe["close"] > dataframe["open"]),
            dataframe["high"] - dataframe["close"],
            dataframe["high"] - dataframe["open"],
        )

    @staticmethod
    @check_dataframe_columns(required_columns=["open", "close", "low"])
    def candlestick_down_shadow(dataframe: pd.DataFrame) -> ndarray:
        """Get lower shadow"""
        return np.where(
            (dataframe["close"] > dataframe["open"]),
            dataframe["open"] - dataframe["low"],
            dataframe["close"] - dataframe["low"],
        )

    @staticmethod
    @check_dataframe_columns(required_columns=["open", "close"])
    def candlestick_direction(dataframe: pd.DataFrame) -> ndarray:
        """
        Get candle direction
        :param dataframe: pd.DataFrame
        :return: ndarray: 1 - up, -1 - down
        """
        return np.where((dataframe["close"] - dataframe["open"]) > 0, 1, -1)

    @staticmethod
    def price_pct_change(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get price percent change"""
        return dataframe["close"].pct_change(1, fill_method="ffill")
