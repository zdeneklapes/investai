# -*- coding: utf-8 -*-
# TODO: Test this
from functools import wraps

import numpy as np
import pandas as pd
from numpy import ndarray


def check_dataframe_columns(
    required_columns=["open", "high", "low", "close"]
):
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
    def candlestick_size(df: pd.DataFrame) -> pd.DataFrame:
        """Get candle size"""
        return abs(df["high"] - df["low"])

    @staticmethod
    @check_dataframe_columns(required_columns=["open", "close"])
    def body_size(df: pd.DataFrame) -> pd.DataFrame:
        """Get body size"""
        return abs(df["close"] - df["open"])

    @staticmethod
    @check_dataframe_columns(required_columns=["open", "close", "high"])
    def candlestick_up_shadow(df: pd.DataFrame) -> ndarray:
        """Get upper shadow"""
        return np.where((df["close"] > df["open"]), df["high"] - df["close"], df["high"] - df["open"])

    @staticmethod
    @check_dataframe_columns(required_columns=["open", "close", "low"])
    def candlestick_down_shadow(df: pd.DataFrame) -> ndarray:
        """Get lower shadow"""
        return np.where((df["close"] > df["open"]), df["open"] - df["low"], df["close"] - df["low"])

    @staticmethod
    @check_dataframe_columns(required_columns=["open", "close"])
    def candlestick_direction(df: pd.DataFrame) -> ndarray:
        """
        Get candle direction
        :param df: pd.DataFrame
        :return: ndarray: 1 - up, -1 - down
        """
        return np.where((df["close"] - df["open"]) > 0, 1, -1)
