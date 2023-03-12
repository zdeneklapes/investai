# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
import pandas as pd


class DataEngineer:
    """Data Engineer"""

    @staticmethod
    def clean_dataset_from_missing_tickers_by_date(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Take only those dates where we have data for all stock in each day
        Parameters
        ----------
        dataframe: pd.DataFrame

        Returns
        -------
        df: pd.DataFrame
        """
        ticker_in_each_date = dataframe.groupby("date").size()
        where_are_all_tickers = ticker_in_each_date.values == ticker_in_each_date.values.max()

        # FIXME: This is not correct, because we can have missing data in the middle of the dataset
        earliest_date = ticker_in_each_date[where_are_all_tickers].index[0]
        df = dataframe[dataframe["date"] > earliest_date]
        return df

    @staticmethod
    def get_split_date(dataframe: pd.DataFrame, coef: float) -> str:
        """Return split date"""
        dates = dataframe["date"].unique()
        return dates[int(len(dates) * coef)]

    @staticmethod
    def fill_missing_data(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fill missing data"""
        df = dataframe.fillna(method="bfill")
        df = df.fillna(method="ffill")
        df = df.fillna(0)
        df = df.replace(np.inf, 0)
        df = df.replace(-np.inf, 0)
        return df

    @staticmethod
    def feature_correlation_matrix(dataframe: pd.DataFrame,
                                   threshold: float = 0.6,
                                   method: Literal['pearson', 'kendall', 'spearman'] = 'pearson') -> pd.DataFrame:
        """
        Return correlation matrix of features
        :param dataframe: pd.DataFrame
        :param threshold: float
        :param method: Literal['pearson', 'kendall', 'spearman']
        :return: pd.DataFrame
        """
        corr = dataframe.corr(method=method)
        triu = pd.DataFrame(np.triu(corr.T).T, corr.columns, corr.columns)
        threshhold = triu[(triu > threshold) & (triu < 1)]
        return threshhold
