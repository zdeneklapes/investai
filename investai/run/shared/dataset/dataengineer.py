# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
import pandas as pd


class DataEngineer:
    """Data Engineer"""

    @staticmethod
    def check_dataset_correctness_assert(dataframe: pd.DataFrame):
        """Check if all raw_data are correct"""
        assert (
            dataframe.groupby("date").size().unique().size == 1
        ), "The size of each group must be equal, that means in each date is teh same number of stock raw_data"
        assert not dataframe.isna().any().any(), "Can't be any Nan/np.inf values"

    @staticmethod
    def clean_dataset_from_missing_tickers_by_date(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Take only those dates where we have raw_data for all stock in each day
        Parameters
        ----------
        dataframe: pd.DataFrame

        Returns
        -------
        df: pd.DataFrame
        """
        tickers_count_in_date = dataframe.groupby("date").size()
        binary_where_are_all_tickers = tickers_count_in_date.values == tickers_count_in_date.values.max()
        # TODO: Fixme: This is not correct, because we can have missing raw_data in the middle of the dataset
        earliest_date = tickers_count_in_date[binary_where_are_all_tickers].index[0]
        updated_dataframe = dataframe[dataframe["date"] > earliest_date]
        return updated_dataframe

    @staticmethod
    def get_split_date(dataframe: pd.DataFrame, coef: float) -> str:
        """Return split date"""
        dates = dataframe["date"].unique()
        return dates[int(len(dates) * coef)]

    @staticmethod
    def fill_missing_data(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fill missing raw_data"""
        df = dataframe.fillna(method="bfill")
        df = df.fillna(method="ffill")
        df = df.fillna(0)
        df = df.replace(np.inf, 0)
        df = df.replace(-np.inf, 0)
        return df

    @staticmethod
    def feature_correlation_matrix(
        dataframe: pd.DataFrame, threshold: float = 0.6, method: Literal["pearson", "kendall", "spearman"] = "pearson"
    ) -> pd.DataFrame:
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
