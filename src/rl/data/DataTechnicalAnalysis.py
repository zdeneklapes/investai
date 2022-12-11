# -*- coding: utf-8 -*-
from typing import Callable

import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

from rl.data import DataPreprocessing
from rl.data.types import TimeInterval


class DataTechnicalAnalysis(DataPreprocessing):
    def __init__(
        self,
        start_date: str,
        end_date: str,
        cb_readfile: Callable = pd.read_csv,
        ticker_list: list = None,
        time_interval: TimeInterval = "1d",
    ):
        super().__init__(start_date, end_date, cb_readfile, ticker_list, time_interval=time_interval)

    def preprocess_data(self) -> pd.DataFrame:
        # Data
        self.download_data(self.ticker_list)
        fe = FeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature=False)
        df = fe.preprocess_data(self.dataframe)

        # add covariance matrix as states
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]

        cov_list = []
        return_list = []

        # look back is one year
        lookback = 252
        for i in range(lookback, len(df.index.unique())):
            data_lookback = df.loc[i - lookback : i, :]
            price_lookback = data_lookback.pivot_table(index="date", columns="tic", values="close")
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)

            covs = return_lookback.cov().values
            cov_list.append(covs)

        df_cov = pd.DataFrame({"date": df.date.unique()[lookback:], "cov_list": cov_list, "return_list": return_list})
        df = df.merge(df_cov, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)

        return df
