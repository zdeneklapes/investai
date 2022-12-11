# -*- coding: utf-8 -*-
import os
from typing import Callable

import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

from common.utils import now_time
from configuration.settings import DatasetDir
from rl.data import DataPreprocessing


class DataTechnicalAnalysis(DataPreprocessing):
    def __init__(
        self,
        start_date: str,
        end_date: str,
        non_preproccesed_data_filepath: str = None,
        preprocessed_data_filepath: str = None,
        cb: Callable = pd.read_csv,
        ticker_list: list = None,
    ):
        super().__init__(
            start_date, end_date, non_preproccesed_data_filepath, preprocessed_data_filepath, cb, ticker_list
        )
        self._non_preprocessed_data_filepath: str = None
        self._preprocessed_data_filepath: str = None or os.path.join(
            DatasetDir.AI4FINANCE,
            preprocessed_data_filepath if preprocessed_data_filepath else f"dataset_ta_{now_time()}.csv",
        )

    def get_preprocessed_data(self) -> pd.DataFrame:
        fe = FeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature=False)

        df = fe.preprocess_data(self.change_date(self.fetch_data_from_yahoo_finance()))

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

        self.data_preprocessed = df
        return self.data_preprocessed
