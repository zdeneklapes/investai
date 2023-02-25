# -*- coding: utf-8 -*-
import pandas as pd


class Statistics:
    @staticmethod
    def add_cov(df: pd.DataFrame) -> pd.DataFrame:
        ##
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
            covs = return_lookback.cov().values

            #
            return_list.append(return_lookback)
            cov_list.append(covs)

        df_cov = pd.DataFrame({"date": df.date.unique()[lookback:], "cov_list": cov_list, "return_list": return_list})
        df = df.merge(df_cov, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df
