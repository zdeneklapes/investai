# -*- coding: utf-8 -*-
from copy import deepcopy  # noqa
from typing import Tuple

import numpy as np  # noqa
import pandas as pd
from IPython.display import display  # noqa
from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program
from shared.utils import calculate_return_from_weights, reload_module  # noqa
from tqdm import trange


# TODO: add baseline S@P500
# TODO: add baseline DJI30


class Baseline:
    def __init__(self, program):
        self.program = program
        self.returns = pd.DataFrame()

    def get_weights(self, mean, s) -> pd.DataFrame:
        ef = EfficientFrontier(mean, s, weight_bounds=self.bounds)
        ef.min_volatility()
        cleaned_weights_min_var = ef.clean_weights()
        ef = EfficientFrontier(mean, s, weight_bounds=self.bounds)
        ef.max_quadratic_utility()
        cleaned_weights_max_quadratic_utility = ef.clean_weights()
        ef = EfficientFrontier(mean, s, weight_bounds=self.bounds)
        ef.max_sharpe()
        cleaned_weights_max_sharpe = ef.clean_weights()
        return {
            "minimum_variance": cleaned_weights_min_var,
            "maximum_quadratic_utility": cleaned_weights_max_quadratic_utility,
            "maximum_sharpe": cleaned_weights_max_sharpe,
        }

    def get_returns(self, df: pd.DataFrame, bounds: Tuple = (0, 1)) -> pd.DataFrame:
        """Source: https://github.com/AI4Finance-Foundation/FinRL-Tutorials
        :param df:
        :param bounds:
        """
        dates = df["date"].unique()
        del df["date"]

        if not self.returns.empty:
            return self.returns

        iterable = (
            trange(4, dates.size - 1) if self.program.args.project_verbose else range(4, dates.size - 1)
        )
        for i in iterable:
            mean_annual_return = mean_historical_return(df.iloc[:i])
            s_covariance_matrix = CovarianceShrinkage(df.iloc[:i]).ledoit_wolf()

            #
            weights = self.get_weights(mean_annual_return, s_covariance_matrix)

            assert list(df.iloc[i].index) == list(weights["minimum_variance"].keys())

            #
            return_min_var = calculate_return_from_weights(
                df.iloc[i].values,
                df.iloc[i - 1].values,
                list(weights["minimum_variance"].values()),
            )
            return_max_quadratic_util = calculate_return_from_weights(
                df.iloc[i].values,
                df.iloc[i - 1].values,
                list(weights["maximum_quadratic_utility"].values()),
            )
            return_max_sharpe = calculate_return_from_weights(
                df.iloc[i].values,
                df.iloc[i - 1].values,
                list(weights["maximum_sharpe"].values()),
            )

            #
            next_return = {
                "date": dates[i],
                f"minimum_variance_{bounds[0]}_{bounds[1]}": return_min_var,
                f"maximum_quadratic_utility_{bounds[0]}_{bounds[1]}": return_max_quadratic_util,
                f"maximum_sharpe_{bounds[0]}_{bounds[1]}": return_max_sharpe, }
            self.returns = pd.concat([self.returns, pd.DataFrame(next_return, index=[0])], ignore_index=True)

        return self.returns

    def save(self, file_path) -> None:
        if self.program.args.project_verbose > 0:
            self.program.log.info(f"Saving return to: {file_path}")
        self.returns.to_csv(file_path, index=True)

    def load(self, file_path: str) -> None:
        if self.program.args.project_verbose > 0:
            self.program.log.info(f"Loading returns from: {file_path}")
        self.returns = pd.read_csv(file_path, index_col=0)


def t1():
    from dotenv import load_dotenv

    program = Program()
    program.args.project_verbose = 1
    program.args.dataset_path = program.args.folder_dataset.joinpath("stockfadailydataset.csv").as_posix()
    load_dotenv(dotenv_path=program.args.folder_root.joinpath(".env").as_posix())

    dataset = StockFaDailyDataset(program=program, tickers=DOW_30_TICKER,
                                  dataset_split_coef=program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    d_tics = dataset.dataset[["tic", "close", "date"]].sort_values(by=["tic", "date"])
    d = {"date": d_tics["date"].unique()}
    d.update({tic: d_tics[d_tics["tic"] == tic]["close"] for tic in d_tics["tic"].unique()})
    df = pd.DataFrame(d)
    baseline = Baseline(program)
    # baseline.get_returns()

    return {
        "dataset": dataset,
        "d_tics": d_tics,
        "df": df,
        "baseline": baseline,
    }


def main():
    program = Program()
    program.args.project_verbose = 1
    program.args.dataset_path = program.args.folder_dataset.joinpath("stockfadailydataset.csv").as_posix()

    #
    dataset = StockFaDailyDataset(program=program, tickers=DOW_30_TICKER,
                                  dataset_split_coef=program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    d_tics = dataset.dataset[["tic", "close", "date"]].sort_values(by=["tic", "date"])
    d = {"date": d_tics["date"].unique()}
    d.update({tic: d_tics[d_tics["tic"] == tic]["close"] for tic in d_tics["tic"].unique()})
    df = pd.DataFrame(d)

    #
    baseline = Baseline(program)
    baseline.get_returns(df=df, bounds=(0, 1))
    baseline.save(program.args.baseline_path)


if __name__ == "__main__":
    main()
