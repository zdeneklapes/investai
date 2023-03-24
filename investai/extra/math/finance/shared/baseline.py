# -*- coding: utf-8 -*-
from copy import deepcopy  # noqa
from typing import Dict, Tuple, NoReturn

import numpy as np  # noqa
import pandas as pd
import matplotlib.pyplot as plt  # noqa
from matplotlib.axes import Axes  # noqa
import seaborn as sns  # noqa

from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program
from shared.utils import calculate_return_from_weights, reload_module  # noqa
from run.shared.memory import Memory
from tqdm import trange
import yfinance as yf


# TODO: add baseline S@P500
# TODO: add baseline DJI30

class Baseline(Memory):
    def __init__(self, program: Program, df: pd.DataFrame = pd.DataFrame()):
        super().__init__(program, df)

    def warren_buffet_returns(self) -> NoReturn:
        """Source:"""

    def indexes_returns(self) -> NoReturn:
        """Source:"""
        indexes = ["^DJI", "^GSPC", "^IXIC", "^RUT"]  # DJI, S&P500, NASDAQ, Russell 2000
        df_indexes_1 = yf.download(indexes, period="max", interval="1d", ignore_tz=True)
        df_indexes_2 = df_indexes_1.dropna()
        df_indexes_3 = df_indexes_2["Close"]
        df_indexes_4 = df_indexes_3.pct_change(periods=1).dropna()
        df_indexes_4.insert(0, "date", df_indexes_4.index)
        self.merge_with_df(df_indexes_4)

    def _create_weights(self, mean, s, bounds: Tuple) -> Dict:
        round_float_correction = 1e-3
        #
        ef = EfficientFrontier(mean, s, weight_bounds=bounds)
        ef.min_volatility()
        cleaned_weights_min_var = ef.clean_weights()
        assert sum(cleaned_weights_min_var.values()) <= 1 + round_float_correction
        ef = EfficientFrontier(mean, s, weight_bounds=bounds)
        ef.max_quadratic_utility()
        cleaned_weights_max_quadratic_utility = ef.clean_weights()
        assert sum(cleaned_weights_max_quadratic_utility.values()) <= 1 + round_float_correction
        ef = EfficientFrontier(mean, s, weight_bounds=bounds)
        ef.max_sharpe(risk_free_rate=-10.0)  # FIXME: -10.0 is a hack, should be ?, but it works
        cleaned_weights_max_sharpe = ef.clean_weights()
        assert sum(cleaned_weights_max_sharpe.values()) <= 1 + round_float_correction
        return {
            "minimum_variance": cleaned_weights_min_var,
            "maximum_quadratic_utility": cleaned_weights_max_quadratic_utility,
            "maximum_sharpe": cleaned_weights_max_sharpe,
        }

    def pypfopt_returns(self, dataset: pd.DataFrame, bounds: Tuple = (0, 1)) -> NoReturn:
        """
        source: https://github.com/AI4Finance-Foundation/FinRL-Tutorials
        Create returns using pyportfolioopt library and results are stored in self.df and returned
        :param dataset: pd.DataFrame: dataset
        :param bounds: Tuple: (min, max) bounds for weights
        :return: pd.DataFrame: self.df with returns
        """
        start_date: np.datetime64 = dataset["date"].unique()[0] - np.timedelta64(1, 'D')
        start_return = [0]
        rewards = pd.DataFrame({
            "date": [np.datetime_as_string(start_date, unit='D')],
            f"minimum_variance_{bounds[0]}_{bounds[1]}": start_return,
            f"maximum_quadratic_utility_{bounds[0]}_{bounds[1]}": start_return,
            f"maximum_sharpe_{bounds[0]}_{bounds[1]}": start_return,
        })
        dates = dataset["date"].unique()
        del dataset["date"]

        iterable = (trange(4, dates.size - 1) if self.program.args.project_verbose else range(4, dates.size - 1))
        for i in iterable:
            frequency = 252 if dataset.iloc[:i].shape[0] > 252 else dataset.iloc[:i].shape[0]
            mean_return = mean_historical_return(dataset.iloc[:i], frequency=frequency, compounding=True)
            s_covariance_matrix = CovarianceShrinkage(dataset.iloc[:i], frequency=frequency).ledoit_wolf()

            #
            weights = self._create_weights(mean_return, s_covariance_matrix, bounds)

            assert list(dataset.iloc[i].index) == list(weights["minimum_variance"].keys())

            #
            return_min_var = calculate_return_from_weights(
                dataset.iloc[i].values,
                dataset.iloc[i - 1].values,
                list(weights["minimum_variance"].values()),
            )
            return_max_quadratic_util = calculate_return_from_weights(
                dataset.iloc[i].values,
                dataset.iloc[i - 1].values,
                list(weights["maximum_quadratic_utility"].values()),
            )
            return_max_sharpe = calculate_return_from_weights(
                dataset.iloc[i].values,
                dataset.iloc[i - 1].values,
                list(weights["maximum_sharpe"].values()),
            )

            #
            next_return = {
                "date": dates[i],
                f"minimum_variance_{bounds[0]}_{bounds[1]}": return_min_var,
                f"maximum_quadratic_utility_{bounds[0]}_{bounds[1]}": return_max_quadratic_util,
                f"maximum_sharpe_{bounds[0]}_{bounds[1]}": return_max_sharpe, }
            rewards = pd.concat([rewards, pd.DataFrame(next_return, index=[0])], ignore_index=True)

        self.merge_with_df(rewards)

    def merge_with_df(self, rewards: pd.DataFrame, date_format: str = "%Y-%m-%d") -> NoReturn:
        rewards['date'] = pd.to_datetime(rewards["date"], format=date_format)
        if self.df.empty:
            self.df = rewards
        else:
            try:
                self.df['date'] = pd.to_datetime(self.df["date"], format=date_format)
            except ValueError as e:
                raise ValueError(f"self.df['date'] can't be converted to datetime64[ns], format {date_format}") from e
            self.df = pd.merge(self.df, rewards, on="date", how="left")
        self.df.dropna(inplace=True)


class TestBaseline:
    def __init__(self):
        self.program = Program()
        self.program.args.project_verbose = 1
        self.program.args.dataset_paths = self.program.args.folder_dataset.joinpath(
            "stockfadailydataset.csv").as_posix()
        self.program.args.baseline_path = self.program.args.folder_baseline.joinpath("baseline.csv").as_posix()
        self.baseline = Baseline(self.program)

    def t1(self):
        #
        dataset = StockFaDailyDataset(program=self.program, tickers=DOW_30_TICKER,
                                      split_coef=self.program.args.dataset_split_coef)
        dataset.load_csv(self.program.args.dataset_paths[0].as_posix())
        d_tics = dataset.df[["tic", "close", "date"]].sort_values(by=["tic", "date"])
        d = {"date": d_tics["date"].unique()}
        d.update({tic: d_tics[d_tics["tic"] == tic]["close"] for tic in d_tics["tic"].unique()})
        df = pd.DataFrame(d)

        #
        # baseline.get_returns()
        return {
            "dataset": dataset,
            "d_tics": d_tics,
            "df": df,
        }

    def t2(self):
        baseline = Baseline(program=Program())
        baseline.load_csv(self.program.args.baseline_path)

        # Plot
        baseline.df["date"] = pd.to_datetime(baseline.df["date"])
        rewards_df = baseline.df.loc[:, baseline.df.columns.drop(["date"])]
        rewards_df = (rewards_df + 1).cumprod()
        rewards_df.index = baseline.df["date"]

        sns.lineplot(data=rewards_df)
        plt.show()

    def t3(self):
        self.baseline.load_csv(self.program.args.baseline_path)

    def t4(self):
        program = Program()
        program.args.project_verbose = 1
        program.args.dataset_paths = [program.args.folder_dataset.joinpath("stockfadailydataset.csv")]

        # NOTE: for pypfopt baseline dataset_path must be defined
        dataset = StockFaDailyDataset(program=program, tickers=DOW_30_TICKER,
                                      split_coef=program.args.dataset_split_coef)
        dataset.load_csv(program.args.dataset_paths[0].as_posix())

        # NOTE: Tested thing:
        # dataset.df = dataset.df[dataset.df["date"] >= "2020-02-01"]
        dataset.df = dataset.df[(dataset.df["date"] >= "2020-03-10") & (dataset.df["date"] <= "2020-03-18")]
        dataset.df.index = dataset.df["date"].factorize()[0]

        #
        d_tics = dataset.df[["tic", "close", "date"]].sort_values(by=["tic", "date"])
        d = {"date": d_tics["date"].unique()}
        d.update({tic: d_tics[d_tics["tic"] == tic]["close"] for tic in d_tics["tic"].unique()})
        df = pd.DataFrame(d)

        #
        baseline = Baseline(program=program)
        baseline.pypfopt_returns(dataset=df, bounds=(0, 1))
        baseline.indexes_returns()
        baseline.df.reset_index(inplace=True, drop=True)

        #
        return baseline


def t():
    return TestBaseline().t4()


def main():
    program = Program()

    # NOTE: for pypfopt baseline dataset_path must be defined
    dataset = StockFaDailyDataset(program=program, tickers=DOW_30_TICKER, split_coef=program.args.dataset_split_coef)
    dataset.load_csv(program.args.dataset_paths[0].as_posix())

    # Prepare dataset for pypfopt computing
    d_tics = dataset.df[["tic", "close", "date"]].sort_values(by=["tic", "date"])
    d = {"date": d_tics["date"].unique()}
    d.update({tic: d_tics[d_tics["tic"] == tic]["close"] for tic in d_tics["tic"].unique()})
    df = pd.DataFrame(d)

    # Compute baselines
    baseline = Baseline(program=program)
    baseline.pypfopt_returns(dataset=df, bounds=(0, 1))
    baseline.indexes_returns()
    baseline.df.reset_index(inplace=True, drop=True)
    # TODO: add baseline Warren Buffet; NOTE: dataset_path needn't be defined
    baseline.save_csv(file_path=program.args.baseline_path.as_posix())


if __name__ == "__main__":
    main()
    # t()
