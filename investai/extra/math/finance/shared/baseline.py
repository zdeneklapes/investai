# -*- coding: utf-8 -*-
from copy import deepcopy  # noqa
from typing import Dict, Tuple

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
from run.shared.memory import Memory
from tqdm import trange


# TODO: add baseline S@P500
# TODO: add baseline DJI30

class Baseline(Memory):
    def __init__(self, program: Program, df: pd.DataFrame = pd.DataFrame()):
        super().__init__(program, df)

    def warren_buffet_returns(self) -> pd.DataFrame:
        """Source:"""

    def indexes_returns(self, program: Program, dataset: pd.DataFrame) -> pd.DataFrame:
        """Source:"""

    def _create_weights(self, mean, s, bounds: Tuple) -> Dict:
        ef = EfficientFrontier(mean, s, weight_bounds=bounds)
        ef.min_volatility()
        cleaned_weights_min_var = ef.clean_weights()
        ef = EfficientFrontier(mean, s, weight_bounds=bounds)
        ef.max_quadratic_utility()
        cleaned_weights_max_quadratic_utility = ef.clean_weights()
        ef = EfficientFrontier(mean, s, weight_bounds=bounds)
        ef.max_sharpe()
        cleaned_weights_max_sharpe = ef.clean_weights()
        return {
            "minimum_variance": cleaned_weights_min_var,
            "maximum_quadratic_utility": cleaned_weights_max_quadratic_utility,
            "maximum_sharpe": cleaned_weights_max_sharpe,
        }

    def pypfopt_returns(self, dataset: pd.DataFrame, bounds: Tuple = (0, 1)) -> pd.DataFrame:
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
            mean_annual_return = mean_historical_return(dataset.iloc[:i])
            s_covariance_matrix = CovarianceShrinkage(dataset.iloc[:i]).ledoit_wolf()

            #
            weights = self._create_weights(mean_annual_return, s_covariance_matrix, bounds)

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

        self.df = rewards
        return self.df


def t1():
    from dotenv import load_dotenv

    program = Program()
    program.args.project_verbose = 1
    program.args.dataset_path = program.args.folder_dataset.joinpath("stockfadailydataset.csv").as_posix()
    load_dotenv(dotenv_path=program.args.folder_root.joinpath(".env").as_posix())

    dataset = StockFaDailyDataset(program=program, tickers=DOW_30_TICKER,
                                  split_coef=program.args.dataset_split_coef)
    dataset.load_csv(program.args.dataset_path)

    d_tics = dataset.df[["tic", "close", "date"]].sort_values(by=["tic", "date"])
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

    # NOTE: for pypfopt baseline dataset_path must be defined
    dataset = StockFaDailyDataset(program=program, tickers=DOW_30_TICKER, split_coef=program.args.dataset_split_coef)
    dataset.load_csv(program.args.dataset_path)
    d_tics = dataset.df[["tic", "close", "date"]].sort_values(by=["tic", "date"])
    d = {"date": d_tics["date"].unique()}
    d.update({tic: d_tics[d_tics["tic"] == tic]["close"] for tic in d_tics["tic"].unique()})
    df = pd.DataFrame(d)
    baseline = Baseline(program=program)
    baseline.pypfopt_returns(dataset=df, bounds=(0, 1))
    # NOTE: for these baselines dataset_path needn't be defined
    # TODO: add baseline S@P500
    # TODO: add baseline DJI30
    # TODO: add baseline Warren Buffet
    baseline.save_csv(file_path=program.args.baseline_path.as_posix())


if __name__ == "__main__":
    main()
