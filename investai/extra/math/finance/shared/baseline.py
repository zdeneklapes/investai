# -*- coding: utf-8 -*-
# TODO: add baseline S@P500
# TODO: add baseline DJI30

from copy import deepcopy  # noqa
import numpy as np  # noqa
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from tqdm import trange

from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program

# For Debugging
from shared.utils import reload_module, get_return_from_weights  # noqa
from IPython.display import display  # noqa


class Baseline:
    def __init__(self, dataframe: pd.DataFrame, program, bounds: tuple = (0, 1)):
        self.program = program
        self.dataframe = dataframe
        self.dates = self.dataframe['date'].unique()
        del self.dataframe['date']
        self.bounds = bounds
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

    def get_returns(self) -> pd.DataFrame:
        """Source: https://github.com/AI4Finance-Foundation/FinRL-Tutorials"""
        if not self.returns.empty:
            return self.returns

        iterable = trange(4, self.dates.size - 1) \
            if self.program.args.project_verbose \
            else range(4, self.dates.size - 1)
        for i in iterable:
            mean_annual_return = mean_historical_return(self.dataframe.iloc[:i])
            s_covariance_matrix = CovarianceShrinkage(self.dataframe.iloc[:i]).ledoit_wolf()

            #
            weights = self.get_weights(mean_annual_return, s_covariance_matrix)

            assert list(self.dataframe.iloc[i].index) == list(weights['minimum_variance'].keys())

            #
            return_min_var = get_return_from_weights(self.dataframe.iloc[i].values,
                                                     self.dataframe.iloc[i - 1].values,
                                                     list(weights["minimum_variance"].values()))
            return_max_quadratic_utility = get_return_from_weights(self.dataframe.iloc[i].values,
                                                                   self.dataframe.iloc[i - 1].values,
                                                                   list(weights["maximum_quadratic_utility"].values()))
            return_max_sharpe = get_return_from_weights(self.dataframe.iloc[i].values,
                                                        self.dataframe.iloc[i - 1].values,
                                                        list(weights["maximum_sharpe"].values()))

            #
            self.returns = pd.concat([self.returns, pd.DataFrame({
                "date": self.dates[i],
                f"minimum_variance_{self.bounds[0]}_{self.bounds[1]}": return_min_var,
                f"maximum_quadratic_utility_{self.bounds[0]}_{self.bounds[1]}": return_max_quadratic_utility,
                f"maximum_sharpe_{self.bounds[0]}_{self.bounds[1]}": return_max_sharpe,
            }, index=[0])], ignore_index=True)

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
    program.args.dataset_path = program.project_structure.datasets.joinpath("stockfadailydataset.csv").as_posix()
    load_dotenv(dotenv_path=program.project_structure.root.joinpath(".env").as_posix())

    dataset = StockFaDailyDataset(program=program,
                                  tickers=DOW_30_TICKER,
                                  dataset_split_coef=program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    d_tics = dataset.dataset[['tic', 'close', 'date']].sort_values(by=['tic', 'date'])
    d = {"date": d_tics['date'].unique()}
    d.update({tic: d_tics[d_tics['tic'] == tic]['close'] for tic in d_tics['tic'].unique()})
    df = pd.DataFrame(d)
    baseline = Baseline(df, program, bounds=(0, 1))
    # baseline.get_returns()

    return {
        "dataset": dataset,
        "d_tics": d_tics,
        "df": df,
        "baseline": baseline,
    }


def main():
    from dotenv import load_dotenv

    program = Program()
    program.args.project_verbose = 1
    program.args.dataset_path = program.project_structure.datasets.joinpath("stockfadailydataset.csv").as_posix()
    load_dotenv(dotenv_path=program.project_structure.root.joinpath(".env").as_posix())

    #
    dataset = StockFaDailyDataset(program=program, tickers=DOW_30_TICKER,
                                  dataset_split_coef=program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    d_tics = dataset.dataset[['tic', 'close', 'date']].sort_values(by=['tic', 'date'])
    d = {"date": d_tics['date'].unique()}
    d.update({tic: d_tics[d_tics['tic'] == tic]['close'] for tic in d_tics['tic'].unique()})
    df = pd.DataFrame(d)

    #
    baseline = Baseline(df, program, bounds=(0, 1))
    baseline.get_returns()
    baseline.save(program.args.baseline_path
                  if program.args.baseline_path
                  else program.project_structure.baselines.joinpath("baseline.csv").as_posix())


if __name__ == '__main__':
    main()
