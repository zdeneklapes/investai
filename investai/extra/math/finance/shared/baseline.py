# -*- coding: utf-8 -*-

from copy import deepcopy  # noqa
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program

# For Debugging
from shared.utils import reload_module  # noqa
from IPython.display import display  # noqa


class Baseline:
    def __init__(self, dataframe: pd.DataFrame, bounds: tuple = (0, 1)):
        self.dataframe = dataframe
        self.bounds = bounds
        self.returns = None

    def get_returns(self) -> pd.DataFrame:
        """Source: https://github.com/AI4Finance-Foundation/FinRL-Tutorials"""
        if self.returns is not None:
            return self.returns

        portfolio = pd.DataFrame(columns=[
            f"minimum_variance_{self.bounds[0]}_{self.bounds[1]}",
            f"maximum_quadratic_utility_{self.bounds[0]}_{self.bounds[1]}",
            f"maximum_sharpe_{self.bounds[0]}_{self.bounds[1]}",
        ], data=0.0)
        for date in self.dataframe.index.unique()[3:]:
            mean_annual_return = mean_historical_return(self.dataframe[self.dataframe.index <= date])
            s_covariance_matrix = CovarianceShrinkage(self.dataframe[self.dataframe.index <= date]).ledoit_wolf()

            #
            ef = EfficientFrontier(mean_annual_return, s_covariance_matrix, weight_bounds=self.bounds)
            ef.min_volatility()
            cleaned_weights_min_var = ef.clean_weights()
            ef = EfficientFrontier(mean_annual_return, s_covariance_matrix, weight_bounds=self.bounds)
            ef.max_quadratic_utility()
            cleaned_weights_max_quadratic_utility = ef.clean_weights()
            ef = EfficientFrontier(mean_annual_return, s_covariance_matrix, weight_bounds=self.bounds)
            ef.max_sharpe()
            cleaned_weights_max_sharpe = ef.clean_weights()

            return_min_var = (self.dataframe.loc[date].values
                              - self.dataframe.loc[self.dataframe.index < date].iloc[-1].values
                              ) * list(cleaned_weights_min_var.values())
            return_max_quadratic_utility = (self.dataframe.loc[date].values
                                            - self.dataframe.loc[self.dataframe.index < date].iloc[-1].values
                                            ) * list(cleaned_weights_max_quadratic_utility.values())
            return_max_sharpe = (self.dataframe.loc[date].values
                                 - self.dataframe.loc[self.dataframe.index < date].iloc[-1].values
                                 ) * list(cleaned_weights_max_sharpe.values())

            #
            portfolio.loc[date] = {
                f"minimum_variance_{self.bounds[0]}_{self.bounds[1]}": sum(return_min_var),
                f"maximum_quadratic_utility_{self.bounds[0]}_{self.bounds[1]}": sum(return_max_quadratic_utility),
                f"maximum_sharpe_{self.bounds[0]}_{self.bounds[1]}": sum(return_max_sharpe),
            }
        self.returns = portfolio
        return portfolio


def portfolio_value_from_returns(returns: pd.Series) -> pd.Series:
    pass


def t1():
    from dotenv import load_dotenv

    program = Program()
    program.args.dataset_path = program.project_structure.datasets.joinpath("stockfadailydataset.csv").as_posix()
    load_dotenv(dotenv_path=program.project_structure.root.joinpath(".env").as_posix())

    dataset = StockFaDailyDataset(program=program,
                                  tickers=DOW_30_TICKER,
                                  dataset_split_coef=program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    d_tics = dataset.dataset[['tic', 'close', 'date']].sort_values(by=['tic', 'date'])
    index_date = d_tics['date'].unique()
    df = pd.DataFrame({tic: d_tics[d_tics['tic'] == tic]['close'] for tic in d_tics['tic'].unique()})
    df.index = pd.to_datetime(index_date, format="%Y-%m-%d")
    baseline = Baseline(df, bounds=(0, 1))

    return {
        "dataset": dataset,
        "d_tics": d_tics,
        "index_date": index_date,
        "df": df,
        "baseline": baseline,
    }


if __name__ == '__main__':
    t1()
