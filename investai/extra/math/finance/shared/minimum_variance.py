# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
from copy import deepcopy  # noqa

from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program

# For Debugging
from shared.utils import reload_module  # noqa
from IPython.display import display  # noqa


def minimum_variance(df: pd.DataFrame):
    """Source: https://github.com/AI4Finance-Foundation/FinRL-Tutorials"""
    # calculate_portfolio_minimum_variance
    unique_dates = df['date'].unique()
    for i in range(unique_dates.size - 1):
        df_temp = df[df.date == unique_dates[i]].reset_index(drop=True)
        df_temp_next = df[df.date == unique_dates[i + 1]].reset_index(drop=True)
        # Sigma = risk_models.sample_cov(df_temp.return_list[0])
        # calculate covariance matrix
        cov_matrix = df_temp.return_list[0].cov()
        # portfolio allocation
        ef_min_var = EfficientFrontier(None, cov_matrix, weight_bounds=(0, 0.1))
        # minimum variance
        raw_weights_min_var = ef_min_var.min_volatility()
        # get weights
        cleaned_weights_min_var = ef_min_var.clean_weights()

        # current capital
        cap = df.iloc[0, i]
        # current cash invested for each stock
        current_cash = [element * cap for element in list(cleaned_weights_min_var.values())]
        # current held shares
        current_shares = list(np.array(current_cash)
                              / np.array(df_temp.close))
        # next time period price
        next_price = np.array(df_temp_next.close)
        ##next_price * current share to calculate next total account value
        df.iloc[0, i + 1] = np.dot(current_shares, next_price)

    portfolio = df.T
    portfolio.columns = ['account_value']


def t1():
    from dotenv import load_dotenv

    program = Program()
    program.args.dataset_path=program.project_structure.datasets.joinpath("stockfadailydataset.csv").as_posix()
    load_dotenv(dotenv_path=program.project_structure.root.joinpath(".env").as_posix())

    dataset = StockFaDailyDataset(program=program, tickers=DOW_30_TICKER,
                                  dataset_split_coef=program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)
    return {
        "dataset": dataset,
    }


if __name__ == '__main__':
    t1()
