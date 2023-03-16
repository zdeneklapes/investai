# -*- coding: utf-8 -*-
import warnings
from pathlib import Path
from typing import Literal
import cProfile
import pstats

import numpy as np
import pandas as pd
import matplotlib


def now_time(_format: str = "%Y-%m-%dT%H-%M-%S") -> str:
    import datetime

    return datetime.datetime.now().strftime(_format)


def line_profiler_stats(func):
    def wrapper(*args, **kwargs):
        import line_profiler

        time_profiler = line_profiler.LineProfiler()
        try:
            return time_profiler(func)(*args, **kwargs)
        finally:
            time_profiler.print_stats()

    return wrapper


def profileit(func):
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        ps = pstats.Stats(prof).sort_stats("cumtime")
        ps.print_stats()
        return retval

    return wrapper


def cProfile_decorator(sort_by: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            try:
                return func(*args, **kwargs)
            finally:
                pr.disable()
                pr.print_stats(sort=sort_by)

        return wrapper

    return decorator


# This function reload the module
def reload_module(module):
    import importlib
    importlib.reload(module)


def config():
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO: zipline problem
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    matplotlib.use("Agg")


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param raw_data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def get_dataset(
    df, purpose: Literal["train", "test"], date_col_name: str = "date", split_constant: int = 0.75
) -> pd.DataFrame:
    split_date = df.iloc[int(df.index.size * split_constant)]["date"]
    if purpose == "train":
        return data_split(df, df[date_col_name].min(), split_date, date_col_name)  # df[df["date"] >= date_split]
    elif purpose == "test":
        return data_split(df, split_date, df[date_col_name].max(), date_col_name)  # df[df["date"] >= date_split]
    else:
        raise ValueError(f"Unknown purpose: {purpose}")


def get_start_end(df: pd.DataFrame) -> dict:
    start = df.dataset["date"].min()
    end = df.dataset["date"].max()
    return {"start": start, "end": end}


def ignore_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO: zipline problem
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def find_git_root(path):
    path = Path(path).resolve()
    if (path / '.git').is_dir():
        return path
    if path == path.parent:
        raise Exception('Not a Git repository')
    return find_git_root(path.parent)


def portfolio_value_from_returns(returns: pd.Series) -> pd.Series:
    """
    From returns calculate portfolio value
    :param returns:
    :return:
    """
    return (returns + 1).cumprod()


def calculate_return_from_weights(t_now: np.array, t_prev: np.array, weights: np.array) -> float:
    """
    Calculate portfolio rewards
    :param t_now: (pd.Series) current portfolio value
    :param t_prev: (pd.Series) previous portfolio value
    :param weights: (pd.Series) current portfolio weights
    :return: (pd.Series) portfolio rewards
    """
    current_balance_pct = (t_now / t_prev * weights).sum()
    return current_balance_pct - 1


def calculate_sharpe(returns: pd.Series):
    # TODO: check correctness
    if returns.std() != 0:
        sharpe = (252 ** 0.5) * returns.mean() / returns.std()
        return sharpe
    else:
        return 0
