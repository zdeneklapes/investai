# -*- coding: utf-8 -*-
import os
import warnings
from pathlib import Path
from typing import List, Literal
import cProfile
import pstats

import pandas as pd
import matplotlib

from shared.dir.project_dir import ProjectDir
from shared.dir.experiment_dir import ExperimentDir
from shared.program import Program
from run.shared.learned_algorithm import LearnedAlgorithm


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
    :param data: (df) pandas dataframe, start, end
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


def print_learned_algo(algos: List[LearnedAlgorithm], metric_row: str):
    for i in algos:
        try:
            print(i.perf_stats_all.loc[metric_row][0])
        except Exception as e:
            print(f"{e}")


def print_sorted_list_stats(algos: List[LearnedAlgorithm], metric_row: str):
    best_idx = len(algos) - 1
    worst_idk = 0

    def get_annual_return(algo: LearnedAlgorithm):
        return algo.perf_stats_all.loc[metric_row][0]

    print(f"{get_annual_return(algos[best_idx])}:   {algos[best_idx].filename.name}")
    print(f"{get_annual_return(algos[worst_idk])}:  {algos[worst_idk].filename.name}")


def get_start_end(df: pd.DataFrame) -> dict:
    start = df.dataset["date"].min()
    end = df.dataset["date"].max()
    return {"start": start, "end": end}


def ignore_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO: zipline problem
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def init_program(dataset_name: str = None) -> Program:
    program = Program(
        project_dir=ProjectDir(__file__),
        exp_dir=ExperimentDir(Path(os.getcwd())),
        DEBUG=False,
    )
    if dataset_name:
        program.dataset = get_dataset(
            pd.read_csv(program.exp_dir.out.datasets.joinpath(f"{dataset_name}.csv"), index_col=0), purpose="test"
        )
    program.exp_dir.create_dirs()
    return program
