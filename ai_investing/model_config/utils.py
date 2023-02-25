# -*- coding: utf-8 -*-
#
import sys

#
sys.path.append("./ai_investing/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

#
import os
import warnings
from pathlib import Path
from typing import List, Literal

#
import pandas as pd

#

#
from project_configs.project_dir import ProjectDir
from project_configs.experiment_dir import ExperimentDir
# from rl.envs.StockTradingEnv import StockTradingEnv
from project_configs.program import Program
from model_config.learned_algorithm import LearnedAlgorithm


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


# ######################################################################################################################
# Helpers
# ######################################################################################################################
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


# def get_learned_algos(program: TestProgram) -> List[LearnedAlgorithm]:
#     def get_algorithm(filename: Path):
#         if "a2c" in filename.as_posix():
#             return LearnedAlgorithm(algorithm="a2c", filename=filename, learned_algorithm=A2C.load(filename))
#
#     return [get_algorithm(filepath) for filepath in program.exp_dir.out.models.glob("*")]


# ######################################################################################################################
# Functions
# ######################################################################################################################
def ignore_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO: zipline problem
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def init_program(dataset_name: str = None) -> Program:
    program = Program(
        project_dir=ProjectDir(root=Path("/Users/zlapik/my-drive-zlapik/0-todo/ai-investing")),
        exp_dir=ExperimentDir(Path(os.getcwd())),
        DEBUG=False,
    )
    if dataset_name:
        program.dataset = get_dataset(
            pd.read_csv(program.exp_dir.out.datasets.joinpath(f"{dataset_name}.csv"), index_col=0), purpose="test"
        )
    program.exp_dir.create_dirs()
    return program


# def predicate(program: TestProgram, algos: List[LearnedAlgorithm]) -> List[LearnedAlgorithm]:
#     for algo in algos:
#         env_kwargs = get_env_kwargs(program.dataset)
#         env_gym = StockTradingEnv(df=program.dataset, **env_kwargs)
#         algo.df_account_value, algo.df_actions = CustomDRLAgent.DRL_prediction(
#             model=algo.learned_algorithm, environment=env_gym
#         )
#     return algos


# def calc_performance_statistics(program: TestProgram):
#     for algo in program.algos:
#         perf_stats_all = backtest_stats(account_value=algo.df_account_value, value_col_name="account_value")
#         algo.perf_stats_all = pd.DataFrame(perf_stats_all)
#
#     sorted_list = sorted(program.algos, key=lambda x: x.perf_stats_all.loc[program.metric][0])
#     return sorted_list



    # if type == "data_detailed":
    #     df = pd.DataFrame()
    #     return pd.concat([df, tickers_data[tic].data_detailed for tic in tickers_data], ignore_index=True)
