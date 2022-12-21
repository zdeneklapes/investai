# -*- coding: utf-8 -*-
# ######################################################################################################################
# Imports
# ######################################################################################################################
#
import sys

#
sys.path.append("./src/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

#
import os
import warnings
from pathlib import Path
from typing import List

#
import pandas as pd

#
from stable_baselines3 import A2C

#
from configuration.settings import ProjectDir, ExperimentDir
from rl.plot.plot import backtest_stats
from rl.envs.StockTradingEnv import StockTradingEnv
from rl.experiments._1_same_bigger_data_fundamental.train import (
    CustomDRLAgent,
    get_dataset,
    get_env_kwargs,
    dataset_name,
    base_cols,
    data_cols,
    ratios_cols,
)
from rl.experiments.common.classes import Program, LearnedAlgorithm


# ######################################################################################################################
# Helpers
# ######################################################################################################################
def print_cols():
    print(base_cols)
    print(data_cols)
    print(ratios_cols)


def print_learned_algo_len(algos: List[LearnedAlgorithm]):
    print(len(algos))


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


def get_learned_algos(program: Program) -> List[LearnedAlgorithm]:
    def get_algorithm(filename: Path):
        if "a2c" in filename.as_posix():
            return LearnedAlgorithm(algorithm="a2c", filename=filename, learned_algorithm=A2C.load(filename))

    return [get_algorithm(filepath) for filepath in program.exp_dir.out.algorithms.glob("*")]


# ######################################################################################################################
# Functions
# ######################################################################################################################
def ignore_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO: zipline problem
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def init_program() -> Program:
    program = Program(
        prj_dir=ProjectDir(root=Path("/Users/zlapik/my-drive-zlapik/0-todo/ai-investing")),
        exp_dir=ExperimentDir(Path(os.getcwd())),
        DEBUG=False,
    )
    program.dataset = get_dataset(
        pd.read_csv(program.exp_dir.out.datasets.joinpath(f"{dataset_name}.csv"), index_col=0), purpose="test"
    )
    program.exp_dir.check_and_create_dirs()
    return program


def predicate(program: Program) -> List[LearnedAlgorithm]:
    algos = get_learned_algos(program)
    for algo in algos:
        env_kwargs = get_env_kwargs(program.dataset)
        env_gym = StockTradingEnv(df=program.dataset, **env_kwargs)
        algo.df_account_value, algo.df_actions = CustomDRLAgent.DRL_prediction(
            model=algo.learned_algorithm, environment=env_gym
        )
    return algos


def calc_performance_statistics(algos: List[LearnedAlgorithm], program: Program):
    for algo in algos:
        perf_stats_all = backtest_stats(account_value=algo.df_account_value, value_col_name="account_value")
        algo.perf_stats_all = pd.DataFrame(perf_stats_all)

    sorted_list = sorted(algos, key=lambda x: x.perf_stats_all.loc[program.metric_name][0])
    return sorted_list
