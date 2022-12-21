# -*- coding: utf-8 -*-
import attr
from pathlib import Path
from typing import List, Union

#
import pandas as pd

#
from stable_baselines3 import A2C, DDPG, PPO, TD3

#
from configuration.settings import ProjectDir, ExperimentDir
from rl.plot.plot import backtest_plot, backtest_stats, get_baseline
from rl.experiments.common.utils import get_dataset


@attr.define
class Program:
    prj_dir: ProjectDir
    exp_dir: ExperimentDir
    dataset: pd.DataFrame = attr.field(init=False)
    DEBUG: bool = False

    def __attrs_post_init__(self):
        self.exp_dir.check_and_create_dirs()

    def init_dataset(self, dataset_name: str):
        self.dataset = get_dataset(
            pd.read_csv(self.exp_dir.out.datasets.joinpath(f"{dataset_name}.csv"), index_col=0), purpose="test"
        )


@attr.define
class Baseline:
    ticker: str
    start: str
    end: str

    def __post_init__(self):
        self.baseline_df = get_baseline(self.ticker, self.start, self.end)
        self.stats = backtest_stats(self.baseline_df, value_col_name="close")

        print(self.baseline_df.index.min())
        print(self.baseline_df.index.max())


@attr.define
class LearnedAlgorithm:
    algorithm: str
    filename: Path
    learned_algorithm: Union[A2C, PPO, DDPG, A2C, TD3]
    df_account_value = pd.DataFrame()
    df_actions = pd.DataFrame()
    perf_stats_all = pd.DataFrame()


class CompareAlgoBaseline:
    def __init__(self, algos: List[LearnedAlgorithm], baseline: Baseline):
        self.algos: List[LearnedAlgorithm] = algos
        self.baseline: Baseline = baseline

    def plot_baseline(self):
        backtest_plot(
            self.algos[0].df_account_value,
            baseline_ticker="SPY",
            baseline_start=self.baseline.start,
            baseline_end=self.baseline.end,
            value_col_name="account_value",
        )


@attr.define(kw_only=True)
class TestProgram(Program):
    algos: List[LearnedAlgorithm]
    metric: str
