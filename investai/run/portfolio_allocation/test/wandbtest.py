# -*- coding: utf-8 -*-
"""TODO docstring"""
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange
import pandas as pd
from matplotlib.axes import Axes
import matplotlib
import matplotlib.pyplot as plt  # noqa
import matplotlib.dates as md
import seaborn as sns  # noqa
import numpy as np  # noqa
from pprint import pprint  # noqa

from extra.math.finance.shared.baseline import Baseline
from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.shared.algorithmsb3 import ALGORITHM_SB3_TYPE
from run.shared.callback.wandb_util import wandb_summary
from run.shared.environmentinitializer import EnvironmentInitializer
from run.shared.memory import Memory
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program
from shared.utils import calculate_sharpe_ratio, reload_module  # noqa

matplotlib.use('agg')  # because otherwise is not allowed to run in a not main thread


class WandbTest:
    def __init__(self, program: Program, dataset: StockFaDailyDataset):
        self.program: Program = program
        self.dataset: StockFaDailyDataset = dataset

    def _deinit_environment(self, env):
        self.program.log.info("Deinit environment")
        env.close()

    def test(self, model: ALGORITHM_SB3_TYPE, deterministic=True) -> None:
        # Test
        environment: DummyVecEnv = EnvironmentInitializer(self.program, self.dataset).portfolio_allocation(
            self.dataset.test_dataset
        )
        env_unwrapped = environment.envs[0].env
        obs = environment.reset()

        # "-2" because we don't want to go till terminal state, because the environment will be reset
        iterable = (
            trange(len(env_unwrapped.dataset.index.unique()) - 2, desc="Test")
            if self.program.args.project_verbose
            else range(len(env_unwrapped.dataset.index.unique()) - 2)
        )

        for _ in iterable:
            action, _ = model.predict(obs, deterministic=deterministic)
            environment.step(action)
            if self.program.is_wandb_enabled():
                self.create_log(env_unwrapped.memory)

        env_unwrapped.memory.save_json(
            self.program.args.folder_out.joinpath("test_memory.json").as_posix())

        # Finish
        if self.program.is_wandb_enabled():
            self.create_summary(env_unwrapped.memory, env_unwrapped.dataset)
            self.create_baseline_chart(env_unwrapped.memory)

    def create_log(self, memory: Memory):
        log_dict = {"test/reward": memory.df.iloc[-1]["reward"]}
        wandb.log(log_dict)

    def create_summary(self, memory: Memory, dataset: pd.DataFrame):
        info = {
            # Rewards
            "test/total_reward": (memory.df["reward"] + 1).cumprod().iloc[-1],
            # TODO: reward annualized
            # Dates
            "test/dataset_start_date": dataset["date"].unique()[0],
            "test/dataset_end_date": dataset["date"].unique()[-1],
            "test/start_date": dataset["date"].unique()[0],
            "test/end_date": memory.df["date"].iloc[-1],
            # Ratios
            "test/sharpe_ratio": calculate_sharpe_ratio(memory.df["reward"]),
            # TODO: Calmar ratio
        }
        wandb_summary(info)

    def create_baseline_chart(self, memory_env: Memory):
        memory_env.df['date'] = pd.to_datetime(memory_env.df['date'], format='%Y-%m-%d')

        #
        baseline = Baseline(self.program)
        baseline.load_csv(self.program.args.baseline_path.as_posix())
        baseline.df['date'] = pd.to_datetime(baseline.df['date'], format='%Y-%m-%d')
        #
        memory_without_action = memory_env.df[memory_env.df.columns.difference(['action'])]
        df_chart = pd.merge(memory_without_action, baseline.df, on='date')
        df_cumprod = (df_chart.drop(columns=['date']) + 1).cumprod()
        df_cumprod.rename(columns={'reward': 'model'}, inplace=True)

        # W&B table
        portfolios_return_table = wandb.Table(data=df_cumprod, columns=df_cumprod.columns.values)
        wandb.log({"test/portfolios_return_table": portfolios_return_table})

        # Seaborne chart
        df_cumprod.index = df_chart['date']
        fig: plt.figure = fig_rewards(df_cumprod)
        wandb.log({"test/portfolios_return_table_chart": wandb.Image(fig)})


def fig_rewards(df: pd.DataFrame) -> plt.figure:
    """Source: https://stackoverflow.com/a/63219756/14471542"""
    #
    fig: plt.figure
    ax: Axes
    fig, ax = plt.subplots()
    #
    ax = sns.lineplot(data=df)
    ax.set_title("Portfolio allocation rewards")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative reward")

    # specify the position of the major ticks at the beginning of the week
    ax.xaxis.set_major_locator(md.YearLocator())
    # specify the format of the labels as 'year-month-day'
    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
    # (optional) rotate by 90° the labels in order to improve their spacing
    plt.setp(ax.xaxis.get_majorticklabels())
    # specify the position of the minor ticks at each day
    ax.xaxis.set_minor_locator(md.MonthLocator(bymonthday=1))

    #
    x_dates = df.index.strftime('%Y-%m-%d').sort_values().unique().tolist()
    ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')

    # set ticks length
    ax.tick_params(axis='x', which='major', length=10,
                   direction='out', width=2, colors='black', grid_color='b', grid_alpha=0.5)
    ax.tick_params(axis='x', which='minor', length=5,
                   direction='out', width=2, colors='black', grid_color='b', grid_alpha=0.5)

    plt.tight_layout()
    return fig


def t1():
    program = Program()
    ret_val = {}

    program.args.baseline_path = program.args.folder_baseline.joinpath("baseline_pypfopt.csv")
    program.args.memory_path = program.args.folder_memory.joinpath("test_memory.json")
    program.args.project_verbose = True

    #
    baseline = Baseline(program)
    baseline.load_csv(program.args.baseline_path.as_posix())
    baseline.df['date'] = baseline.df['date'].astype(np.datetime64)
    memory = Memory(program=program)
    memory.load_json(program.args.memory_path.as_posix())
    memory.df['date'] = memory.df['date'].astype(np.datetime64)

    #

    ret_val['p'] = program
    ret_val['b'] = baseline
    ret_val['m'] = memory
    ret_val['memory_without_action'] = memory.df[memory.df.columns.difference(['action'])]
    ret_val['df_chart'] = pd.merge(ret_val['memory_without_action'], baseline.df, on='date')
    ret_val['df_cumprod'] = (
        (ret_val['df_chart'][ret_val['df_chart'].columns.difference(['date'])] + 1).apply(lambda x: x.cumprod())
    )
    ret_val['df_cumprod']['date'] = ret_val['df_chart']['date']
    ret_val['df_cumprod'].rename(columns={'reward': 'model'}, inplace=True)
    ret_val['df_cumprod'].index = ret_val['df_cumprod']['date']
    ret_val['df_cumprod'].drop(columns=['date'], inplace=True)
    return ret_val


def main():
    program = Program()
    dataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
    dataset.load_csv(program.args.dataset_path)

    # TODO: get best model from wandb
    # get_best_model()


if __name__ == "__main__":
    main()
