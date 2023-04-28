# -*- coding: utf-8 -*-
"""TODO docstring"""
import wandb
import inspect
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa
import numpy as np  # noqa
from pprint import pprint  # noqa
from pathlib import Path  # noqa

from extra.math.finance.shared.baseline import Baseline
from run.portfolio_allocation.thesis.dataset.stockfadailydataset import StockFaDailyDataset
from run.shared.sb3.algorithms import ALGORITHM_SB3_STR2CLASS
from run.shared.callback.wandb_util import wandb_summary
from run.shared.environmentinitializer import EnvironmentInitializer
from run.shared.memory import Memory
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program
from shared.utils import calculate_sharpe_ratio
from shared.reload import reload_module  # noqa

matplotlib.use('agg')  # because otherwise is not allowed to run in a not main thread


class Test:
    def __init__(self, program: Program, dataset_path: Path):
        self.program: Program = program
        self.dataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
        self.dataset.load_csv(file_path=dataset_path.as_posix())
        self.dataset_path = dataset_path

    @property
    def _baseline(self) -> Baseline:
        baseline = Baseline(self.program)
        baseline.load_csv(self.program.args.baseline_path.as_posix())
        # baseline.df['date'] = pd.to_datetime(baseline.df['date'], format='%Y-%m-%d')
        return baseline

    def _deinit_environment(self, env):
        self.program.log.info("Deinit environment")
        env.close()

    def test(self, model_path: str, algorithm: str, deterministic=True) -> Memory:
        if self.program.args.project_verbose:
            self.program.log.info(f"START {inspect.currentframe().f_code.co_name}")
            self.program.log.info(f"Loading model_path: {model_path}, algorithm: {algorithm}")
        model = ALGORITHM_SB3_STR2CLASS[algorithm].load(model_path)
        # Environment
        environment: DummyVecEnv = EnvironmentInitializer(self.program, self.dataset).portfolio_allocation(
            self.dataset.test_dataset
        )
        env_unwrapped = environment.envs[0].env

        # Iteration: TODO: "-2" because we don't want to go till terminal state, because the environment will be reset

        memory = Memory(self.program)
        for i in range(self.program.args.test):
            obs = environment.reset()
            iterable = (
                trange(len(env_unwrapped.dataset.index.unique()) - 2, desc="Test")
                if self.program.args.project_verbose
                else range(len(env_unwrapped.dataset.index.unique()) - 2)
            )
            for _ in iterable:
                action, _ = model.predict(obs, deterministic=deterministic)
                environment.step(action)
                if self.program.is_wandb_enabled():
                    log_dict = {f"test/reward_{i}/model": env_unwrapped.memory.df.iloc[-1]["reward"], }
                    if i == 0: log_dict["date"] = env_unwrapped.memory.df.iloc[-1]["date"]
                    wandb.log(log_dict)

            if memory.df.empty:
                memory.df = env_unwrapped.memory.df[['date', 'reward']]
                memory.df.rename(columns={'reward': f'reward_{i}'}, inplace=True)
            else:
                memory.df = memory.df.merge(env_unwrapped.memory.df[['date', 'reward']], how='outer', on='date')
                memory.df.rename(columns={'reward': f'reward_{i}'}, inplace=True)

        if self.program.is_wandb_enabled():
            self.create_summary(env_unwrapped.memory, env_unwrapped.dataset)

        if self.program.args.project_verbose:
            self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")
        return memory

    def create_summary(self, memory: Memory, dataset: pd.DataFrame):
        info = {
            # Rewards
            "test/total_reward": (memory.df["reward"] + 1).cumprod().iloc[-1],
            # TODO: reward annualized
            # Dates
            "test/dataset_start_date": str(dataset["date"].unique()[0]),
            "test/dataset_end_date": str(dataset["date"].unique()[-1]),
            "test/start_date": str(dataset["date"].unique()[0]),
            "test/end_date": memory.df["date"].iloc[-1],
            # Ratios
            "test/sharpe_ratio": calculate_sharpe_ratio(memory.df["reward"]),
            # TODO: Calmar ratio
        }
        wandb_summary(info)

    def create_baseline_chart(self, memory_env: Memory):
        memory_env.df['date'] = pd.to_datetime(memory_env.df['date'], format='%Y-%m-%d')

        #
        baseline = self._baseline
        baseline.df['date'] = pd.to_datetime(baseline.df['date'], format="%Y-%m-%d")

        #
        memory_without_action = memory_env.df[memory_env.df.columns.difference(['action'])]
        df_chart = pd.merge(memory_without_action, baseline.df, on='date')
        df_cumprod = (df_chart.drop(columns=['date']) + 1).cumprod()
        df_cumprod.rename(columns={'reward': 'model'}, inplace=True)

        # W&B table
        # portfolios_return_table = wandb.Table(data=df_cumprod, columns=df_cumprod.columns.values)
        # wandb.log({"test/portfolios_return_table": portfolios_return_table})

        # Seaborne chart
        df_cumprod.index = df_chart['date']
        fig: plt.figure = sns.lineplot(data=df_cumprod)
        wandb.log({"test/portfolios_return_table_chart": wandb.Image(fig)})
        fig.cla()


class TestWandbTest:
    def __init__(self):
        self.program = Program()
        self.program.args.baseline_path = self.program.args.folder_baseline.joinpath("baseline_pypfopt.csv")
        self.program.args.memory_path = self.program.args.folder_memory.joinpath("test_memory.json")
        self.program.args.project_verbose = True

    def t1(self):
        ret_val = {}

        #
        baseline = Baseline(self.program)
        baseline.load_csv(self.program.args.baseline_path.as_posix())
        baseline.df['date'] = baseline.df['date'].astype(np.datetime64)
        memory = Memory(program=self.program)
        memory.load_json(self.program.args.memory_path.as_posix())
        memory.df['date'] = memory.df['date'].astype(np.datetime64)

        #

        ret_val['p'] = self.program
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
    dataset.load_csv(program.args.dataset_paths[0].as_posix())

    # TODO: get best model from wandb
    # get_best_model()


if __name__ == "__main__":
    main()
