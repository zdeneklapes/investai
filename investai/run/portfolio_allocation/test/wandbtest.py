# -*- coding: utf-8 -*-
"""TODO docstring"""
import wandb
from extra.math.finance.shared.baseline import Baseline
from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.shared.algorithmsb3 import ALGORITHM_SB3_TYPE
from run.shared.callback.wandb_util import wandb_summary
from run.shared.environmentinitializer import ENVIRONMENT_TYPE, EnvironmentInitializer
from run.shared.memory import Memory
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program
from shared.utils import calculate_sharpe_ratio
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange


class WandbTest:
    def __init__(self, program: Program, dataset: StockFaDailyDataset):
        self.program: Program = program
        self.dataset: StockFaDailyDataset = dataset

    def _deinit_environment(self, env):
        self.program.log.info("Deinit environment")
        env.close()

    def test(self, model: ALGORITHM_SB3_TYPE, deterministic=True) -> None:
        environment: DummyVecEnv = EnvironmentInitializer(self.program, self.dataset).portfolio_allocation(
            self.dataset.test_dataset
        )
        obs = environment.reset()

        # "-2" because we don't want to go till terminal state, because the environment will be reset
        iterable = (
            trange(len(environment.envs[0].env.dataset.index.unique()) - 2, desc="Test")
            if self.program.args.project_verbose
            else range(len(environment.envs[0].env.dataset.index.unique()) - 2)
        )

        # Test
        for _ in iterable:
            action, _ = model.predict(obs, deterministic=deterministic)
            environment.step(action)
            if self.program.is_wandb_enabled():
                self.create_log(environment.envs[0].env.memory)

        if self.program.is_wandb_enabled():
            # Finish
            self.create_summary(environment.envs[0].env)
            self.create_baseline_chart(environment.envs[0].env)

    def create_log(self, memory: Memory):
        log_dict = {"memory/test_reward": memory.df.iloc[-1]["reward"]}
        wandb.log(log_dict)

    def create_summary(self, environment: ENVIRONMENT_TYPE):
        info = {
            # Rewards
            "test/total_reward": (environment.memory.df["reward"] + 1).cumprod().iloc[-1],
            # TODO: reward annualized
            # Dates
            "test/dataset_start_date": environment.dataset["date"].unique()[0],
            "test/dataset_end_date": environment.dataset["date"].unique()[-1],
            "test/start_date": environment.dataset["date"].unique()[0],
            "test/end_date": environment.memory.df["date"].iloc[-1],
            # Ratios
            "test/sharpe_ratio": calculate_sharpe_ratio(environment.memory.df["reward"]),
            # TODO: Calmar ratio
        }
        wandb_summary(info)

    def create_baseline_chart(self, environment: ENVIRONMENT_TYPE):
        baseline = Baseline(self.program, self.dataset.test_dataset)
        baseline.load(self.program.args.baseline_path.as_posix())


def get_best_model(self):
    pass


def main():
    from dotenv import load_dotenv

    program = Program()
    load_dotenv(dotenv_path=program.args.folder_root.as_posix())
    dataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    # TODO: get best model from wandb
    # get_best_model()


if __name__ == "__main__":
    main()
