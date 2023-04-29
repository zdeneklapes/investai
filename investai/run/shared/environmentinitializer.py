# -*- coding: utf-8 -*-
"""TODO docstring"""
from pathlib import Path
from typing import Union

import pandas as pd
from run.portfolio_allocation.thesis.dataset.stockfadailydataset import StockFaDailyDataset
from run.portfolio_allocation.envs.portfolioallocation2env import PortfolioAllocation2Env
from run.portfolio_allocation.envs.portfolioallocationenv import PortfolioAllocationEnv
from shared.program import Program
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ENVIRONMENT_TYPE = Union[PortfolioAllocationEnv, PortfolioAllocation2Env]


class EnvironmentInitializer:
    def __init__(self, program: Program, dataset: StockFaDailyDataset):
        self.program = program
        self.dataset = dataset

    def portfolio_allocation(self, dataset: pd.DataFrame) -> DummyVecEnv:
        env = None
        if self.program.args.env_id == "0":
            if self.program.args.project_verbose:
                self.program.log.info(f"Init environment: {PortfolioAllocationEnv.__name__}")
            env = PortfolioAllocationEnv(
                dataset=dataset,
                tickers=self.dataset.tickers,
                features=self.dataset.get_features(),
                start_index=self.program.args.start_index,
            )
        elif self.program.args.env_id == "1":
            if self.program.args.project_verbose:
                self.program.log.info(f"Init environment: {PortfolioAllocation2Env.__name__}")
            env = PortfolioAllocation2Env(
                program=self.program,
                dataset=dataset,
                tickers=self.dataset.tickers,
                columns_to_drop_in_observation=["date", "tic"],
                start_index=self.program.args.start_index
            )
        env = Monitor(
            env, Path(self.program.args.wandb_dir).as_posix(), allow_early_resets=True
        )  # stable_baselines3.common.monitor.Monitor
        env = DummyVecEnv([lambda: env])
        return env
