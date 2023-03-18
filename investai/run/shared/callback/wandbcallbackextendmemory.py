# -*- coding: utf-8 -*-
from typing import Optional, Literal

import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback

from shared.utils import calculate_sharpe_ratio
from run.shared.callback.wandb_util import wandb_summary
from run.shared.memory import Memory


class WandbCallbackExtendMemory(WandbCallback):
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
    ):
        return super().__init__(verbose, model_save_path, model_save_freq, gradient_save_freq, log)

    def _on_step(self) -> bool:
        for index, env in enumerate(self.locals['env'].envs):
            log_dict = {"memory/train_reward": env.unwrapped.memory.df.iloc[-1]['reward']}
            wandb.log(log_dict)
        return super()._on_step()

    def _on_training_end(self) -> None:
        dataset: pd.DataFrame = self.locals['env'].envs[0].unwrapped.dataset
        memory: Memory = self.locals['env'].envs[0].unwrapped.memory
        info = {
            # Rewards
            "train/total_reward": (memory.df['reward'] + 1).cumprod().iloc[-1],
            # TODO: reward annualized
            # Dates
            "train/dataset_start_date": dataset['date'].unique()[0],
            "train/dataset_end_date": dataset['date'].unique()[-1],
            "train/start_date": dataset['date'].unique()[0],
            "train/end_date": memory.df['date'].iloc[-1],
            # Ratios
            "train/sharpe_ratio": calculate_sharpe_ratio(memory.df['reward']),
            # TODO: Calmar ratio
        }
        wandb_summary(info)
        super()._on_training_end()
