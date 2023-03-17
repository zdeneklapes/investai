# -*- coding: utf-8 -*-
from typing import Optional, Literal

import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback

from run.shared.memory import Memory
from shared.utils import calculate_sharpe_ratio
from run.shared.callback.wandb_util import wandb_summary


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
        if hasattr(self.locals['env'].envs[0].unwrapped, '_memory'):
            memory: Memory = getattr(self.locals['env'].envs[0].unwrapped, '_memory')
            memory_dict = memory.df.iloc[-1].to_dict()
            del memory_dict['action']
            del memory_dict['date']
            log_dict = {f"memory/train_{k}": v for k, v in memory_dict.items()}
            wandb.log(log_dict)
        return super()._on_step()

    def _on_training_end(self) -> None:
        if hasattr(self.locals['env'].envs[0].unwrapped, '_memory') \
            and hasattr(self.locals['env'].envs[0].unwrapped, '_df'):
            memory: Memory = getattr(self.locals['env'].envs[0].unwrapped, '_memory')
            df: pd.DataFrame = getattr(self.locals['env'].envs[0].unwrapped, '_df')
            info = {"train/total_reward": (memory.df['reward'] + 1.0).cumprod().iloc[-1],
                    # TODO: Add sharpe ratio
                    # TODO: reward annualized
                    "train/start_date": df['date'].unique()[0],
                    "train/end_date": df['date'].unique()[-1],
                    "train/sharpe_ratio": calculate_sharpe_ratio(memory.df['reward']), }
            wandb_summary(info)
        super()._on_training_end()
