# -*- coding: utf-8 -*-
from typing import Optional, Literal

import numpy as np
import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback

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
        for index, env in enumerate(self.locals['env'].envs):
            log_dict = {f"env/{index}/train/reward": self.locals["rewards"][0],
                        f"env/{index}/train/action": self.locals["actions"][0],
                        f"env/{index}/train/observation": self.locals["new_obs"][0],
                        f"env/{index}/train/date": self.locals["infos"][0]['date'], }
            wandb.log(log_dict)
        return super()._on_step()

    def _on_training_end(self) -> None:
        df: pd.DataFrame = getattr(self.locals['env'].envs[0].unwrapped, '_df')
        rewards: np.ndarray = self.locals["replay_buffer"].rewards[:self.num_timesteps]
        info = {
            # Rewards
            "train/total_reward": (rewards + 1).cumprod()[-1],
            # TODO: reward annualized
            # Dates
            "train/dataset_start_date": df['date'].unique()[0],
            "train/dataset_end_date": df['date'].unique()[-1],
            "train/start_date": df['date'].unique()[0],
            "train/end_date": self.locals['infos'][0]['date'],
            # Ratios
            "train/sharpe_ratio": calculate_sharpe_ratio(rewards),
            # TODO: Calmar ratio
        }
        wandb_summary(info)
        super()._on_training_end()
