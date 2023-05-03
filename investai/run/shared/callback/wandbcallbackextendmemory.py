# -*- coding: utf-8 -*-
from typing import Literal, Optional

import pandas as pd
import wandb
from run.shared.callback.wandb_util import wandb_summary
from run.shared.memory import Memory
from shared.utils import calculate_sharpe_ratio
from shared.program import Program
from wandb.integration.sb3 import WandbCallback


class WandbCallbackExtendMemory(WandbCallback):
    """Custom WandbCallback that logs more information than the original one."""

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
        program: Program = None,
    ):
        self.program: Program = program
        return super().__init__(verbose, model_save_path, model_save_freq, gradient_save_freq, log)

    def _on_step(self) -> bool:
        for index, env in enumerate(self.locals["env"].envs):
            env.unwrapped.memory.df: pd.DataFrame
            if not env.unwrapped.memory.df.empty:  # Because of reset() env
                log_dict = {
                    "train/reward": env.unwrapped.memory.df.iloc[-1]["reward"],
                    "date": env.unwrapped.memory.df.iloc[-1]["date"]
                }
                wandb.log(log_dict)
        return super()._on_step()

    def _on_training_end(self) -> None:
        environment_portfolio_allocation = self.locals["env"].envs[0].unwrapped
        dataset: pd.DataFrame = environment_portfolio_allocation.dataset
        memory: Memory = environment_portfolio_allocation.memory
        info = {
            # Rewards
            "train/total_reward": (memory.df["reward"] + 1).cumprod().iloc[-1],
            # TODO: reward annualized
            # Dates
            "train/start_date": dataset["date"].unique()[0],
            "train/end_date": dataset["date"].unique()[-1],
            # Ratios
            "train/sharpe_ratio": calculate_sharpe_ratio(memory.df["reward"]),
            # TODO: Calmar ratio
        }
        wandb_summary(info)
        # environment_portfolio_allocation.memory.save_json(
        #     self.program.args.folder_out.joinpath("train_memory.json").as_posix())
        super()._on_training_end()
