# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Literal

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback

from run.portfolio_allocation.envs.portfolioallocationenv import PortfolioAllocationEnv
from run.shared.memory import Memory


class CustomCheckpointCallback(CheckpointCallback):
    """
    A custom callback that saves a model every ``save_freq`` steps.
    """

    def __init__(
        self,
        #
        memory_name: str,
        #
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        self.memory_name = memory_name
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_training_end(self) -> None:
        for env in self.locals['env'].envs:
            if isinstance(env, PortfolioAllocationEnv):
                memory: Memory = self.locals['env'].envs[0]._memory
                memory_path = Path(self.save_path).joinpath(self.memory_name)
                memory.save_memory(memory_path)


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
            log_dict = {f"memory/{k}": v for k, v in memory_dict.items()}
            wandb.log(log_dict)
        return super()._on_step()

    def _on_training_end(self) -> None:
        if hasattr(self.locals['env'].envs[0].unwrapped, '_memory'):
            memory: Memory = getattr(self.locals['env'].envs[0].unwrapped, '_memory')
            wandb.run.summary["portfolio_return_sum"] = memory.df['portfolio_return'].sum()
            wandb.run.summary["portfolio_value_start"] = memory.df['portfolio_value'].iloc[0]
            wandb.run.summary["portfolio_value_end"] = memory.df['portfolio_value'].iloc[-1]
            wandb.run.summary["date_start"] = memory.df['date'].unique()[0]
            wandb.run.summary["date_end"] = memory.df['date'].unique()[-1]
        # if hasattr(self.locals['env'].envs[0].unwrapped, '_memory'):
        #     memory: Memory = getattr(self.locals['env'].envs[0].unwrapped, '_memory')
        #     wandb.run.summary["portfolio_return_sum"] = memory.df['portfolio_return'].sum()
        #     wandb.run.summary["portfolio_value_start"] = memory.df['portfolio_value'].iloc[0]
        #     wandb.run.summary["portfolio_value_end"] = memory.df['portfolio_value'].iloc[-1]
        #     wandb.run.summary["date_start"] = memory.df['date'].unique()[0]
        #     wandb.run.summary["date_end"] = memory.df['date'].unique()[-1]
        super()._on_training_end()


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True
