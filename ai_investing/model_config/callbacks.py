# -*- coding: utf-8 -*-
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from examples.portfolio_allocation_fa_dji30.PortfolioAllocationEnv import Memory, PortfolioAllocationEnv


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
                memory.save(memory_path)
