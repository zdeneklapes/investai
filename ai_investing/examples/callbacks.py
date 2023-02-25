# -*- coding: utf-8 -*-
#
from typing import Union
from pathlib import Path

#
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """
    A custom callback that saves a model every ``save_freq`` steps.
    :param save_freq:
    :param save_path:
    """

    def __init__(self, save_freq: int, save_path: Union[Path]):
        super(CheckpointCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        if not self.save_path.parent.exists():  # check directory exists
            raise FileNotFoundError(f"Directory not found: {self.save_path.parent.as_posix()}")
        else:
            print(f"Learned algorithm will be saved to: {self.save_path.as_posix()}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path.as_posix() + f"_{self.n_calls}")
        return True

    def _on_training_end(self) -> None:
        self.model.save(self.save_path.as_posix() + f"_{self.n_calls}")
