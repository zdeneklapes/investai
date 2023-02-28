# -*- coding: utf-8 -*-
from stable_baselines3.common.callbacks import CheckpointCallback


class CustomCheckpointCallback(CheckpointCallback):
    """
    A custom callback that saves a model every ``save_freq`` steps.
    """

    def _on_training_end(self) -> None:
        self.model.save(self._checkpoint_path())
