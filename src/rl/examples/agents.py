# -*- coding: utf-8 -*-
from typing import Union

#
import numpy as np
import torch
from finrl.agents.stablebaselines3.models import DRLAgent, TensorboardCallback
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from rl.experiments.common.callbacks import CheckpointCallback
from rl.experiments.common.classes import Program
from common.utils import now_time


class CustomDRLAgent(DRLAgent):
    def __init__(self, env, program: Program):
        super().__init__(env)
        self.program: Program = program
        self.algorithm_name: str = "a2c"

    def train_model(
        self, model, tb_log_name="run", total_timesteps=1000000, checkpoint_freq: int = 10000, **kwargs
    ) -> Union[A2C, DDPG, PPO, SAC, TD3]:
        callback_list = CallbackList(
            [
                TensorboardCallback(),
                ProgressBarCallback(),
                CheckpointCallback(
                    checkpoint_freq, self.program.exp_dir.out.models.joinpath(f"{self.algorithm_name}_{now_time()}")
                ),
            ]
        )
        learned_algo = model.learn(
            total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=callback_list, **kwargs
        )
        return learned_algo

    def get_model(
        self,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        device: Union[torch.device, "str"] = "cpu",
        tensorboard_log=None,
    ):
        #
        if "action_noise" in model_kwargs:
            NOISE = {"normal": NormalActionNoise, "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise}
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )

        #
        return A2C(
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            **model_kwargs,
        )
