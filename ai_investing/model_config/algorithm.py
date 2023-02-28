# -*- coding: utf-8 -*-
"""
TODO: Better save checkpoint and description info.txt
"""
from typing import Union

#
import gym
import attrs
import numpy as np
import torch
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from project_configs.program import Program

MODEL = {
    "a2c": A2C,
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}

ALGORITHM_PARAMS = {  # noqa: F841 # pylint: disable=unused-variable
    "a2c": {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    },
}


@attrs.define
class Algorithm:
    program: Program
    algorithm: str
    env: gym.Env

    def get_action_noise(self, action_noise: str = "normal"):
        noise_map = {
            "normal": NormalActionNoise,
            "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise
        }
        n_actions = self.env.action_space.shape[-1]
        return noise_map[action_noise](mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    def get_model(
        self,
        model_name: str,
        policy="MlpPolicy",
        policy_kwargs=None,
        verbose=1,
        seed=None,
        device: Union[torch.device, "str"] = "cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=None,
    ) -> Union[A2C, DDPG, PPO, SAC, TD3]:
        #
        _model_kwargs = ALGORITHM_PARAMS[model_name.lower()]

        # Action noise
        if "action_noise" in _model_kwargs:
            _model_kwargs["action_noise"] = self.get_action_noise()

        #
        return MODEL[model_name.lower()](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            **_model_kwargs,
        )
