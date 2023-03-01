# -*- coding: utf-8 -*-
"""
TODO: Better save checkpoint and description info.txt
"""
from typing import Union, Final

#
import gym
import attrs
import numpy as np
import torch
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from project_configs.program import Program

SB3_MODEL_TYPE = Union[A2C, DDPG, PPO, SAC, TD3]

MODEL: Final = {
    "a2c": A2C,
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}

STABLE_BASELINE_PARAMETERS: Final = {
    "a2c": {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.0007,
        # "batch_size": 128,
    },
    "ddpg": {
        "batch_size": 128,
        "buffer_size": 50000,
        "learning_rate": 0.001
    },
    "ppo": {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
    },
    "td3": {
        "batch_size": 100,
        "buffer_size": 1000000,
        "learning_rate": 0.001,
    },
    "sac": {
        "batch_size": 64,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
    "erl": {
        "learning_rate": 3e-5,
        "batch_size": 2048,
        "gamma": 0.985,
        "seed": 312,
        "net_dimension": 512,
        "target_step": 5000,
        "eval_gap": 30,
    }

}

RAY_PARAMETERS = {
    # TODO
}

ELEGANT_RL_PARAMETERS = {
    # TODO
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
        policy="MlpPolicy",
        policy_kwargs=None,
        verbose=1,
        seed=None,
        device: Union[torch.device, "str"] = "cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=None,
    ) -> SB3_MODEL_TYPE:
        #
        _model_kwargs = STABLE_BASELINE_PARAMETERS[self.algorithm.lower()]

        # Action noise
        if "action_noise" in _model_kwargs:
            _model_kwargs["action_noise"] = self.get_action_noise()

        #
        return MODEL[self.algorithm.lower()](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            **_model_kwargs,
        )
