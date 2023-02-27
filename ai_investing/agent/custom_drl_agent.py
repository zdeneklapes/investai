# -*- coding: utf-8 -*-
"""
TODO: Better save checkpoint and description info.txt
"""
from typing import Union

#
import numpy as np
import torch
from agents.stablebaselines3_models import DRLAgent, TensorboardCallback
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from model_config.callbacks import CheckpointCallback
from project_configs.program import Program


class CustomDRLAgent(DRLAgent):
    def __init__(self, env, program: Program, algorithm: str):
        super().__init__(env)
        self.program: Program = program
        self.algorithm: str = algorithm

    def train_model(
        self, model, tb_log_name="run", total_timesteps=1000000, checkpoint_freq: int = 10000, **kwargs
    ) -> Union[A2C, DDPG, PPO, SAC, TD3]:
        callback_list = CallbackList(
            [
                TensorboardCallback(),
                ProgressBarCallback(),
                CheckpointCallback(frequency=checkpoint_freq,
                                   save_path=self.program.experiment_dir.try_number.joinpath(f"{self.algorithm}")),
            ]
        )
        trained_model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=callback_list,
            **kwargs

        )
        return trained_model

    def get_model(
        self,
        model_name: str,
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

        model = {
            "a2c": A2C,
            "ddpg": DDPG,
            "ppo": PPO,
            "sac": SAC,
            "td3": TD3,
        }
        #
        return model[model_name.lower()](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            **model_kwargs,
        )
