# -*- coding: utf-8 -*-
import sys
import typing

import gym
import stable_baselines3
import torch
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3  # noqa

sweep_configuration_types_per_algorithm = {
    "A2C": {
        "policy": typing.Union[str, typing.Type[stable_baselines3.common.policies.ActorCriticPolicy]],
        "env": typing.Union[gym.core.Env, stable_baselines3.common.vec_env.base_vec_env.VecEnv, str],
        "learning_rate": typing.Union[float, typing.Callable[[float], float]],
        "n_steps": int,
        "gamma": float,
        "gae_lambda": float,
        "ent_coef": float,
        "vf_coef": float,
        "max_grad_norm": float,
        "rms_prop_eps": float,
        "use_rms_prop": bool,
        "use_sde": bool,
        "sde_sample_freq": int,
        "normalize_advantage": bool,
        "tensorboard_log": typing.Optional[str],
        "policy_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "verbose": int,
        "seed": typing.Optional[int],
        "device": typing.Union[torch.device, str],
        "_init_setup_model": bool,
    },
    "PPO": {
        "policy": typing.Union[str, typing.Type[stable_baselines3.common.policies.ActorCriticPolicy]],
        "env": typing.Union[gym.core.Env, stable_baselines3.common.vec_env.base_vec_env.VecEnv, str],
        "learning_rate": typing.Union[float, typing.Callable[[float], float]],
        "n_steps": int,
        "batch_size": int,
        "n_epochs": int,
        "gamma": float,
        "gae_lambda": float,
        "clip_range": typing.Union[float, typing.Callable[[float], float]],
        "clip_range_vf": typing.Union[None, float, typing.Callable[[float], float]],
        "normalize_advantage": bool,
        "ent_coef": float,
        "vf_coef": float,
        "max_grad_norm": float,
        "use_sde": bool,
        "sde_sample_freq": int,
        "target_kl": typing.Optional[float],
        "tensorboard_log": typing.Optional[str],
        "policy_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "verbose": int,
        "seed": typing.Optional[int],
        "device": typing.Union[torch.device, str],
        "_init_setup_model": bool,
    },
    "SAC": {
        "policy": typing.Union[str, typing.Type[stable_baselines3.sac.policies.SACPolicy]],
        "env": typing.Union[gym.core.Env, stable_baselines3.common.vec_env.base_vec_env.VecEnv, str],
        "learning_rate": typing.Union[float, typing.Callable[[float], float]],
        "buffer_size": int,
        "learning_starts": int,
        "batch_size": int,
        "tau": float,
        "gamma": float,
        "train_freq": typing.Union[int, typing.Tuple[int, str]],
        "gradient_steps": int,
        "action_noise": typing.Optional[stable_baselines3.common.noise.ActionNoise],
        "replay_buffer_class": typing.Optional[typing.Type[stable_baselines3.common.buffers.ReplayBuffer]],
        "replay_buffer_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "optimize_memory_usage": bool,
        "ent_coef": typing.Union[str, float],
        "target_update_interval": int,
        "target_entropy": typing.Union[str, float],
        "use_sde": bool,
        "sde_sample_freq": int,
        "use_sde_at_warmup": bool,
        "tensorboard_log": typing.Optional[str],
        "policy_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "verbose": int,
        "seed": typing.Optional[int],
        "device": typing.Union[torch.device, str],
        "_init_setup_model": bool,
    },
    "TD3": {
        "policy": typing.Union[str, typing.Type[stable_baselines3.td3.policies.TD3Policy]],
        "env": typing.Union[gym.core.Env, stable_baselines3.common.vec_env.base_vec_env.VecEnv, str],
        "learning_rate": typing.Union[float, typing.Callable[[float], float]],
        "buffer_size": int,
        "learning_starts": int,
        "batch_size": int,
        "tau": float,
        "gamma": float,
        "train_freq": typing.Union[int, typing.Tuple[int, str]],
        "gradient_steps": int,
        "action_noise": typing.Optional[stable_baselines3.common.noise.ActionNoise],
        "replay_buffer_class": typing.Optional[typing.Type[stable_baselines3.common.buffers.ReplayBuffer]],
        "replay_buffer_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "optimize_memory_usage": bool,
        "policy_delay": int,
        "target_policy_noise": float,
        "target_noise_clip": float,
        "tensorboard_log": typing.Optional[str],
        "policy_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "verbose": int,
        "seed": typing.Optional[int],
        "device": typing.Union[torch.device, str],
        "_init_setup_model": bool,
    },
    "DQN": {
        "policy": typing.Union[str, typing.Type[stable_baselines3.dqn.policies.DQNPolicy]],
        "env": typing.Union[gym.core.Env, stable_baselines3.common.vec_env.base_vec_env.VecEnv, str],
        "learning_rate": typing.Union[float, typing.Callable[[float], float]],
        "buffer_size": int,
        "learning_starts": int,
        "batch_size": int,
        "tau": float,
        "gamma": float,
        "train_freq": typing.Union[int, typing.Tuple[int, str]],
        "gradient_steps": int,
        "replay_buffer_class": typing.Optional[typing.Type[stable_baselines3.common.buffers.ReplayBuffer]],
        "replay_buffer_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "optimize_memory_usage": bool,
        "target_update_interval": int,
        "exploration_fraction": float,
        "exploration_initial_eps": float,
        "exploration_final_eps": float,
        "max_grad_norm": float,
        "tensorboard_log": typing.Optional[str],
        "policy_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "verbose": int,
        "seed": typing.Optional[int],
        "device": typing.Union[torch.device, str],
        "_init_setup_model": bool,
    },
    "DDPG": {
        "policy": typing.Union[str, typing.Type[stable_baselines3.td3.policies.TD3Policy]],
        "env": typing.Union[gym.core.Env, stable_baselines3.common.vec_env.base_vec_env.VecEnv, str],
        "learning_rate": typing.Union[float, typing.Callable[[float], float]],
        "buffer_size": int,
        "learning_starts": int,
        "batch_size": int,
        "tau": float,
        "gamma": float,
        "train_freq": typing.Union[int, typing.Tuple[int, str]],
        "gradient_steps": int,
        "action_noise": typing.Optional[stable_baselines3.common.noise.ActionNoise],
        "replay_buffer_class": typing.Optional[typing.Type[stable_baselines3.common.buffers.ReplayBuffer]],
        "replay_buffer_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "optimize_memory_usage": bool,
        "tensorboard_log": typing.Optional[str],
        "policy_kwargs": typing.Optional[typing.Dict[str, typing.Any]],
        "verbose": int,
        "seed": typing.Optional[int],
        "device": typing.Union[torch.device, str],
        "_init_setup_model": bool,
    },
}

sweep_configuration = {
    "method": "random",
    "name": "sweep_1",
    "metric": {"goal": "minimize", "name": "train/loss"},
    "parameters": {
        # 'policy': {
        #     "values": ["MlpPolicy", "MlpLstmPolicy", "MlpLnLstmPolicy",
        #                "CnnPolicy", "CnnLstmPolicy", "CnnLnLstmPolicy", ],
        # },
        "policy": {"values": ["MlpPolicy"]},  # , "CnnPolicy", "MultiInputPolicy", ],
        # 'env': ,
        "learning_rate": {"min": 0.0001, "max": 0.01},
        "n_steps": {"values": [32, 64, 128, 256, 512, 1024, 2048]},
        "gamma": {"min": 0.9, "max": 0.999},
        "gae_lambda": {"min": 0.8, "max": 0.999},
        "ent_coef": {"min": 0.0001, "max": 0.01},
        "vf_coef": {"min": 0.0001, "max": 0.01},
        "max_grad_norm": {"min": 0.5, "max": 0.99},
        "rms_prop_eps": {"min": 0.0001, "max": 0.01},
        # 'use_rms_prop': bool,
        # 'use_sde': bool,
        "sde_sample_freq": {"min": 4, "max": 32},
        # 'normalize_advantage': bool,
        # 'tensorboard_log': Optional,
        # 'policy_kwargs': Optional,
        # 'verbose': int,
        # 'seed': Optional,
        # 'device': Union,
        # '_init_setup_model': bool,
        "batch_size": {
            "values": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        },
        "n_epochs": {"min": 1, "max": 10},
        "clip_range": {"min": 0.1, "max": 0.3},
        "clip_range_vf": {
            "values": [None, 0.05, 0.1, 0.15, 0.2],
        },
        "target_kl": {"min": 0.01, "max": 0.05},
        "buffer_size": {
            "values": [1000, 2000, 3000, 4000, 5000],
        },
        "learning_starts": {"min": 100, "max": 1000},
        "tau": {"min": 0.001, "max": 0.01},
        "train_freq": {"min": 1, "max": 4},
        "gradient_steps": {"min": 1, "max": 4},
        # 'action_noise': Optional,
        # 'replay_buffer_class': Optional,
        # 'replay_buffer_kwargs': Optional,
        # 'optimize_memory_usage': bool,
        "target_update_interval": {"min": 1, "max": 4},
        "target_entropy": {"min": 0.1, "max": 0.2},
        # 'use_sde_at_warmup': bool,
        "policy_delay": {"min": 1, "max": 4},
        "target_policy_noise": {"min": 0.1, "max": 0.2},
        "target_noise_clip": {"min": 0.1, "max": 0.2},
        "exploration_fraction": {"min": 0.1, "max": 0.2},
        "exploration_initial_eps": {"min": 0.1, "max": 0.2},
        "exploration_final_eps": {"min": 0.1, "max": 0.2},
    },
}


def get_sweep_configuration_types(print_types=False) -> dict:
    sweep_configuration_types = {}
    for k, v in sweep_configuration_types_per_algorithm.items():
        sweep_configuration_types.update(v)

    if print_types:
        print("{")
        for k, v in sweep_configuration_types.items():
            print(f"'{k}': {v.__name__},")
        print("}")
    return sweep_configuration_types


def print_foo():
    for k, v in sweep_configuration['parameters'].items():
        if 'values' in v:
            print(f"{v['values']}")
        elif 'min' in v:
            print(f"<{v['min']}, {v['max']}>")


if __name__ == "__main__":
    # get_sweep_configuration_types(print_types=True)
    print_foo()
    sys.exit(0)
