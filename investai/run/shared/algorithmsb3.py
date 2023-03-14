# -*- coding: utf-8 -*-
from typing import Union
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN, DDPG

ALGORITHM_SB3_TYPE = Union[PPO, A2C, SAC, TD3, DQN, DDPG]

ALGORITHM_SB3 = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "dqn": DQN,
    "ddpg": DDPG,
}
