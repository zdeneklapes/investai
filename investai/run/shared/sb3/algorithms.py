# -*- coding: utf-8 -*-
from typing import Union

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

ALGORITHM_SB3_CLASS = Union[PPO, A2C, SAC, TD3, DQN, DDPG]

ALGORITHM_SB3_STR2CLASS = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "dqn": DQN,
    "ddpg": DDPG,
}
