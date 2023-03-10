# -*- coding: utf-8 -*-
import dataclasses
from pathlib import Path
from typing import Union

import pandas as pd
from stable_baselines3 import A2C, PPO, DDPG, TD3


@dataclasses.dataclass
class LearnedAlgorithm:
    algorithm: str
    filename: Path
    learned_algorithm: Union[A2C, PPO, DDPG, A2C, TD3]
    df_account_value = pd.DataFrame()
    df_actions = pd.DataFrame()
    perf_stats_all = pd.DataFrame()
