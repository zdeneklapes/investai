# -*- coding: utf-8 -*-
from typing import Any, Dict, Final, List, Optional

# FIXME: Stable-baselines3 requires gym.spaces not gymnasium.spaces
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gymnasium.utils import seeding
from run.shared.memory import Memory
from scipy.special import softmax
from shared.utils import calculate_return_from_weights

# import gymnasium as gym
# from gymnasium import spaces


class PortfolioAllocationEnv(gym.Env):
    """Portfolio Allocation Environment using OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, dataset: pd.DataFrame, tickers: List[str], features: List[str], start_index: int = 0):
        # Immutable
        self._df: Final = dataset
        self._start_from_index: Final = start_index
        self._tickers: Final = tickers  # Used for: Action and Observation space
        self._features: Final = features  # Used for Observation space

        # Sets: self._data_index, self._portfolio_value, self._memory
        self.__init_environment()

        # Inherited
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self._tickers),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._tickers), len(self._features)))

    def __init_environment(self):
        """Initialize environment
        self._data_index: Index of the current raw_data
        self._portfolio_value: Portfolio value
        self._memory: Memory of the environment
        :param initial_portfolio_value:
        """
        self._time = self._start_from_index
        self._memory = Memory(
            df=pd.DataFrame(
                dict(
                    reward=[0],
                    action=[[1 / len(self._tickers)] * len(self._tickers)],
                    date=[self._current_data["date"].unique()[0]],
                )
            )
        )

    @property
    def _current_data(self) -> pd.DataFrame:
        """self._data_index must be set correctly"""
        return self._df.loc[self._time, :]

    @property
    def _current_state(self) -> object:
        """self._data_index must be set correctly"""
        return self._current_data.drop(columns=["date", "tic"])

    @property
    def _terminal(self) -> bool:
        """Check if the episode is terminated"""
        # TODO: portfolio_value <= 0
        return self._time >= len(self._df.index.unique()) - 1

    def step(self, action):
        # TODO: Why is softmax used here?
        self._time += 1  # Go to next raw_data (State & Observation Space)
        normalized_actions = softmax(action)  # action are the tickers weight in the portfolio
        reward = calculate_return_from_weights(
            self._df.loc[self._time, :]["close"].values,
            self._df.loc[self._time - 1, :]["close"].values,
            np.array(normalized_actions),
        )
        # Memory
        info = pd.DataFrame(
            {"reward": reward, "action": normalized_actions, "date": self._current_data["date"].unique()[0]}
        )
        self._memory.concat(**info)

        # Observation, Reward, Terminated, Truncated, Info, Done
        return self._current_state, reward, self._terminal, info.to_dict()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.__init_environment()
        return self._current_state  # First observation

    def render(self, mode="human"):
        return self._current_state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
