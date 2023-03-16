# -*- coding: utf-8 -*-
# TODO: action +1 more action from 30 actions increase to 31 actions, because Agent can also decide for cash
from typing import Any, Dict, Optional, List, Final

import numpy as np
import pandas as pd

# FIXME: Stable-baselines3 requires gym.spaces not gymnasium.spaces
import gym
from gym import spaces
# import gymnasium as gym
# from gymnasium import spaces

from gymnasium.utils import seeding
from scipy.special import softmax

from run.shared.memory import Memory
from shared.utils import get_return_from_weights


class PortfolioAllocationEnv(gym.Env):
    """Portfolio Allocation Environment using OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_portfolio_value: int, tickers: List[str], features: List[str],
                 start_data_from_index: int = 0):
        # Immutable
        self._df: Final = df
        self._start_from_index: Final = start_data_from_index
        self._tickers: Final = tickers  # Used for: Action and Observation space
        self._features: Final = features  # Used for Observation space

        # Sets: self._data_index, self._portfolio_value, self._memory
        self.__init_environment(initial_portfolio_value=initial_portfolio_value)

        # Inherited
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self._tickers),))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(len(self._tickers),
                                                   len(self._features))
                                            )

    def __init_environment(self, initial_portfolio_value: int):
        """Initialize environment
        self._data_index: Index of the current raw_data
        self._portfolio_value: Portfolio value
        self._memory: Memory of the environment
        :param initial_portfolio_value:
        """
        self._data_index = self._start_from_index
        self._memory = Memory(df=pd.DataFrame(dict(portfolio_value=[initial_portfolio_value],
                                                   reward=[0],
                                                   action=[[1 / len(self._tickers)] * len(self._tickers)],
                                                   date=[self._current_data.date.unique()[0]])))

    @property
    def _current_data(self) -> pd.DataFrame:
        """self._data_index must be set correctly"""
        return self._df.loc[self._data_index, :]

    @property
    def _current_state(self) -> object:
        """self._data_index must be set correctly"""
        return self._current_data.drop(columns=["date", "tic"])

    @property
    def _terminal(self) -> bool:
        """Check if the episode is terminated"""
        # TODO: portfolio_value <= 0
        return self._data_index >= len(self._df.index.unique()) - 1

    def step(self, action):
        # TODO: Why is softmax used here?
        self._data_index += 1  # Go to next raw_data (State & Observation Space)
        normalized_actions = softmax(action)  # action are the tickers weight in the portfolio
        reward = get_return_from_weights(self._df.iloc[self._data_index],
                                         self._df.iloc[self._data_index - 1],
                                         normalized_actions)
        current_portfolio_value = (
            self._memory.df["portfolio_value"].iloc[-1]  # previous portfolio value
            * (1 + reward)  # portfolio return
        )

        # Memory
        log_dict = {
            "portfolio_value": current_portfolio_value,
            "reward": reward,
            "action": normalized_actions,
            "date": self._current_data.date.unique()[0]
        }
        self._memory.append(**log_dict)

        # Observation, Reward, Terminated, Truncated, Info, Done
        return self._current_state, reward, self._terminal, log_dict

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.__init_environment(initial_portfolio_value=self._memory._initial_portfolio_value)
        return self._current_state  # First observation

    def render(self, mode='human'):
        return self._current_state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
