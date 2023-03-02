# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, List, Final
import attrs
from pathlib import Path

import numpy as np
import pandas as pd

# FIXME: Stable-baselines3 requires gym.spaces not gymnasium.spaces
import gym
from gym import spaces
# import gymnasium as gym
# from gymnasium import spaces

from gymnasium.utils import seeding
from scipy.special import softmax
from stable_baselines3.common.vec_env import DummyVecEnv


@attrs.define
class Memory:
    """Memory class for storing the history of the agent performance in the environment"""
    df: pd.DataFrame

    def append(self, portfolio_value, portfolio_return, action, date):
        """Append memory
        :param portfolio_value: Portfolio value
        :param portfolio_return: Portfolio return
        :param action: Action
        :param date: Date
        """
        df_new = pd.DataFrame({
            "portfolio_value": [portfolio_value],
            "portfolio_return": [portfolio_return],
            "action": [action],
            "date": [date]
        })
        self.df = pd.concat([self.df, df_new], axis=0, ignore_index=True)

    @property
    def _initial_portfolio_value(self) -> int:
        """Initial portfolio value"""
        return self.df["portfolio_value"].iloc[0]

    @property
    def _current_portfolio_value(self) -> int:
        """Current portfolio value"""
        return self.df["portfolio_value"].iloc[-1]

    def save(self, save_path: Path):
        """Save memory to csv file
        :param save_path: Path to save the memory
        """
        self.df.to_json(save_path.as_posix(), index=True)

    def load(self, save_path: Path):
        """Save memory to csv file
        :param save_path: Path to save the memory
        """
        self.df.from_json(save_path.as_posix(), index=True)


class PortfolioAllocationEnv(gym.Env):
    """Portfolio Allocation Environment using OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_portfolio_value: int, tickers: List[str], features: List[str],
                 save_path: Path, start_from_index: int = 0):
        # Immutable
        self._save_path: Final = save_path
        self._df: Final = df
        self._start_from_index: Final = start_from_index
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
        self._data_index: Index of the current data
        self._portfolio_value: Portfolio value
        self._memory: Memory of the environment
        :param initial_portfolio_value:
        """
        self._data_index = self._start_from_index
        self._memory = Memory(df=pd.DataFrame(dict(portfolio_value=[initial_portfolio_value],
                                                   portfolio_return=[0],
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

    def _get_reward(self) -> float:
        """Calculate reward based on portfolio value and actions"""
        return (
            (self._memory.df['portfolio_value'].iloc[-1] - self._memory.df['portfolio_value'].iloc[-2])
            / self._memory.df['portfolio_value'].iloc[-2]
        )

    def _get_portfolio_return(self, weights) -> float:
        """Calculate portfolio return
        :param weights: Weights of the portfolio
        :return: Portfolio return
        """

        current_close = self._current_data["close"].values
        previous_close = self._df.loc[self._data_index - 1, "close"].values
        individual_return = current_close / previous_close - 1
        portfolio_return = (individual_return * weights).sum()
        return portfolio_return

    def step(self, action):
        # TODO: Why is softmax used here?
        normalized_actions = softmax(action)  # action are the tickers weight in the portfolio

        self._data_index += 1  # Go to next data (State & Observation Space)
        current_portfolio_value = (
            self._memory._current_portfolio_value
            * (1 + self._get_portfolio_return(normalized_actions))
        )

        # Memory
        self._memory.append(portfolio_value=current_portfolio_value,
                            portfolio_return=self._get_portfolio_return(normalized_actions),
                            action=normalized_actions,
                            date=self._current_data.date.unique()[0])

        # Observation, Reward, Terminated, Truncated, Info, Done
        return self._current_state, self._get_reward(), self._terminal, {}

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

    def get_stable_baseline3_environment(self):
        environment = DummyVecEnv([lambda: self])
        observation_space = environment.reset()
        return environment, observation_space
