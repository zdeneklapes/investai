# -*- coding: utf-8 -*-
# TODO Dataset: TA
# TODO Dataset: TA + FA
# TODO: action +1 more action from 30 actions increase to 31 actions, because Agent can also decide for cash
# TODO: make it more reusable, so change self.features to self.not_include_in_observation_columns
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


class PortfolioAllocation2Env(gym.Env):
    """Portfolio Allocation Environment using OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, dataset: pd.DataFrame, tickers: List[str], columns_to_drop_in_observation: List[str], start_index: int = 0
    ):
        # Immutable
        self.dataset: Final = dataset
        self._start_time: Final = start_index
        self._tickers: Final = tickers  # Used for: Action and Observation space
        self._columns_to_drop_in_observation: Final = columns_to_drop_in_observation  # Used for Observation space

        self.__init_environment()

        # Inherited
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self._tickers) + 1,))  # tickers + 1 cash
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._tickers), len(self.dataset.columns.drop(self._columns_to_drop_in_observation))),
        )

    def __init_environment(self):
        """Initialize environment
        self._data_index: Index of the current raw_data
        self._portfolio_value: Portfolio value
        """
        self._time = self._start_time
        self.memory = Memory(df=pd.DataFrame())

    @property
    def _current_data(self) -> pd.DataFrame:
        """self._data_index must be set correctly"""
        return self.dataset.loc[self._time, :]

    @property
    def _current_state(self) -> object:
        """self._data_index must be set correctly"""
        return self._current_data.drop(columns=self._columns_to_drop_in_observation)

    @property
    def _terminal(self) -> bool:
        """Check if the episode is terminated"""
        # TODO: portfolio_value <= 0
        return self._time >= len(self.dataset.index.unique()) - 1

    def step(self, action):
        # TODO: Why is softmax used here?
        self._time += 1  # Go to next raw_data (State & Observation Space)
        normalized_actions = softmax(action)  # action are the tickers weight in the portfolio

        # action are the tickers weight in the portfolio (without cash)
        asset_allocation_actions = normalized_actions[1:]  # Remove cash
        reward = calculate_return_from_weights(
            self.dataset.loc[self._time, :]["close"].values,
            self.dataset.loc[self._time - 1, :]["close"].values,
            np.array(asset_allocation_actions),
        )

        #
        info = pd.DataFrame({"reward": reward, "action": action, "date": self._current_data["date"].unique()[0]})
        self.memory.concat(info)

        # Observation, Reward, Terminated, Info
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
