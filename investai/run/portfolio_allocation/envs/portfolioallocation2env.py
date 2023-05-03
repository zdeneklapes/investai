# -*- coding: utf-8 -*-
from typing import Any, Dict, Final, List

# TODO: Stable-baselines3 requires gym.spaces not gymnasium.spaces (change it in future)
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gymnasium.utils import seeding
from run.shared.memory import Memory
from scipy.special import softmax
from shared.utils import calculate_return_from_weights
from shared.program import Program


# import gymnasium as gym
# from gymnasium import spaces


class PortfolioAllocation2Env(gym.Env):
    """Portfolio Allocation Environment using OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, program: Program, dataset: pd.DataFrame, tickers: List[str],
                 columns_to_drop_in_observation: List[str],
                 start_index: int = 0):
        # Immutable
        self.program: Final[Program] = program
        self.dataset: Final[pd.DataFrame] = dataset
        self._start_time: Final[int] = start_index
        self._tickers: Final[List[str]] = tickers  # Used for: Action and Observation space
        self._columns_to_drop_in_observation: Final[List[str]] = columns_to_drop_in_observation  # In Observation space

        # Inherited
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self._tickers) + 1,))  # tickers + 1 cash
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._tickers), len(self.dataset.columns.drop(self._columns_to_drop_in_observation))),
        )

        self.__reinit()

    def _memory_record(self, reward: float, action: np.ndarray, date: np.datetime64) -> Dict[str, Any]:
        ret_dict = {"reward": reward, "action": [action], "date": date}
        return ret_dict

    def __reinit(self):
        """Initialize environment
        self._data_index: Index of the current raw_data
        self._portfolio_value: Portfolio value
        """
        self._time = self._start_time
        self.memory = Memory(program=self.program,
                             df=pd.DataFrame(self._memory_record(reward=0,  # start reward, so +1 = 1
                                                                 action=np.array([0] * self.action_space.shape[0]),
                                                                 date=self._current_date)))

    @property
    def _current_date(self) -> np.datetime64:
        """self._data_index must be set correctly"""
        return self.dataset['date'].unique()[self._time]

    @property
    def _current_data_in_time(self) -> pd.DataFrame:
        """self._data_index must be set correctly"""
        return self.dataset.loc[self._time, :]

    @property
    def _current_observation(self) -> object:
        """self._data_index must be set correctly"""
        return self._current_data_in_time.drop(columns=self._columns_to_drop_in_observation)

    @property
    def _terminal(self) -> bool:
        """Check if the episode is terminated"""
        return self._time >= len(self.dataset.index.unique()) - 1

    def step(self, action: np.ndarray):
        self._time += 1  # Go to next raw_data (State & Observation Space)
        normalized_actions: np.ndarray = softmax(action)  # action are the tickers weight in the portfolio

        # action are the tickers weight in the portfolio (without cash)
        asset_allocation_actions = normalized_actions[1:]  # Remove cash
        reward = calculate_return_from_weights(self.dataset.loc[self._time, :]["close"].values,
                                               self.dataset.loc[self._time - 1, :]["close"].values,
                                               asset_allocation_actions, )

        #
        info = pd.DataFrame(self._memory_record(reward=reward,
                                                action=normalized_actions,
                                                date=self._current_date))
        self.memory.concat(info)

        # Observation, Reward, Terminated, Info
        return self._current_observation, reward, self._terminal, info.to_dict()

    def reset(
        self,
        # *,
        # seed: Optional[int] = None,
        # options: Optional[Dict[str, Any]] = None,
    ):
        self.__reinit()
        if self.program.args.train_verbose > 0:
            self.program.log.info(f"Reset environment in time: {self._time}:{self._current_date}")
        return self._current_observation  # First observation

    def render(self, mode="human"):
        return self._current_observation

    def seed(self, seed=None):
        seed = self._seed(seed)
        if "d" in self.program.args.project_verbose: self.program.log.debug(f"Seed: {seed}")
        return seed

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if "d" in self.program.args.project_verbose: self.program.log.debug(
            f"Seed: {seed}, np_random: {self.np_random}")
        return [seed]
