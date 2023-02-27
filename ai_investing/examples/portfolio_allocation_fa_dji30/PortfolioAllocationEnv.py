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
    portfolio_value: List
    portfolio_return: List
    action: List
    date: List

    def append_memory(self, portfolio_value, portfolio_return, action, date):
        """Append memory
        :param portfolio_value: Portfolio value
        :param portfolio_return: Portfolio return
        :param action: Action
        :param date: Date
        """
        self.portfolio_value.append(portfolio_value)
        self.portfolio_return.append(portfolio_return)
        self.action.append(action)
        self.date.append(date)


class PortfolioAllocationEnv(gym.Env):
    """Portfolio Allocation Environment using OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_portfolio_value: int, tickers: List[str], features: List[str],
                 save_path: Path, start_from_index: int = 0):
        # Immutable
        self._save_path: Final = save_path
        self._df: Final = df
        self._tickers: Final = tickers  # Used for: Action and Observation space
        self._features: Final = features  # Used for Observation space

        # Sets: self._data_index, self._portfolio_value, self._memory
        self.__init_environment(initial_portfolio_value=initial_portfolio_value, data_index=start_from_index)

        # Inherited
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self._tickers),))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(len(self._tickers),
                                                   len(self._features))
                                            )

    def __init_environment(self, initial_portfolio_value: int, data_index: int):
        """Initialize environment
        self._data_index: Index of the current data
        self._portfolio_value: Portfolio value
        self._memory: Memory of the environment
        :param initial_portfolio_value:
        """
        self._data_index = data_index
        self._memory = Memory(
            portfolio_value=[initial_portfolio_value],
            portfolio_return=[0],
            action=[[1 / len(self._tickers)] * len(self._tickers)],
            date=[self._current_data.date.unique()[0]]
        )

    @property
    def _current_data(self) -> pd.DataFrame:
        """self._data_index must be set correctly"""
        return self._df.loc[self._data_index, :]

    @property
    def _current_state(self) -> object:
        """self._data_index must be set correctly"""
        return self._current_data.drop(columns=["date", "tic"])

    @property
    def _reward(self) -> float:
        """Calculate reward based on portfolio value and actions"""
        return (
            (self._memory.portfolio_value[-1] - self._memory.portfolio_value[-2])
            / self._memory.portfolio_value[-2]
        )

    @property
    def _is_terminal(self) -> bool:
        # TODO: portfolio_value <= 0
        return self._data_index >= len(self._df.index.unique()) - 1

    @property
    def _initial_portfolio_value(self) -> int:
        return self._memory.portfolio_value[0]

    @property
    def _last_portfolio_value(self) -> int:
        return self._memory.portfolio_value[-1]

    def get_portfolio_return(self, weights) -> float:
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
        if self._is_terminal:
            df = pd.DataFrame(data={  # noqa # pylint: disable=unused-variable
                'date': self._memory.date,
                'portfolio_value': self._memory.portfolio_value,
                'portfolio_return': self._memory.portfolio_return,
                'action': self._memory.action
            })
            df.to_csv(self._save_path.joinpath("result.csv"), index=True)
        else:
            # TODO: Why is softmax used here?
            weights = softmax(action)  # action are the tickers weight in the portfolio

            self._data_index += 1  # Go to next data (State & Observation Space)
            current_portfolio_value = self._last_portfolio_value * (1 + self.get_portfolio_return(weights))

            # Memory
            self._memory.append_memory(portfolio_value=current_portfolio_value,
                                       portfolio_return=self.get_portfolio_return(weights),
                                       action=weights,
                                       date=self._current_data.date.unique()[0])

            print(self._last_portfolio_value)
            # Observation, Reward, Terminated, Truncated, Info, Done
        return self._current_state, self._reward, self._is_terminal, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.__init_environment(initial_portfolio_value=self._initial_portfolio_value, data_index=0)
        return self._current_state

    def render(self, mode='human'):
        return self._current_state

    # def save_asset_memory(self):
    #     date_list = self._date_memory
    #     portfolio_return = self._portfolio_return_memory
    #     df_account_value = pd.DataFrame({'date': date_list, 'daily_return': portfolio_return})
    #     return df_account_value
    #
    # def save_action_memory(self):
    #     # date and close price length must match actions length
    #     date_list = self._date_memory
    #     df_date = pd.DataFrame(date_list)
    #     df_date.columns = ['date']
    #
    #     action_list = self._action_memory
    #     df_actions = pd.DataFrame(action_list)
    #     df_actions.columns = self._current_data.tic.values
    #     df_actions.index = df_date.date
    #     # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
    #     return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_stable_baseline3_environment(self):
        environment = DummyVecEnv([lambda: self])
        observation_space = environment.reset()
        return environment, observation_space
