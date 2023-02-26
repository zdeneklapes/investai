# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, List, Final
import attrs

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy.special import softmax
from stable_baselines3.common.vec_env import DummyVecEnv


@attrs.define
class Memory:
    portfolio_value: List
    portfolio_return: List
    action: List
    date: List

    def append_memory(self, _portfolio_value, _portfolio_return, _action, _date):
        """Append memory
        :param _portfolio_value: Portfolio value
        :param _portfolio_return: Portfolio return
        :param _action: Action
        :param _date: Date
        """
        self.portfolio_value.append(_portfolio_value)
        self.portfolio_return.append(_portfolio_return)
        self.action.append(_action)
        self.date.append(_date)


class PortfolioAllocationEnv(gym.Env):
    """Portfolio Allocation Environment using OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_portfolio_value, tickers, features, start_from_index=0):
        # Immutable
        self._df: Final = df
        self._initial_amount: Final = initial_portfolio_value
        self._tickers: Final = tickers  # Used for: Action and Observation space
        self._features: Final = features  # Used for Obser

        # Sets: self._data_index, self._portfolio_value, self._memory
        self.__init_environment(_data_index=start_from_index)

        # Inherited
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self._tickers),))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(len(self._features),
                                                   len(self._tickers))
                                            )

    def __init_environment(self, _data_index: int = 0):
        """Initialize environment
        self._data_index: Index of the current data
        self._portfolio_value: Portfolio value
        self._memory: Memory of the environment
        """
        self._data_index = _data_index
        self._portfolio_value = self._initial_amount
        self._memory = Memory(
            portfolio_value=[self._portfolio_value],
            portfolio_return=[0],
            action=[[1 / len(self._tickers)] * len(self._tickers)],
            date=[self._current_data.date.unique()[0]]
        )

    @property
    def _current_data(self):
        """self._data_index must be set correctly"""
        return self._df.loc[self._data_index, :]

    @property
    def _current_state(self):
        """self._data_index must be set correctly"""
        return self._current_data.values.tolist()

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

    def step(self, action):
        if self._is_terminal:
            df = pd.DataFrame(data={  # noqa # pylint: disable=unused-variable
                'date': self._memory.date,
                'portfolio_value': self._memory.portfolio_value,
                'portfolio_return': self._memory.portfolio_return,
                'action': self._memory.action
            })
            # TODO: Save df to a file
            # TODO: Stats
        else:
            weights = softmax(action)  # action are the tickers weight in the portfolio
            previous_data = self._current_data

            self._data_index += 1  # Go to next data (State & Observation Space)

            # Calculate portfolio return: individual stocks' return * weight
            portfolio_return = sum(
                (
                    (
                        self._current_data["close"].values
                        / previous_data["close"].values
                    ) - 1  # TODO: Why is here -1?
                ) * weights
            )
            # update portfolio value
            self._portfolio_value *= (1 + portfolio_return)

            # Memory
            self._memory.append_memory(self._portfolio_value,
                                       portfolio_return,
                                       weights,
                                       self._current_data.date.unique()[0])

        # Observation, Reward, Terminated, Truncated, Info, Done
        return self._current_state, self._reward, self._is_terminal, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.__init_environment(_data_index=0)
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
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
