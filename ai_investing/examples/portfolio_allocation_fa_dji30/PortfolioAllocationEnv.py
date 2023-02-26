# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib
from scipy.special import softmax

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class PortfolioAllocationEnv(gym.Env):
    """Portfolio Allocation Environment using OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_portfolio_value, tickers, features, start_from_index=0):
        self._df = df
        self._ticker_dimension = tickers
        self._initial_amount = initial_portfolio_value

        # Spaces
        self._state_space = tickers
        self._feature_space = features

        # load data from a pandas dataframe
        self._start_from_index = start_from_index
        self._current_data = self._df.loc[self._start_from_index, :]
        self._current_state = np.append(np.array(self.covs),
                                        [self._current_data[tech].values.tolist() for tech in self._feature_space],
                                        axis=0)
        self._terminal = False
        self._portfolio_value = self._initial_amount

        # Inherited
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self._state_space),))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(len(self._feature_space),
                                                   len(self._state_space))
                                            )

        # Memory
        self._asset_memory = [self._initial_amount]  # memorize portfolio value each step
        self._portfolio_return_memory = [0]  # memorize portfolio return each step
        self._actions_memory = [[1 / self._ticker_dimension] * self._ticker_dimension]
        self._data_memory = [self._current_data.date.unique()[0]]

    def step(self, action):
        # print(self.day)
        self._terminal = self._start_from_index >= len(self._df.index.unique()) - 1
        # print(actions)

        if self._terminal:
            df = pd.DataFrame(self._portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(), 'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()

            plt.plot(self._portfolio_return_memory, 'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self._asset_memory[0]))
            print("end_total_asset:{}".format(self._portfolio_value))

            df_daily_return = pd.DataFrame(self._portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() != 0:
                sharpe = (252 ** 0.5) * df_daily_return['daily_return'].mean() / \
                         df_daily_return['daily_return'].std()
                print("Sharpe: ", sharpe)
            print("=================================")

            return self._current_state, self.reward, self._terminal, {}

        else:
            # Actions are the portfolio weight - normalize to sum of 1
            weights = softmax(action)
            self._actions_memory.append(weights)
            last_day_memory = self._current_data

            # load next state
            self._start_from_index += 1
            self._current_data = self._df.loc[self._start_from_index, :]
            # self.covs = self.data['cov_list'].values[0]
            self._current_state = np.append(np.array(self.covs),
                                            [self._current_data[tech].values.tolist() for tech in self._feature_space],
                                            axis=0)
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(((self._current_data.close.values / last_day_memory.close.values) - 1) * weights)
            # update portfolio value
            new_portfolio_value = self._portfolio_value * (1 + portfolio_return)
            self._portfolio_value = new_portfolio_value

            # save into memory
            self._portfolio_return_memory.append(portfolio_return)
            self._data_memory.append(self._current_data.date.unique()[0])
            self._asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            # print("Step reward: ", self.reward)
            # self.reward = self.reward*self.reward_scaling

        return self._current_state, self.reward, self._terminal, {}

    def _calculate_reward(self, weights: np.ndarray) -> float:
        """Calculate reward based on portfolio value and actions"""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self._asset_memory = [self._initial_amount]
        self._start_from_index = 0
        self._current_data = self._df.loc[self._start_from_index, :]
        # load states
        self.covs = self._current_data['cov_list'].values[0]
        self._current_state = np.append(np.array(self.covs),
                                        [self._current_data[tech].values.tolist() for tech in self._feature_space],
                                        axis=0)
        self._portfolio_value = self._initial_amount
        # self.cost = 0
        # self.trades = 0
        self._terminal = False
        self._portfolio_return_memory = [0]
        self._actions_memory = [[1 / self._ticker_dimension] * self._ticker_dimension]
        self._data_memory = [self._current_data.date.unique()[0]]
        return self._current_state

    def render(self, mode='human'):
        return self._current_state

    def save_asset_memory(self):
        date_list = self._data_memory
        portfolio_return = self._portfolio_return_memory
        df_account_value = pd.DataFrame({'date': date_list, 'daily_return': portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self._data_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self._actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self._current_data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
