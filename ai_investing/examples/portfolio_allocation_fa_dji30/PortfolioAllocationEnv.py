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

    def __init__(
        self,
        df,
        stock_dim,
        initial_amount,
        state_space,
        action_space,
        feature_list,
        day=0,
    ):
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount

        # Spaces
        self.state_space = state_space
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))
        self.feature_list = feature_list
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_space + len(self.feature_list), self.state_space))

        # load data from a pandas dataframe
        self.day = day
        self.data = self.df.loc[self.day, :]
        self.state = np.append(np.array(self.covs),
                               [self.data[tech].values.tolist() for tech in self.feature_list], axis=0)
        self.terminal = False
        self.portfolio_value = self.initial_amount

        # Memory
        self.asset_memory = [self.initial_amount]  # memorize portfolio value each step
        self.portfolio_return_memory = [0]  # memorize portfolio return each step
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, action):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(), 'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()

            plt.plot(self.portfolio_return_memory, 'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() != 0:
                sharpe = (252 ** 0.5) * df_daily_return['daily_return'].mean() / \
                         df_daily_return['daily_return'].std()
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            # Actions are the portfolio weight - normalize to sum of 1
            weights = softmax(action)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # self.covs = self.data['cov_list'].values[0]
            self.state = np.append(np.array(self.covs),
                                   [self.data[tech].values.tolist() for tech in self.feature_list], axis=0)
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            # print("Step reward: ", self.reward)
            # self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def _calculate_reward(self, weights: np.ndarray) -> float:
        """Calculate reward based on portfolio value and actions"""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state = np.append(np.array(self.covs),
                               [self.data[tech].values.tolist() for tech in self.feature_list], axis=0)
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode='human'):
        return self.state

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date': date_list, 'daily_return': portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
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
