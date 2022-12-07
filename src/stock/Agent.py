# -*- coding: utf-8 -*-
from stock.StockTradingEnv import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent


class Agent:
    def __init__(self, data):
        self.data = data

    def get_agent(self) -> DRLAgent:
        ratio_list = [
            "OPM",
            "NPM",
            "ROA",
            "ROE",
            "cur_ratio",
            "quick_ratio",
            "cash_ratio",
            "inv_turnover",
            "acc_rec_turnover",
            "acc_pay_turnover",
            "debt_ratio",
            "debt_to_equity",
            "PE",
            "PB",
            "Div_yield",
        ]

        stock_dimension = len(self.data.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(ratio_list) * stock_dimension
        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

        # Parameters for the environment
        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": ratio_list,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
        }

        # Establish the training environment using StockTradingEnv() class
        e_train_gym = StockTradingEnv(df=self.data, **env_kwargs)

        env_train, _ = e_train_gym.get_sb_env()

        agent = DRLAgent(env=env_train)
        return agent
