# -*- coding: utf-8 -*-
import os
from typing import Literal

import pandas as pd
from agents.rllib_models import DRLAgent
from finrl.plot import backtest_plot, backtest_stats, get_baseline
from finrl.config import RESULTS_DIR, TEST_START_DATE, TEST_END_DATE, TRAINED_MODEL_DIR

from rl.envs.StockTradingEnv import StockTradingEnv
from common.utils import now_time


class Agent:
    def __init__(self, train_data, trade_data, model_type: Literal["ppo", "ddpg", "td3", "a2c", "sac"] = "ppo"):
        self.train_data = train_data
        self.trade_data = trade_data
        self.model_type: str = model_type
        self.tested = {
            "account_value": None,
            "actions": None,
        }
        self.trained_agent = None

    def get_env_params(self):
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

        stock_dimension = len(self.train_data.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(ratio_list) * stock_dimension
        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

        # Parameters for the environment
        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": ratio_list,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,  # TODO: 1_000_000 * 1e-4 = 100 (% of account value at the end)
        }

        return env_kwargs

    def get_agent(self) -> DRLAgent:
        # Establish the training environment using StockTradingEnv() class
        e_train_gym = StockTradingEnv(df=self.train_data, **self.get_env_params())

        # env_train, _ = e_train_gym.get_sb_env()

        # TODO: fill in the rest of the params
        env_config = {
            "price_array": [0],
            "tech_array": [0],
            "turbulence_array": [0],
        }

        agent = DRLAgent(env=e_train_gym, **env_config)
        return agent

    def train(self):
        agent = self.get_agent()
        A2C_PARAMS = {"n_steps": 1000, "ent_coef": 0.01, "learning_rate": 0.0007, "device": "cuda"}
        model, _ = agent.get_model(self.model_type)

        # set up logger
        # tmp_path = os.path.join(RESULTS_DIR, self.model_type)
        # new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # model.set_logger(new_logger)

        trained = agent.train_model(model=model, model_name=self.model_type, model_config=A2C_PARAMS)
        #
        self.trained_agent = trained

    def test(self):
        e_trade_gym = StockTradingEnv(df=self.trade_data, **self.get_env_params())

        # TODO: agent_path in calling DRLAgent.DRL_prediction()
        df_account_value, df_actions = DRLAgent.DRL_prediction(model=self.trained_agent, env=e_trade_gym)

        self.tested["account_value"] = df_account_value
        self.tested["actions"] = df_actions

    def backtest(self):
        perf_stats_all_sac = backtest_stats(account_value=self.tested["account_value"])
        perf_stats_all_sac = pd.DataFrame(perf_stats_all_sac)
        perf_stats_all_sac.to_csv("./" + RESULTS_DIR + "/perf_stats_all_sac_" + now_time() + ".csv")
        print("==============Get Baseline Stats===========")
        baseline_df = get_baseline(ticker="^DJI", start=TEST_START_DATE, end=TEST_END_DATE)

        _ = backtest_stats(baseline_df, value_col_name="close")
        backtest_plot(
            self.tested["account_value"],
            baseline_ticker="^DJI",
            baseline_start=TEST_START_DATE,
            baseline_end=TEST_END_DATE,
        )

    def save_trained_model(self) -> str:
        filename = os.path.join(TRAINED_MODEL_DIR, self.model_type, f"{now_time()}")
        if self.trained_agent is not None:
            self.trained_agent.save(filename)
        else:
            raise Exception(f"No trained agent to save (agent is: {self.trained_agent})")
        return filename

    def load_trained_model(self, filepath: str):
        """
        TODO
        """
        filepath_split = filepath.split(sep="/")
        for type_, model_ in {}:
            if type_ in filepath_split:
                return model_.load(filepath)


if __name__ == "__main__" and "__file__" in globals():
    # TODO: Show usage here!
    pass