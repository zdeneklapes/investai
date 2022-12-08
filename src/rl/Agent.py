# -*- coding: utf-8 -*-
import os
from typing import Literal

from rl.StockTradingEnv import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_plot, backtest_stats, get_baseline
from finrl.config import RESULTS_DIR, TEST_START_DATE, TEST_END_DATE, TRAINED_MODEL_DIR

from stable_baselines3.common.logger import configure
import pandas as pd

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

        env_train, _ = e_train_gym.get_sb_env()

        agent = DRLAgent(env=env_train)
        return agent

    def train(self):
        agent = self.get_agent()
        A2C_PARAMS = {"n_steps": 1000, "ent_coef": 0.01, "learning_rate": 0.0007, "device": "cpu"}
        model = agent.get_model(self.model_type, model_kwargs=A2C_PARAMS)

        # set up logger
        tmp_path = os.path.join(RESULTS_DIR, self.model_type)
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        trained = agent.train_model(model=model, tb_log_name=model, total_timesteps=30000)
        #
        self.trained_agent = trained

    def test(self):
        e_trade_gym = StockTradingEnv(df=self.trade_data, **self.get_env_params())
        df_account_value, df_actions = DRLAgent.DRL_prediction(model=self.trained_agent, environment=e_trade_gym)

        self.tested["account_value"] = df_account_value
        self.tested["actions"] = df_actions

    def backtest(self):
        perf_stats_all_sac = backtest_stats(account_value=self.tested["account_value"])
        perf_stats_all_sac = pd.DataFrame(perf_stats_all_sac)
        perf_stats_all_sac.to_csv("./" + RESULTS_DIR + "/perf_stats_all_sac_" + now_time() + ".csv")
        print("==============Get Baseline Stats===========")
        baseline_df = get_baseline(ticker="^DJI", start=TEST_START_DATE, end=TEST_END_DATE)

        stats = backtest_stats(baseline_df, value_col_name="close")
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
