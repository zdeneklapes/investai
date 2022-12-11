# -*- coding: utf-8 -*-

# ##############################################################################
# # Stock trading with fundamentals
# ##############################################################################

# ##############################################################################
# * This notebook is based on the tutorial:
# https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530


# ##############################################################################
# TODO
# ##############################################################################
# TODO: Add argument parser


# ##############################################################################
# INCLUDES
# ##############################################################################

##
import os
import sys
import warnings

import matplotlib
from finrl import config as finrl_config

# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.main import check_and_make_directories
from finrl.config import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRAINED_MODEL_DIR,
)
from finrl.config_tickers import DOW_30_TICKER

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from common.Args import Args, argument_parser  # noqa: E402
from data.DataPreprocessing import DataPreprocessing, DataTechnicalAnalysis  # noqas: E402
from rl.customagents.RayAgent import RayAgent  # noqa: E402
from rl.customagents.Agent import Agent  # noqa: E402
from rl.gym_envs.StockPortfolioAllocationEnv import StockPortfolioAllocationEnv  # noqa: E402
from common.utils import now_time  # noqa: E402


##
# ##############################################################################
# FUNCTIONS
# ##############################################################################
def config():
    #
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO: zipline problem
    warnings.filterwarnings("ignore", category=FutureWarning)  # TODO: ?
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # TODO: ?

    #
    matplotlib.use("Agg")

    #
    check_and_make_directories(
        [
            finrl_config.DATA_SAVE_DIR,
            finrl_config.TRAINED_MODEL_DIR,
            finrl_config.TENSORBOARD_LOG_DIR,
            finrl_config.RESULTS_DIR,
        ]
    )


def run_stable_baseline(args: Args):
    if args.train:
        ##
        data = DataPreprocessing(TRAIN_START_DATE, TRAIN_END_DATE, ticker_list=DOW_30_TICKER)
        data.retrieve_data(args)
        train_data = data_split(data.data_preprocessed, TRAIN_START_DATE, TRAIN_END_DATE)
        trade_data = data_split(data.data_preprocessed, TEST_START_DATE, TEST_END_DATE)
        ##
        agent = Agent(train_data, trade_data)
        agent.train()
        agent.save_trained_model()
        if args.test:
            agent.test()


def run_ray_rllib(args: Args):
    if not args.train:
        return

    ##
    data = DataTechnicalAnalysis(TRAIN_START_DATE, TRAIN_END_DATE, ticker_list=DOW_30_TICKER)
    data.retrieve_data(args)
    train_data = data_split(data.data_preprocessed, TRAIN_START_DATE, TRAIN_END_DATE)
    # trade = data_split(df, '2020-01-01', config.END_DATE)
    ##
    stock_dimension = len(train_data.tic.unique())
    state_space = stock_dimension
    tech_indicator_list = ["macd", "rsi_30", "cci_30", "dx_30"]
    # feature_dimension = len(tech_indicator_list)
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-1,
    }
    env = StockPortfolioAllocationEnv(df=train_data, **env_kwargs)
    agent = RayAgent(env=env)
    model_name = "ppo"

    #
    model, model_config = agent.get_model(model_name=model_name)
    # configuration = PPOConfig()

    #
    trained_model = agent.train_model(
        model=model,
        model_name=model_name,
        model_config=model_config,
        total_episodes=100,
    )

    filename = os.path.join(TRAINED_MODEL_DIR, model_name, f"{now_time()}")
    trained_model.save(filename)

    if args.test:
        # trade_data = data_split(data.data_preprocessed, TEST_START_DATE, TEST_END_DATE)
        raise NotImplementedError


##
# ##############################################################################
# # Main
# ##############################################################################
if __name__ == "__main__" and "__file__" in globals():
    config()
    args: Args = argument_parser()

    if args.ray:
        run_ray_rllib(args)

    if args.stable_baseline:
        run_stable_baseline(args)
