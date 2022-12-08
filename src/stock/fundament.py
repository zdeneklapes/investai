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
import typing as tp
import warnings

import matplotlib
import pandas as pd
import stable_baselines3 as sb3
from finrl import config as finrl_config

# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.plot import backtest_plot, backtest_stats, get_baseline
from stable_baselines3.common.logger import configure, Logger
from finrl.config import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)
from finrl.config_tickers import (
    DOW_30_TICKER, CSI_300_TICKER, NAS_100_TICKER, CAC_40_TICKER, DAX_30_TICKER, HSI_50_TICKER
)

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from common.Args import Args, argument_parser  # noqa: E402
from common.utils import now_time
from config.settings import PROJECT_STUFF_DIR, AI_FINANCE_DIR  # noqa: E402
from stock.StockTradingEnv import StockTradingEnv  # noqa: E402
from stock.hyperparameter import HyperParameter  # noqas: E402
from stock.Ratio import Ratio  # noqa: E402
from stock.Data import Data
from stock.Agent import Agent


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


##
# ##############################################################################
# # Train/Test
# ##############################################################################

# ##############################################################################
# ## Save/Load Helpers


# ##############################################################################
# # Main
# ##############################################################################
if __name__ == "__main__" and "__file__" in globals():
    config()
    args: Args = argument_parser()

    if args.train:
        ##
        data = Data(
            TRAIN_START_DATE,
            TRAIN_END_DATE,
            ticker_list=DOW_30_TICKER,
            preprocessed_data=os.path.join(AI_FINANCE_DIR, "dataset_fundament_20221027-01h18.csv")
        )
        # data.get_preprocessed_data()
        # data.save_preprocessed_data()
        data.load_data()
        ##
        train_data = data_split(data.data_preprocessed, TRAIN_START_DATE, TRAIN_END_DATE)
        trade_data = data_split(data.data_preprocessed, TEST_START_DATE, TEST_END_DATE)
        agent = Agent(train_data, trade_data)
        agent.train('a2c')
        agent.test()

##

