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
)
from finrl.config_tickers import DOW_30_TICKER

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from common.Args import Args, argument_parser  # noqa: E402
from data.Data import Data  # noqas: E402
from rl.Agent import Agent  # noqa: E402


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
        data = Data(TRAIN_START_DATE, TRAIN_END_DATE, ticker_list=DOW_30_TICKER)
        if args.default_dataset:
            data.load_data()
        elif args.input_dataset:
            data.preprocessed_data_filepath = args.input_dataset
            data.load_data()
        else:
            data.get_preprocessed_data()

        ##
        train_data = data_split(data.data_preprocessed, TRAIN_START_DATE, TRAIN_END_DATE)
        trade_data = data_split(data.data_preprocessed, TEST_START_DATE, TEST_END_DATE)
        agent = Agent(train_data, trade_data)
        agent.train()
        agent.save_trained_model()
        if args.test:
            agent.test()

##
