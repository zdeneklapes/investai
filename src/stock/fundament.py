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
import os
import sys
import typing as tp
import warnings

import matplotlib
import pandas as pd
import stable_baselines3 as sb3
from finrl import config as finrl_config

# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.plot import backtest_plot, backtest_stats, get_baseline
from stable_baselines3.common.logger import configure, Logger
from finrl.config import (
    TEST_START_DATE,
    TEST_END_DATE,
)

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from common.Args import Args, argument_parser  # noqa: E402
from config.settings import PROJECT_STUFF_DIR  # noqa: E402
from stock.StockTradingEnv import StockTradingEnv  # noqa: E402
from stock.hyperparameter import HyperParameter  # noqas: E402
from Ratio import Ratio  # noqa: E402
from common.utils import now_time

# ##############################################################################
# GLOBAL VARS
# ##############################################################################
MODELS_MAP = {
    "a2c": sb3.A2C,
    "ddpg": sb3.DDPG,
    "td3": sb3.TD3,
    "ppo": sb3.PPO,
    "sac": sb3.SAC,
}


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


# ##############################################################################
# # Train/Test
# ##############################################################################

# ##############################################################################
# ## Save/Load Helpers
def save_trained_model(trained_model: tp.Any, type_model: str) -> str:
    model_filename = os.path.join(finrl_config.TRAINED_MODEL_DIR, type_model, f"{now_time()}")
    trained_model.save(model_filename)
    return model_filename


def load_trained_model(filepath: str):
    filepath_split = filepath.split(sep="/")
    for type_, model_ in MODELS_MAP.items():
        if type_ in filepath_split:
            return model_.load(filepath)


# ##############################################################################
# ## Train Models
def start_training(agent: str, data: pd.DataFrame, steps: int, params: dict = None) -> str:
    print("=== Training ===")

    #
    train_env = StockTradingEnv(df=data, **HyperParameter.get_env_kwargs(data))

    agent = DRLAgent(env=train_env.get_sb_env()[0])
    model = agent.get_model(agent, model_kwargs=params)
    new_logger: Logger = configure(
        folder=os.path.join(finrl_config.RESULTS_DIR, agent), format_strings=["stdout", "csv", "tensorboard"]
    )
    model.set_logger(new_logger)

    #
    trained_model = agent.train_model(model=model, tb_log_name=agent, total_timesteps=steps)
    filename = save_trained_model(trained_model, type_model=agent)
    return filename


# ##############################################################################
# ### Trade
def test(filepath: str, test_data: pd.DataFrame) -> tp.Any:
    print("==============Test Results===========")

    env_stock_train = StockTradingEnv(df=test_data, **HyperParameter.get_env_kwargs(test_data))
    account_value, actions = DRLAgent.DRL_prediction(model=load_trained_model(filepath), environment=env_stock_train)
    perf_stats_all = backtest_stats(account_value=account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)

    for type_, _ in MODELS_MAP.items():
        if type_ in filepath.split(sep="/"):
            perf_stats_all.to_csv(os.path.join(finrl_config.RESULTS_DIR, type_, f"perf_stats_all{now_time()}.csv"))

    print(perf_stats_all)

    return account_value, actions


def backtest_baseline():
    print("==============Baseline Stats===========")
    baseline_df = get_baseline(ticker="^DJI", start=TEST_START_DATE, end=TEST_END_DATE)
    stats = backtest_stats(baseline_df, value_col_name="close")
    print(stats)
    # TODO: Save stats to log file


def run_train(args: Args) -> tp.List[str]:
    # vars
    trained_models: tp.List = []
    train_data: pd.DataFrame = get_train_data(pd.read_csv(args.dataset_file))

    # Train Models
    for agent, params, steps in HyperParameter.get_hyperparameters():
        filename = start_training(agent, train_data, steps, params)
        trained_models.append(filename)

    return trained_models


def run_test(args: Args):
    test_data = get_test_data(pd.read_csv(filepath_or_buffer=args.dataset_file))

    #
    for model in args.models:
        account_value, _ = test(model, test_data)
        backtest_plot(
            account_value=account_value,
            baseline_ticker="^DJI",
            baseline_start=TEST_START_DATE,
            baseline_end=TEST_END_DATE,
        )
        #     test(model, test_data)
        backtest_baseline()


# ##############################################################################
# # Main
# ##############################################################################
if __name__ == "__main__" and "__file__" in globals():
    config()
    args: Args = argument_parser()

    # Data Preprocessing
    if args.create_dataset and not args.dataset_file:
        args.dataset_file = create_data_file()

    # Train
    if args.train:
        args.models.append(*run_train(args))

    # Test
    if args.models and args.test:
        run_test(args)
