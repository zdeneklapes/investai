# -*- coding: utf-8 -*-
# %%
################################################################################
# # Stock trading with fundamentals
################################################################################

################################################################################
# * This notebook is based on the tutorial:
# https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530


################################################################################
# TODO
################################################################################
# TODO: Add argument parser

import datetime
import itertools

# %%
################################################################################
# INCLUDES
################################################################################
import os
import sys
import typing as tp
import warnings

import matplotlib
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
from finrl import config as finrl_config

# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config_tickers import DOW_30_TICKER
from finrl.main import check_and_make_directories
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import backtest_plot, backtest_stats, get_baseline
from stable_baselines3.common.logger import configure, Logger

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

# from common.Args import argument_parser, ArgNames  # noqa: E402
from common.Args import Args, argument_parser  # noqa: E402

# from common.baseexitcode import BaseExitCode  # noqa: E402
from config.settings import PROJECT_STUFF_DIR  # noqa: E402
from stock.StockTradingEnv import StockTradingEnv  # noqa: E402
from stock.hyperparameter import HyperParameter  # noqas: E402
from Ratio import Ratio  # noqa: E402


# %%
################################################################################
# GLOBAL VARS
################################################################################
class GlobalVariables:
    TRAIN_START_DATE = "2009-01-01"
    TRAIN_END_DATE = "2021-01-01"
    TEST_START_DATE = TRAIN_END_DATE
    TEST_END_DATE = "2022-10-01"

    NOW = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    MODELS_MAP = {
        "a2c": sb3.A2C,
        "ddpg": sb3.DDPG,
        "td3": sb3.TD3,
        "ppo": sb3.PPO,
        "sac": sb3.SAC,
    }


# %%
################################################################################
# FUNCTIONS
################################################################################
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


def get_price_data_dji30() -> pd.DataFrame:
    df = YahooDownloader(
        start_date=GlobalVariables.TRAIN_START_DATE, end_date=GlobalVariables.TEST_END_DATE, ticker_list=DOW_30_TICKER
    ).fetch_data()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.sort_values(["date", "tic"], ignore_index=True).head()
    return df


################################################################################
# ## Merge stock price data and ratios into one dataframe
def get_processed_full_data_done(processed_full: pd.DataFrame) -> pd.DataFrame:
    # Calculate P/E, P/B and dividend yield using daily closing price
    processed_full["PE"] = processed_full["close"] / processed_full["EPS"]
    processed_full["PB"] = processed_full["close"] / processed_full["BPS"]
    processed_full["Div_yield"] = processed_full["DPS"] / processed_full["close"]

    # Drop per share items used for the above calculation
    processed_full = processed_full.drop(columns=["day", "EPS", "BPS", "DPS"])  # TODO: Try to remove it
    # Replace NAs infinite values with zero
    processed_full = processed_full.copy()
    processed_full = processed_full.fillna(0)
    processed_full = processed_full.replace(np.inf, 0)

    processed_full.to_csv(
        os.path.join(PROJECT_STUFF_DIR, f"stock/ai4-finance/dataset_fundament_{GlobalVariables.NOW}.csv")
    )

    return processed_full


def get_processed_full_data() -> pd.DataFrame:
    price_data = get_price_data_dji30()
    ratio = Ratio(path="dataset/stock/ai4-finance/dji30_fundamental_data.csv", cb=pd.read_csv)
    ratios = ratio.get_ratios()

    list_ticker = price_data["tic"].unique().tolist()
    list_date = list(pd.date_range(price_data["date"].min(), price_data["date"].max()))
    combination = list(itertools.product(list_date, list_ticker))

    # Merge stock price data and ratios into one dataframe
    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        price_data, on=["date", "tic"], how="left"
    )
    processed_full = processed_full.merge(ratios, how="left", on=["date", "tic"])
    processed_full = processed_full.sort_values(["tic", "date"])

    # Backfill the ratio data to make them daily
    processed_full = processed_full.bfill(axis="rows")
    print(processed_full.shape)
    return get_processed_full_data_done(processed_full)


################################################################################
# # A Market Environment in OpenAI Gym-style
################################################################################

################################################################################
# ## Data Split
# - Training data period: 2009-01-01 to 2019-01-01
# - Trade data period: 2019-01-01 to 2020-12-31


def get_train_data(processed_full: pd.DataFrame) -> pd.DataFrame:
    train_data = data_split(processed_full, GlobalVariables.TRAIN_START_DATE, GlobalVariables.TRAIN_END_DATE)
    return train_data


def get_test_data(processed_full: pd.DataFrame):
    trade_data = data_split(processed_full, GlobalVariables.TEST_START_DATE, GlobalVariables.TEST_END_DATE)
    return trade_data


################################################################################
# ## Set up the training environment
def get_stock_env(df: pd.DataFrame) -> tp.Tuple[DRLAgent, StockTradingEnv]:
    env_stock_train = StockTradingEnv(df=df, **HyperParameter.get_env_kwargs(df))
    agent = DRLAgent(env=env_stock_train.get_sb_env()[0])
    return agent, env_stock_train


################################################################################
# # Train/Test
################################################################################

################################################################################
# ## Save/Load Helpers
def save_trained_model(trained_model: tp.Any, type_model: str) -> str:
    model_filename = os.path.join(finrl_config.TRAINED_MODEL_DIR, type_model, f"{GlobalVariables.NOW}")
    trained_model.save(model_filename)
    return model_filename


def load_trained_model(filepath: str):
    filepath_split = filepath.split(sep="/")
    for type_, model_ in GlobalVariables.MODELS_MAP.items():
        if type_ in filepath_split:
            return model_.load(filepath)


################################################################################
# ## Train Models
def training_model(model_type: str, train_data: pd.DataFrame, total_timesteps: int, params: dict = None) -> str:
    print("=== Training ===")

    #
    env_stock_train = StockTradingEnv(df=train_data, **HyperParameter.get_env_kwargs(train_data))
    agent = DRLAgent(env=env_stock_train.get_sb_env()[0])
    model = agent.get_model(model_type, model_kwargs=params)
    new_logger: Logger = configure(
        folder=os.path.join(finrl_config.RESULTS_DIR, model_type), format_strings=["stdout", "csv", "tensorboard"]
    )
    model.set_logger(new_logger)

    #
    trained_model = agent.train_model(model=model, tb_log_name=model_type, total_timesteps=total_timesteps)
    filename = save_trained_model(trained_model, type_model=model_type)
    return filename


################################################################################
# ### Trade
def test(filepath: str, test_data: pd.DataFrame) -> tp.Any:
    print("==============Test Results===========")

    env_stock_train = StockTradingEnv(df=test_data, **HyperParameter.get_env_kwargs(test_data))
    account_value, actions = DRLAgent.DRL_prediction(model=load_trained_model(filepath), environment=env_stock_train)
    perf_stats_all = backtest_stats(account_value=account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)

    for type_, _ in GlobalVariables.MODELS_MAP.items():
        if type_ in filepath.split(sep="/"):
            perf_stats_all.to_csv(
                os.path.join(finrl_config.RESULTS_DIR, type_, f"perf_stats_all{GlobalVariables.NOW}.csv")
            )

    print(perf_stats_all)

    return account_value, actions


def backtest_baseline():
    print("==============Baseline Stats===========")
    baseline_df = get_baseline(ticker="^DJI", start=GlobalVariables.TEST_START_DATE, end=GlobalVariables.TEST_END_DATE)
    stats = backtest_stats(baseline_df, value_col_name="close")
    print(stats)
    # TODO: Save stats to log file


def run_train(args: Args) -> tp.List[str]:
    # vars
    trained_models: tp.List = []
    train_data: pd.DataFrame = get_train_data(pd.read_csv(args.dataset))

    # Train Models
    for model_name, model_params, time_steps in [
        HyperParameter.A2C_PARAMS,  # DDPG_PARAMS, # PPO_PARAMS, # TD3_PARAMS, # SAC_PARAMS,
    ]:
        filename = training_model(model_name, train_data, time_steps, model_params)
        trained_models.append(filename)

    return trained_models


def run_test(args: Args):
    test_data = get_test_data(pd.read_csv(filepath_or_buffer=args.dataset))

    #
    for model in args.models:
        account_value, _ = test(model, test_data)
        backtest_plot(
            account_value=account_value,
            baseline_ticker="^DJI",
            baseline_start=GlobalVariables.TEST_START_DATE,
            baseline_end=GlobalVariables.TEST_END_DATE,
        )
        #     test(model, test_data)
        backtest_baseline()


# %%
def main() -> None:
    config()
    args: Args = argument_parser()

    # Data Preprocessing
    if args.create_dataset and not args.dataset:
        args.dataset = get_processed_full_data()

    # Train
    if args.train:
        args.models.append(*run_train(args))

    # Test
    if args.models and args.test:
        run_test(args)


# %%
if __name__ == "__main__":
    main()
