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

# %%
################################################################################
# INCLUDES
################################################################################


import os
import sys
import warnings
from pathlib import Path
import itertools
import datetime
import typing as tp

import matplotlib
import numpy as np
import pandas as pd

from finrl import config as finrl_config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split

# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_plot, backtest_stats, get_baseline
from finrl.main import check_and_make_directories
from finrl.config_tickers import DOW_30_TICKER

from stable_baselines3.common.logger import configure, Logger
import stable_baselines3 as sb3

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from StockTraidingEnv import StockTradingEnv  # noqa: E402
from config.settings import DATA_DIR  # noqa: E402
from common.argument_parser import argument_parser, ArgNames  # noqa: E402
import _experiments.stock.hyperparameters as hp  # noqa: E402

# %%
################################################################################
# GLOBAL VARS
################################################################################
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


def get_price_data() -> pd.DataFrame:
    df = YahooDownloader(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE, ticker_list=DOW_30_TICKER).fetch_data()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.sort_values(["date", "tic"], ignore_index=True).head()
    return df


################################################################################
# # Preprocess fundamental data


def get_fundament_data_from_csv() -> pd.DataFrame:
    fundamenatal_data_filename = Path(os.path.join(DATA_DIR, "stock/ai4-finance/dji30_fundamental_data.csv"))
    fundamental_all_data = pd.read_csv(
        fundamenatal_data_filename, low_memory=False, index_col=0
    )  # dtype param make low_memory warning silent
    items_naming = {
        "datadate": "date",  # Date
        "tic": "tic",  # Ticker
        "oiadpq": "op_inc_q",  # Quarterly operating income
        "revtq": "rev_q",  # Quartely revenue
        "niq": "net_inc_q",  # Quartely net income
        "atq": "tot_assets",  # Assets
        "teqq": "sh_equity",  # Shareholder's equity
        "epspiy": "eps_incl_ex",  # EPS(Basic) incl. Extraordinary items
        "ceqq": "com_eq",  # Common Equity
        "cshoq": "sh_outstanding",  # Common Shares Outstanding
        "dvpspq": "div_per_sh",  # Dividends per share
        "actq": "cur_assets",  # Current assets
        "lctq": "cur_liabilities",  # Current liabilities
        "cheq": "cash_eq",  # Cash & Equivalent
        "rectq": "receivables",  # Receivalbles
        "cogsq": "cogs_q",  # Cost of  Goods Sold
        "invtq": "inventories",  # Inventories
        "apq": "payables",  # Account payable
        "dlttq": "long_debt",  # Long term debt
        "dlcq": "short_debt",  # Debt in current liabilites
        "ltq": "tot_liabilities",  # Liabilities
    }

    # Omit items that will not be used
    fundamental_specified_data = fundamental_all_data[items_naming.keys()]

    # Rename column names for the sake of readability
    fundamental_specified_data = fundamental_specified_data.rename(columns=items_naming)
    fundamental_specified_data["date"] = pd.to_datetime(fundamental_specified_data["date"], format="%Y%m%d")
    # fund_data.sort_values(["date", "tic"], ignore_index=True)
    return fundamental_specified_data


################################################################################
# ## Calculate financial ratios
def get_ratios():
    fund_data = get_fundament_data_from_csv()

    # Calculate financial ratios
    date = pd.to_datetime(fund_data["date"], format="%Y%m%d")
    tic = fund_data["tic"].to_frame("tic")

    # Profitability ratios
    # Operating Margin
    OPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="OPM")
    for i in range(0, fund_data.shape[0]):
        if i - 3 < 0:
            OPM[i] = np.nan
        elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
            OPM.iloc[i] = np.nan
        else:
            OPM.iloc[i] = np.sum(fund_data["op_inc_q"].iloc[i - 3 : i]) / np.sum(fund_data["rev_q"].iloc[i - 3 : i])

    # Net Profit Margin
    NPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="NPM")
    for i in range(0, fund_data.shape[0]):
        if i - 3 < 0:
            NPM[i] = np.nan
        elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
            NPM.iloc[i] = np.nan
        else:
            NPM.iloc[i] = np.sum(fund_data["net_inc_q"].iloc[i - 3 : i]) / np.sum(fund_data["rev_q"].iloc[i - 3 : i])

    # Return On Assets
    ROA = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="ROA")
    for i in range(0, fund_data.shape[0]):
        if i - 3 < 0:
            ROA[i] = np.nan
        elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
            ROA.iloc[i] = np.nan
        else:
            ROA.iloc[i] = np.sum(fund_data["net_inc_q"].iloc[i - 3 : i]) / fund_data["tot_assets"].iloc[i]

    # Return on Equity
    ROE = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="ROE")
    for i in range(0, fund_data.shape[0]):
        if i - 3 < 0:
            ROE[i] = np.nan
        elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
            ROE.iloc[i] = np.nan
        else:
            ROE.iloc[i] = np.sum(fund_data["net_inc_q"].iloc[i - 3 : i]) / fund_data["sh_equity"].iloc[i]

        # For calculating valuation ratios in the next subpart, calculate per share items in advance
    # Earnings Per Share
    EPS = fund_data["eps_incl_ex"].to_frame("EPS")

    # Book Per Share
    BPS = (fund_data["com_eq"] / fund_data["sh_outstanding"]).to_frame("BPS")  # Need to check units

    # Dividend Per Share
    DPS = fund_data["div_per_sh"].to_frame("DPS")

    # Liquidity ratios
    # Current ratio
    cur_ratio = (fund_data["cur_assets"] / fund_data["cur_liabilities"]).to_frame("cur_ratio")

    # Quick ratio
    quick_ratio = ((fund_data["cash_eq"] + fund_data["receivables"]) / fund_data["cur_liabilities"]).to_frame(
        "quick_ratio"
    )

    # Cash ratio
    cash_ratio = (fund_data["cash_eq"] / fund_data["cur_liabilities"]).to_frame("cash_ratio")

    # Efficiency ratios
    # Inventory turnover ratio
    inv_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="inv_turnover")
    for i in range(0, fund_data.shape[0]):
        if i - 3 < 0:
            inv_turnover[i] = np.nan
        elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
            inv_turnover.iloc[i] = np.nan
        else:
            inv_turnover.iloc[i] = np.sum(fund_data["cogs_q"].iloc[i - 3 : i]) / fund_data["inventories"].iloc[i]

    # Receivables turnover ratio
    acc_rec_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="acc_rec_turnover")
    for i in range(0, fund_data.shape[0]):
        if i - 3 < 0:
            acc_rec_turnover[i] = np.nan
        elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
            acc_rec_turnover.iloc[i] = np.nan
        else:
            acc_rec_turnover.iloc[i] = np.sum(fund_data["rev_q"].iloc[i - 3 : i]) / fund_data["receivables"].iloc[i]

    # Payable turnover ratio
    acc_pay_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="acc_pay_turnover")
    for i in range(0, fund_data.shape[0]):
        if i - 3 < 0:
            acc_pay_turnover[i] = np.nan
        elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
            acc_pay_turnover.iloc[i] = np.nan
        else:
            acc_pay_turnover.iloc[i] = np.sum(fund_data["cogs_q"].iloc[i - 3 : i]) / fund_data["payables"].iloc[i]

    # Leverage financial ratios
    # Debt ratio
    debt_ratio = (fund_data["tot_liabilities"] / fund_data["tot_assets"]).to_frame("debt_ratio")

    # Debt to Equity ratio
    debt_to_equity = (fund_data["tot_liabilities"] / fund_data["sh_equity"]).to_frame("debt_to_equity")

    # Create a dataframe that merges all the ratios
    ratios = pd.concat(
        [
            date,
            tic,
            OPM,
            NPM,
            ROA,
            ROE,
            EPS,
            BPS,
            DPS,
            cur_ratio,
            quick_ratio,
            cash_ratio,
            inv_turnover,
            acc_rec_turnover,
            acc_pay_turnover,
            debt_ratio,
            debt_to_equity,
        ],
        axis=1,
    )

    # ## 4.4 Deal with NAs and infinite values
    # - We replace N/A and infinite values with zero.

    # Replace NAs infinite values with zero
    final_ratios = ratios.copy()
    final_ratios = final_ratios.fillna(0)
    final_ratios = final_ratios.replace(np.inf, 0)

    return final_ratios


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

    processed_full.to_csv(os.path.join(DATA_DIR, f"stock/ai4-finance/dataset_fundament_{NOW}.csv"))

    return processed_full


def get_processed_full_data() -> pd.DataFrame:
    price_data = get_price_data()
    ratios_fundament_data = get_ratios()

    list_ticker = price_data["tic"].unique().tolist()
    list_date = list(pd.date_range(price_data["date"].min(), price_data["date"].max()))
    combination = list(itertools.product(list_date, list_ticker))

    # Merge stock price data and ratios into one dataframe
    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        price_data, on=["date", "tic"], how="left"
    )
    processed_full = processed_full.merge(ratios_fundament_data, how="left", on=["date", "tic"])
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


def get_train_data(processed_full: pd.DataFrame):
    train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    return train_data


def get_test_data(processed_full: pd.DataFrame):
    trade_data = data_split(processed_full, TEST_START_DATE, TEST_END_DATE)
    return trade_data


################################################################################
# ## Set up the training environment
def get_stock_env(df: pd.DataFrame) -> tp.Tuple[DRLAgent, StockTradingEnv]:
    env_stock_train = StockTradingEnv(df=df, **hp.get_env_kwargs(df))
    agent = DRLAgent(env=env_stock_train.get_sb_env()[0])
    return agent, env_stock_train


################################################################################
# # Train/Test
################################################################################

################################################################################
# ## Save/Load Helpers
def save_trained_model(trained_model: tp.Any, type_model: str) -> str:
    model_filename = os.path.join(finrl_config.TRAINED_MODEL_DIR, type_model, f"{NOW}")
    trained_model.save(model_filename)
    return model_filename


def load_trained_model(filepath: str):
    filepath_split = filepath.split(sep="/")
    for type_, model_ in MODELS_MAP.items():
        if type_ in filepath_split:
            return model_.load(filepath)


################################################################################
# ## Train Models
def training_model(model_type: str, train_data: pd.DataFrame, total_timesteps: int, params: dict = None) -> str:
    print("=== Training ===")

    #
    env_stock_train = StockTradingEnv(df=train_data, **hp.get_env_kwargs(train_data))
    agent = DRLAgent(env=env_stock_train.get_sb_env()[0])
    model: sb3.SAC = agent.get_model(model_type, model_kwargs=params)
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

    env_stock_train = StockTradingEnv(df=test_data, **hp.get_env_kwargs(test_data))
    account_value, actions = DRLAgent.DRL_prediction(model=load_trained_model(filepath), environment=env_stock_train)
    perf_stats_all = backtest_stats(account_value=account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)

    for type_, _ in MODELS_MAP.items():
        if type_ in filepath.split(sep="/"):
            perf_stats_all.to_csv(os.path.join(finrl_config.RESULTS_DIR, type_, f"perf_stats_all{NOW}.csv"))

    print(perf_stats_all)

    return account_value, actions


def backtest_baseline():
    print("==============Baseline Stats===========")
    baseline_df = get_baseline(ticker="^DJI", start=TEST_START_DATE, end=TEST_END_DATE)
    stats = backtest_stats(baseline_df, value_col_name="close")
    print(stats)
    # TODO: Save stats to log file


# %%
def main():
    args = argument_parser()
    config()
    prepared_dataset: pd.DataFrame
    trained_models = args[ArgNames.MODEL]

    if not args[ArgNames.DATASET]:
        prepared_dataset = get_processed_full_data()
    else:
        prepared_dataset = pd.read_csv(args["dataset"], index_col=0)

    if args[ArgNames.TRAIN]:
        train_data = get_train_data(prepared_dataset)
        for is_run_training, model_name, model_params, time_steps in [
            hp.A2C_PARAMS,
            hp.DDPG_PARAMS,
            hp.PPO_PARAMS,
            hp.TD3_PARAMS,
            hp.SAC_PARAMS,
        ]:
            if is_run_training:
                filename = training_model(model_name, train_data, time_steps, model_params)
                trained_models.append(filename)

    if args[ArgNames.TEST]:
        test_data = get_test_data(prepared_dataset)
        for model in trained_models:
            account_value, _ = test(model, test_data)
            backtest_plot(
                account_value, baseline_ticker="^DJI", baseline_start=TEST_START_DATE, baseline_end=TEST_END_DATE
            )
            backtest_baseline()


# %%
if __name__ == "__main__":
    main()
    sys.exit(0)
