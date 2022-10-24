# %%
################################################################################
# coding: utf-8

# # Stock trading with fundamentals
#
# * This notebook is based on the tutorial:
# https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530

# * This project is a result of the almuni-mentored research project at Columbia
# University, Application of Reinforcement Learning to Finance.
# * For detailed explanation, please check out the
# Medium article:
# medium.com/@mariko.sawada1/automated-stock-trading-with-deep-reinforcement-learning-and-financial-data-a63286ccbe2b


################################################################################
# # Part 1. Task Description

# We train a DRL agent for stock trading. The task is modeled as a Markov Decision Process (MDP),
# and the objective function is maximizing (expected) cumulative return.
#
# We specify the state-action-reward as follows:
#
# * **State s**: The state space represents an agent's perception of the market environment.
# Like a human trader analyzes various information, here our agent passively observes many features and
# learn by interacting with the market environment (usually by replaying historical data).
#
# * **Action a**: The action space includes allowed actions that an agent can take
# at each state. For example, a ∈ {−1, 0, 1}, where −1, 0, 1 represent
# selling, holding, and buying. When an action operates multiple shares, a ∈{−k, ..., −1, 0, 1, ..., k}, e.g.. "Buy
# 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively
#
# * **Reward function r(s, a, s′)**: Reward is an incentive for an agent to learn
# a better policy. For example, it can be the change of the portfolio value when taking a at state s and arriving
# at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v
# represent the portfolio values at state s′ and s, respectively
#
#
# **Market environment**: 30 consituent stocks of Dow Jones Industrial Average (DJIA) index. Accessed at
# the starting date of the testing period.
#
#
# The data of the single stock that we will use for this case study is obtained from Yahoo
# Finance API. The data contains Open-High-Low-Close prices and volume.


# %%
###############################################################################
# INCLUDES
###############################################################################

import datetime

from os import path
import sys
from pathlib import Path
import itertools
import typing as tp

import matplotlib
import numpy as np
import pandas as pd

from finrl import config as finrl_config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_plot, backtest_stats, get_baseline
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
)
from finrl.config_tickers import DOW_30_TICKER

from stable_baselines3.common.logger import configure
from stable_baselines3.common.base_class import BaseAlgorithm

# from stable_baselines3.common import logger

from src.config.settings import DATA_DIR

# %%
###############################################################################
# GLOBAL VARS
################################################################################
# # Part 3. Download Stock Data from Yahoo Finance
# Yahoo Finance provides stock data, financial news, financial reports, etc. Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** in FinRL-Meta to fetch data via Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP
# (or up to a total of 48,000 requests a day).

TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2021-07-01"
TEST_START_DATE = "2021-07-01"
TEST_END_DATE = "2022-07-01"


# %%
###############################################################################
# FUNCTIONS
################################################################################


def config():
    matplotlib.use("Agg")


def prepare_data_from_yf() -> pd.DataFrame:
    df = YahooDownloader(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE, ticker_list=DOW_30_TICKER).fetch_data()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.sort_values(["date", "tic"], ignore_index=True).head()
    return df


################################################################################
# # Part 4: Preprocess fundamental data
# - Import finanical data downloaded from Compustat via WRDS(Wharton Research Data Service)
# - Preprocess the dataset and calculate financial ratios
# - Add those ratios to the price data preprocessed in Part 3
# - Calculate price-related ratios such as P/E and P/B

################################################################################
# ## 4.1 Import the financial data
def get_fundament_data() -> pd.DataFrame:
    fundamenatal_data_filename = Path(path.join(DATA_DIR, "stock/ai4-finance/dji30_fundamental_data.csv"))
    fund = pd.read_csv(fundamenatal_data_filename, low_memory=False)  # dtype param make low_memory warning silent

    # ## 4.2 Specify items needed to calculate financial ratios
    # - To learn more about the data description of the dataset, please check WRDS's
    # website(https://wrds-www.wharton.upenn.edu/). Login will be required.

    # List items that are used to calculate financial ratios
    items = [
        "datadate",  # Date
        "tic",  # Ticker
        "oiadpq",  # Quarterly operating income
        "revtq",  # Quartely revenue
        "niq",  # Quartely net income
        "atq",  # Total asset
        "teqq",  # Shareholder's equity
        "epspiy",  # EPS(Basic) incl. Extraordinary items
        "ceqq",  # Common Equity
        "cshoq",  # Common Shares Outstanding
        "dvpspq",  # Dividends per share
        "actq",  # Current assets
        "lctq",  # Current liabilities
        "cheq",  # Cash & Equivalent
        "rectq",  # Recievalbles
        "cogsq",  # Cost of  Goods Sold
        "invtq",  # Inventories
        "apq",  # Account payable
        "dlttq",  # Long term debt
        "dlcq",  # Debt in current liabilites
        "ltq",  # Liabilities
    ]

    # Omit items that will not be used
    fund_data = fund[items]

    # Rename column names for the sake of readability
    fund_data = fund_data.rename(
        columns={
            "datadate": "date",  # Date
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
    )
    return fund_data


################################################################################
# ## 4.3 Calculate financial ratios
# - For items from Profit/Loss statements, we calculate LTM (Last Twelve Months) and use them to derive profitability
# related ratios such as Operating Maring and ROE. For items from balance sheets, we use the numbers on the day.
# - To check the definitions of the financial ratios calculated here, please refer to
# CFI's website: https://corporatefinanceinstitute.com/resources/knowledge/finance/financial-ratios/


def get_ratios(fund_data: pd.DataFrame):
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
# ## 4.5 Merge stock price data and ratios into one dataframe
# - Merge the price dataframe preprocessed in Part 3 and the ratio dataframe created in this part
# - Since the prices are daily and ratios are quartely, we have NAs in the ratio
# columns after merging the two dataframes. We deal with this by backfilling the ratios.


def get_processed_full_data(df: pd.DataFrame, final_ratios: pd.DataFrame) -> pd.DataFrame:
    list_ticker = df["tic"].unique().tolist()
    list_date = list(pd.date_range(df["date"].min(), df["date"].max()))
    combination = list(itertools.product(list_date, list_ticker))

    # Merge stock price data and ratios into one dataframe
    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(df, on=["date", "tic"], how="left")
    processed_full = processed_full.merge(final_ratios, how="left", on=["date", "tic"])
    processed_full = processed_full.sort_values(["tic", "date"])

    # Backfill the ratio data to make them daily
    processed_full = processed_full.bfill(axis="rows")
    print(processed_full.shape)

    # ## 4.6 Calculate market valuation ratios using daily stock price data

    # Calculate P/E, P/B and dividend yield using daily closing price
    processed_full["PE"] = processed_full["close"] / processed_full["EPS"]
    processed_full["PB"] = processed_full["close"] / processed_full["BPS"]
    processed_full["Div_yield"] = processed_full["DPS"] / processed_full["close"]

    # Drop per share items used for the above calculation
    processed_full = processed_full.drop(columns=["day", "EPS", "BPS", "DPS"])
    # Replace NAs infinite values with zero
    processed_full = processed_full.copy()
    processed_full = processed_full.fillna(0)
    processed_full = processed_full.replace(np.inf, 0)
    print(processed_full.shape)

    return processed_full


################################################################################
# # Part 5. A Market Environment in OpenAI Gym-style
# The training process involves observing stock price change, taking an action and reward's calculation. By interacting
# with the market environment, the agent will eventually derive a trading strategy that may maximize (expected) rewards.
#
# Our market environment, based on OpenAI Gym, simulates stock markets with historical market data.

################################################################################
# ## 5.1 Data Split
# - Training data period: 2009-01-01 to 2019-01-01
# - Trade data period: 2019-01-01 to 2020-12-31


def get_train_data(processed_full: pd.DataFrame):
    train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    return train_data


def get_test_data(processed_full: pd.DataFrame):
    trade_data = data_split(processed_full, TEST_START_DATE, TEST_END_DATE)
    return trade_data


################################################################################
# ## 5.2 Set up the training environment


def get_env(train_data: pd.DataFrame) -> tp.Tuple[DRLAgent, StockTradingEnv, dict]:
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

    stock_dimension = len(train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(ratio_list) * stock_dimension  # TODO: Why?
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    print()

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
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)

    # Environment for Training
    env_train, _ = e_train_gym.get_sb_env()

    # Agent
    agent = DRLAgent(env=env_train)
    return agent, e_train_gym, env_kwargs


################################################################################
# # Part 6: Train DRL Agents
# * The DRL algorithms are from **Stable Baselines 3**. Users are also encouraged
# to try **ElegantRL** and **Ray RLlib**.
# * FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG,
# Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.
#
# Set up the agent using DRLAgent() class using the environment created in the previous part
# ### Agent Training: 5 algorithms (A2C, DDPG, PPO, TD3, SAC)

################################################################################
# ## Part 6.1: Saving helper
def save_trained_model(model: tp.Any, name: str) -> str:
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    model_filename = path.join(TRAINED_MODEL_DIR, f"{name}_{now}")
    model.save(model_filename)
    return model_filename


def load_trained_model(name: str):
    model = BaseAlgorithm.load(name)
    return model


################################################################################
# ## Part 6.2: Models
def train_using_a2c(agent: DRLAgent):
    model_a2c = agent.get_model("a2c")

    # set up logger
    tmp_path = RESULTS_DIR + "/a2c"
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_a2c.set_logger(new_logger_a2c)

    trained_a2c = agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=50000)
    save_trained_model(trained_a2c, name="trained_a2c")


def train_using_ddpg(env_train: DRLAgent):
    agent = DRLAgent(env=env_train)
    model_ddpg = agent.get_model("ddpg")

    # set up logger
    tmp_path = RESULTS_DIR + "/ddpg"
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ddpg.set_logger(new_logger_ddpg)

    trained_ddpg = agent.train_model(model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000)
    save_trained_model(trained_ddpg, name="trained_ddpg")


def train_using_ppo(agent: DRLAgent):
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    # set up logger
    tmp_path = RESULTS_DIR + "/ppo"
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ppo.set_logger(new_logger_ppo)

    trained_ppo = agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=50000)
    save_trained_model(trained_ppo, name="trained_ppo")


def train_using_td3(agent: DRLAgent):
    TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}

    model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

    # set up logger
    tmp_path = RESULTS_DIR + "/td3"
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_td3.set_logger(new_logger_td3)

    trained_td3 = agent.train_model(model=model_td3, tb_log_name="td3", total_timesteps=30000)
    save_trained_model(trained_td3, name="trained_td3")


def train_using_sac(agent: DRLAgent):
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }

    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

    # set up logger
    tmp_path = RESULTS_DIR + "/sac"
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_sac.set_logger(new_logger_sac)

    trained_sac = agent.train_model(model=model_sac, tb_log_name="sac", total_timesteps=30000)
    save_trained_model(trained_sac, name="trained_sac")


################################################################################
# ## 7 Trading
# Assume that we have $1,000,000 initial capital at TEST_START_DATE. We use the DDPG model to trade Dow jones 30 stocks.

# ### 7.1 Trade
#
# DRL model needs to update periodically in order to take full
# advantage of the data, ideally we need to retrain our model yearly, quarterly, or monthly. We also need
# to tune the parameters along the way, in this notebook I only use the in-sample data from
# 2009-01 to 2018-12 to tune the parameters once, so there is some alpha decay here as the length of trade date extends.
#
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning
# process and are usually determined by testing some variations.


def test(model: tp.Any, env_gym):
    DRLAgent.DRL_prediction(model=model, environment=env_gym)


################################################################################
# # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated
# backtesting tool is preferred because it reduces the human error. We usually use the Quantopian
# pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual
# plots that provide a comprehensive image of the performance of a trading strategy.

################################################################################
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class


def backtest(df_account_value: tp.Any):
    print("==============Get Backtest Results===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    perf_stats_all_ppo = backtest_stats(account_value=df_account_value)
    perf_stats_all_ppo = pd.DataFrame(perf_stats_all_ppo)
    perf_stats_all_ppo.to_csv("./" + finrl_config.RESULTS_DIR + "/perf_stats_all_ppo_" + now + ".csv")


def backtest_baseline():
    print("==============Get Baseline Stats===========")
    baseline_df = get_baseline(ticker="^DJI", start=TEST_START_DATE, end=TEST_END_DATE)
    stats = backtest_stats(baseline_df, value_col_name="close")
    print(stats)
    # TODO: Save stats to log file


#################################################################################
# ## 7.2 BackTestPlot


def custom_backtest_plot(df_account_value: tp.Any):
    """
    S&P 500: ^GSPC
    Dow Jones Index: ^DJI
    NASDAQ 100: ^NDX

    :param df_account_value:
    :return:
    """

    print("==============Compare to DJIA===========")
    backtest_plot(df_account_value, baseline_ticker="^DJI", baseline_start=TEST_START_DATE, baseline_end=TEST_END_DATE)


# %%

if __name__ == "__main__":
    #
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

    #
    if_using_a2c = False
    if_using_ddpg = False
    if_using_ppo = False
    if_using_td3 = False
    if_using_sac = True

    sys.exit(0)
