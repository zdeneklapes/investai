from os import path
from pathlib import Path
import itertools
import datetime

import pandas as pd
import numpy as np
import matplotlib

from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_plot, backtest_stats, get_baseline
from finrl.main import check_and_make_directories
from finrl.config import DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR
from finrl.config_tickers import DOW_30_TICKER
from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC

# from experiments.stock.StockTradingEnv import StockTradingEnv
from src.config.settings import DATA_DIR

##########################


check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

print(DOW_30_TICKER)

TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2021-07-01"
TEST_START_DATE = "2021-07-01"
TEST_END_DATE = "2022-07-01"

df = YahooDownloader(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE, ticker_list=DOW_30_TICKER).fetch_data()

df.head()

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

df.sort_values(["date", "tic"], ignore_index=True).head()

fundamenatal_data_filename = Path(path.join(DATA_DIR, "stock/ai4-finance/dji30_fundamental_data.csv"))
fund = pd.read_csv(fundamenatal_data_filename, low_memory=False)  # dtype param make low_memory warning silent

fund.head()

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

# Check the data
fund_data.head()

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
quick_ratio = ((fund_data["cash_eq"] + fund_data["receivables"]) / fund_data["cur_liabilities"]).to_frame("quick_ratio")

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

# Check the ratio data
ratios.head()

ratios.tail()

# ## 4.4 Deal with NAs and infinite values
# - We replace N/A and infinite values with zero.


# Replace NAs infinite values with zero
final_ratios = ratios.copy()
final_ratios = final_ratios.fillna(0)
final_ratios = final_ratios.replace(np.inf, 0)

final_ratios.head()

final_ratios.tail()

# ## 4.5 Merge stock price data and ratios into one dataframe

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

# Check the final data
processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)

train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade_data = data_split(processed_full, TEST_START_DATE, TEST_END_DATE)
# Check the length of the two datasets
print(len(train_data))
print(len(trade_data))

train_data.head()

train_data.tail()

trade_data.head()

trade_data.tail()

matplotlib.use("Agg")

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

# In[ ]:


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

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

# Set up the agent using DRLAgent() class using the environment created in the previous part
agent = DRLAgent(env=env_train)

if_using_a2c = False
if_using_ddpg = False
if_using_ppo = False
if_using_td3 = False
if_using_sac = True

agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

if if_using_ppo:
    # set up logger
    tmp_path = RESULTS_DIR + "/ppo"
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ppo.set_logger(new_logger_ppo)

trained_ppo = agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=50000) if if_using_ppo else None

agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model("ddpg")

if if_using_ddpg:
    # set up logger
    tmp_path = RESULTS_DIR + "/ddpg"
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ddpg.set_logger(new_logger_ddpg)

trained_ddpg = agent.train_model(model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000) if if_using_ddpg else None

agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c")

if if_using_a2c:
    # set up logger
    tmp_path = RESULTS_DIR + "/a2c"
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_a2c.set_logger(new_logger_a2c)

trained_a2c = agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=50000) if if_using_a2c else None

agent = DRLAgent(env=env_train)
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}

model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

if if_using_td3:
    # set up logger
    tmp_path = RESULTS_DIR + "/td3"
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_td3.set_logger(new_logger_td3)

trained_td3 = agent.train_model(model=model_td3, tb_log_name="td3", total_timesteps=30000) if if_using_td3 else None

agent = DRLAgent(env=env_train)


def fun():
    pass


SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 1000000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

if if_using_sac:
    # set up logger
    tmp_path = RESULTS_DIR + "/sac"
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_sac.set_logger(new_logger_sac)

trained_sac = agent.train_model(model=model_sac, tb_log_name="sac", total_timesteps=30000) if if_using_sac else None

now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
exp_count = "1"

if if_using_ppo:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_ppo_{now}")
    trained_sac.save(trained_model_filename)

if if_using_ddpg:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_ddpg_{now}")
    trained_sac.save(trained_model_filename)

if if_using_a2c:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_a2c_{now}")
    trained_sac.save(trained_model_filename)

if if_using_td3:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_td3_{now}")
    trained_sac.save(trained_model_filename)

if if_using_sac:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_sac_{now}")
    trained_sac.save(trained_model_filename)

now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
exp_count = "1"

if if_using_ppo:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_ppo_{now}")
    trained_sac.save(trained_model_filename)

if if_using_ddpg:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_ddpg_{now}")
    trained_sac.save(trained_model_filename)

if if_using_a2c:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_a2c_{now}")
    trained_sac.save(trained_model_filename)

if if_using_td3:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_td3_{now}")
    trained_sac.save(trained_model_filename)

if if_using_sac:
    trained_model_filename = path.join(TRAINED_MODEL_DIR, f"{exp_count}_trained_sac_{now}")
    trained_sac.save(trained_model_filename)

trained_sac = SAC.load(path.join(TRAINED_MODEL_DIR, "trained_sac_20221023-21h44"))

trade_data = data_split(processed_full, TEST_START_DATE, TEST_END_DATE)
e_trade_gym = StockTradingEnv(df=trade_data, **env_kwargs)

trade_data.head()

df_account_value_ppo, df_actions_ppo = (
    DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym) if if_using_ppo else [None, None]
)

df_account_value_ddpg, df_actions_ddpg = (
    DRLAgent.DRL_prediction(model=trained_ddpg, environment=e_trade_gym) if if_using_ddpg else [None, None]
)

df_account_value_a2c, df_actions_a2c = (
    DRLAgent.DRL_prediction(model=trained_a2c, environment=e_trade_gym) if if_using_a2c else [None, None]
)

df_account_value_td3, df_actions_td3 = (
    DRLAgent.DRL_prediction(model=trained_td3, environment=e_trade_gym) if if_using_td3 else [None, None]
)

df_account_value_sac, df_actions_sac = (
    DRLAgent.DRL_prediction(model=trained_sac, environment=e_trade_gym) if if_using_sac else [None, None]
)

print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

if if_using_ppo:
    print("\n ppo:")
    perf_stats_all_ppo = backtest_stats(account_value=df_account_value_ppo)
    perf_stats_all_ppo = pd.DataFrame(perf_stats_all_ppo)
    perf_stats_all_ppo.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_ppo_" + now + ".csv")

if if_using_ddpg:
    print("\n ddpg:")
    perf_stats_all_ddpg = backtest_stats(account_value=df_account_value_ddpg)
    perf_stats_all_ddpg = pd.DataFrame(perf_stats_all_ddpg)
    perf_stats_all_ddpg.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_ddpg_" + now + ".csv")

if if_using_a2c:
    print("\n a2c:")
    perf_stats_all_a2c = backtest_stats(account_value=df_account_value_a2c)
    perf_stats_all_a2c = pd.DataFrame(perf_stats_all_a2c)
    perf_stats_all_a2c.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_a2c_" + now + ".csv")

if if_using_td3:
    print("\n atd3:")
    perf_stats_all_td3 = backtest_stats(account_value=df_account_value_td3)
    perf_stats_all_td3 = pd.DataFrame(perf_stats_all_td3)
    perf_stats_all_td3.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_td3_" + now + ".csv")

if if_using_sac:
    print("\n sac:")
    perf_stats_all_sac = backtest_stats(account_value=df_account_value_sac)
    perf_stats_all_sac = pd.DataFrame(perf_stats_all_sac)
    perf_stats_all_sac.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_sac_" + now + ".csv")


def test():
    print("==============Get Baseline Stats===========")
    baseline_df = get_baseline(ticker="^DJI", start=TEST_START_DATE, end=TEST_END_DATE)
    stats = backtest_stats(baseline_df, value_col_name="close")
    print(stats)

    print("==============Compare to DJIA===========")
    if if_using_ppo:
        backtest_plot(
            df_account_value_ppo, baseline_ticker="^DJI", baseline_start=TEST_START_DATE, baseline_end=TEST_END_DATE
        )

    if if_using_ddpg:
        backtest_plot(
            df_account_value_ddpg, baseline_ticker="^DJI", baseline_start=TEST_START_DATE, baseline_end=TEST_END_DATE
        )

    if if_using_a2c:
        backtest_plot(
            df_account_value_a2c, baseline_ticker="^DJI", baseline_start=TEST_START_DATE, baseline_end=TEST_END_DATE
        )

    if if_using_td3:
        backtest_plot(
            df_account_value_td3, baseline_ticker="^DJI", baseline_start=TEST_START_DATE, baseline_end=TEST_END_DATE
        )

    if if_using_sac:
        backtest_plot(
            df_account_value_sac, baseline_ticker="^DJI", baseline_start=TEST_START_DATE, baseline_end=TEST_END_DATE
        )


if __name__ == "__main__":
    ...
    test()
