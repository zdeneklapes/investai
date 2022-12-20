# -*- coding: utf-8 -*-
"""Source: https://github.dev/AI4Finance-Foundation/FinRL-Meta"""
from copy import deepcopy

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio
from pyfolio import timeseries
from meta import config
import yfinance as yf


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def backtest_stats(account_value, value_col_name="account_value"):
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    # print(perf_stats_all)
    return perf_stats_all


def backtest_plot(
    account_value,
    baseline_start=config.TRADE_START_DATE,
    baseline_end=config.TRADE_END_DATE,
    baseline_ticker="^DJI",
    value_col_name="account_value",
):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(ticker=baseline_ticker, start=baseline_start, end=baseline_end)

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns,
            benchmark_rets=baseline_returns,
            set_context=False,
        )


def get_baseline(ticker, start, end, time_interval="1d", time_zone: str = None) -> pd.DataFrame:
    # TODO: time_zone is not used

    tic = yf.Ticker(ticker)
    baseline_df = tic.history(interval=time_interval, start=start, end=end)
    date_format = "%Y-%m-%d"
    baseline_df.index = baseline_df.index.strftime(date_format)
    baseline_df.insert(0, "date", baseline_df.index)
    baseline_df.columns = baseline_df.columns.str.lower()
    return baseline_df


def get_baseline_finrl(ticker, start, end, time_interval="1d", time_zone: str = None) -> pd.DataFrame:
    # TODO: fixme FinRL-Meta Yahoofinance.download_data() is not working
    from meta.data_processors.yahoofinance import Yahoofinance
    import meta

    if time_zone:
        meta.config.TIME_ZONE_SELFDEFINED = meta.config.TIME_ZONE_PARIS
        meta.config.USE_TIME_ZONE_SELFDEFINED = 1
    meta_yf = Yahoofinance("yahoofinance", start, end, time_interval)
    meta_yf.download_data([ticker])
    return meta_yf.dataframe


def trx_plot(df_trade, df_actions, ticker_list):
    df_trx = pd.DataFrame(np.array(df_actions["transactions"].to_list()))
    df_trx.columns = ticker_list
    df_trx.index = df_actions["date"]
    df_trx.index.name = ""

    for i in range(df_trx.shape[1]):
        df_trx_temp = df_trx.iloc[:, i]
        df_trx_temp_sign = np.sign(df_trx_temp)
        buying_signal = df_trx_temp_sign.apply(lambda x: x > 0)
        selling_signal = df_trx_temp_sign.apply(lambda x: x < 0)

        tic_plot = df_trade[(df_trade["tic"] == df_trx_temp.name) & (df_trade["date"].isin(df_trx.index))]["close"]
        tic_plot.index = df_trx_temp.index

        plt.figure(figsize=(10, 8))
        plt.plot(tic_plot, color="g", lw=2.0)
        plt.plot(
            tic_plot,
            "^",
            markersize=10,
            color="m",
            label="buying signal",
            markevery=buying_signal,
        )
        plt.plot(
            tic_plot,
            "v",
            markersize=10,
            color="k",
            label="selling signal",
            markevery=selling_signal,
        )
        _v = len(buying_signal[buying_signal == True]) + len(selling_signal[selling_signal == True])  # noqa: E712
        plt.title(f"{df_trx_temp.name} Num Transactions: {_v}")
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
        plt.xticks(rotation=45, ha="right")
        plt.show()
