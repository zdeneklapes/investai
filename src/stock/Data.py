# -*- coding: utf-8 -*-
import os
from typing import Callable
import dataclasses
import itertools

import numpy as np
import pandas as pd

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.config import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)

from config.settings import PROJECT_STUFF_DIR, AI_FINANCE_DIR
from common.utils import now_time


@dataclasses.dataclass(init=False)
class Data:
    def __init__(self, start_date: str, end_date: str, non_preproccesed_data: str = None, cb: Callable = pd.read_csv,
                 ticker_list: list = None, preprocessed_data: str = None):
        """
        :param non_preproccesed_data: relative path to data from PROJECT_STUFF_DIR
        :param cb: callback to load data
        """
        self.non_preprocessed_data_filepath: str = os.path.join(
            AI_FINANCE_DIR,
            non_preproccesed_data if non_preproccesed_data else "dji30_fundamental_data.csv"
        )
        self.preprocessed_data_filepath: str = os.path.join(
            AI_FINANCE_DIR,
            preprocessed_data if preprocessed_data else "dataset_fundament_20221027-01h18.csv"
        )
        self.cb: Callable = cb
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.ticker_list: list = ticker_list
        self.data_preprocessed: pd.DataFrame = None

    def save_preprocessed_data(self):
        self.data_preprocessed.to_csv(self.preprocessed_data_filepath)

    def load_data(self) -> pd.DataFrame:
        self.data_preprocessed = self.cb(self.preprocessed_data_filepath)

    def get_fundament_data_from_csv(self) -> pd.DataFrame:
        # fundamenatal_data_filename = Path(
        # fundamental_all_data = pd.read_csv(
        #     fundamenatal_data_filename, low_memory=False, index_col=0
        # )  # dtype param make low_memory warning silent
        data_all = self.cb(self.non_preprocessed_data_filepath)
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
            "cogsq": "cogs_q",  # Cost of Goods Sold
            "invtq": "inventories",  # Inventories
            "apq": "payables",  # Account payable
            "dlttq": "long_debt",  # Long term debt
            "dlcq": "short_debt",  # Debt in current liabilities
            "ltq": "tot_liabilities",  # Liabilities
        }

        # Omit items that will not be used
        useful_data = data_all[items_naming.keys()]

        # Rename column names for the sake of readability
        useful_data = useful_data.rename(columns=items_naming)
        useful_data["date"] = pd.to_datetime(useful_data["date"], format="%Y%m%d")
        # fund_data.sort_values(["date", "tic"], ignore_index=True)
        return useful_data

    def get_preprocessed_data(self) -> (pd.DataFrame, pd.DataFrame):
        df = YahooDownloader(
            start_date=self.start_date, end_date=self.end_date, ticker_list=self.ticker_list
        ).fetch_data()

        #
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df.sort_values(["date", "tic"], ignore_index=True).head()

        #
        url = "https://raw.githubusercontent.com/mariko-sawada/FinRL_with_fundamental_data/main/dow_30_fundamental_wrds.csv"
        fund = pd.read_csv(url)

        #
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
                OPM.iloc[i] = np.sum(fund_data["op_inc_q"].iloc[i - 3: i]) / np.sum(fund_data["rev_q"].iloc[i - 3: i])

        # Net Profit Margin
        NPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="NPM")
        for i in range(0, fund_data.shape[0]):
            if i - 3 < 0:
                NPM[i] = np.nan
            elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
                NPM.iloc[i] = np.nan
            else:
                NPM.iloc[i] = np.sum(fund_data["net_inc_q"].iloc[i - 3: i]) / np.sum(
                    fund_data["rev_q"].iloc[i - 3: i]
                )

        # Return On Assets
        ROA = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="ROA")
        for i in range(0, fund_data.shape[0]):
            if i - 3 < 0:
                ROA[i] = np.nan
            elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
                ROA.iloc[i] = np.nan
            else:
                ROA.iloc[i] = np.sum(fund_data["net_inc_q"].iloc[i - 3: i]) / fund_data["tot_assets"].iloc[i]

        # Return on Equity
        ROE = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="ROE")
        for i in range(0, fund_data.shape[0]):
            if i - 3 < 0:
                ROE[i] = np.nan
            elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
                ROE.iloc[i] = np.nan
            else:
                ROE.iloc[i] = np.sum(fund_data["net_inc_q"].iloc[i - 3: i]) / fund_data["sh_equity"].iloc[i]

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
                inv_turnover.iloc[i] = np.sum(fund_data["cogs_q"].iloc[i - 3: i]) / fund_data["inventories"].iloc[i]

        # Receivables turnover ratio
        acc_rec_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="acc_rec_turnover")
        for i in range(0, fund_data.shape[0]):
            if i - 3 < 0:
                acc_rec_turnover[i] = np.nan
            elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
                acc_rec_turnover.iloc[i] = np.nan
            else:
                acc_rec_turnover.iloc[i] = np.sum(fund_data["rev_q"].iloc[i - 3: i]) / fund_data["receivables"].iloc[i]

        # Payable turnover ratio
        acc_pay_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="acc_pay_turnover")
        for i in range(0, fund_data.shape[0]):
            if i - 3 < 0:
                acc_pay_turnover[i] = np.nan
            elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
                acc_pay_turnover.iloc[i] = np.nan
            else:
                acc_pay_turnover.iloc[i] = np.sum(fund_data["cogs_q"].iloc[i - 3: i]) / fund_data["payables"].iloc[i]

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

        # Replace NAs infinite values with zero
        final_ratios = ratios.copy()
        final_ratios = final_ratios.fillna(0)
        final_ratios = final_ratios.replace(np.inf, 0)
        list_ticker = df["tic"].unique().tolist()

        #
        list_date = list(pd.date_range(df["date"].min(), df["date"].max()))
        combination = list(itertools.product(list_date, list_ticker))

        # Merge stock price data and ratios into one dataframe
        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(df, on=["date", "tic"], how="left")
        processed_full = processed_full.merge(final_ratios, how="left", on=["date", "tic"])
        processed_full = processed_full.sort_values(["tic", "date"])

        # Backfill the ratio data to make them daily
        processed_full = processed_full.bfill(axis="rows")

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

        # Check the final data
        processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)

        # train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
        # trade_data = data_split(processed_full, TEST_START_DATE, TEST_END_DATE)
        # Check the length of the two datasets
        # print(len(train_data))
        # print(len(trade_data))

        self.data_preprocessed = processed_full
        return self.data_preprocessed


def save(self):
    pass
