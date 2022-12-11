# -*- coding: utf-8 -*-
import dataclasses
from typing import Callable

import numpy as np
import pandas as pd

from data.DataPreprocessing import DataPreprocessing


@dataclasses.dataclass
class CompanyRatios(DataPreprocessing):
    def __init__(self, non_preproccesed_data: str, cb: Callable):
        super().__init__(non_preproccesed_data, cb)
        self.data = self.get_fundament_data_from_csv()

    def get_ratios(self):
        # Calculate financial ratios
        date = pd.to_datetime(self.data["date"], format="%Y%m%d")
        tic = self.data["tic"].to_frame("tic")

        # Profitability ratios
        # Operating Margin
        OPM = pd.Series(np.empty(self.data.shape[0], dtype=object), name="OPM")
        for i in range(0, self.data.shape[0]):
            if i - 3 < 0:
                OPM[i] = np.nan
            elif self.data.iloc[i, 1] != self.data.iloc[i - 3, 1]:
                OPM.iloc[i] = np.nan
            else:
                OPM.iloc[i] = np.sum(self.data["op_inc_q"].iloc[i - 3 : i]) / np.sum(self.data["rev_q"].iloc[i - 3 : i])

        # Net Profit Margin
        NPM = pd.Series(np.empty(self.data.shape[0], dtype=object), name="NPM")
        for i in range(0, self.data.shape[0]):
            if i - 3 < 0:
                NPM[i] = np.nan
            elif self.data.iloc[i, 1] != self.data.iloc[i - 3, 1]:
                NPM.iloc[i] = np.nan
            else:
                NPM.iloc[i] = np.sum(self.data["net_inc_q"].iloc[i - 3 : i]) / np.sum(
                    self.data["rev_q"].iloc[i - 3 : i]
                )

        # Return On Assets
        ROA = pd.Series(np.empty(self.data.shape[0], dtype=object), name="ROA")
        for i in range(0, self.data.shape[0]):
            if i - 3 < 0:
                ROA[i] = np.nan
            elif self.data.iloc[i, 1] != self.data.iloc[i - 3, 1]:
                ROA.iloc[i] = np.nan
            else:
                ROA.iloc[i] = np.sum(self.data["net_inc_q"].iloc[i - 3 : i]) / self.data["tot_assets"].iloc[i]

        # Return on Equity
        ROE = pd.Series(np.empty(self.data.shape[0], dtype=object), name="ROE")
        for i in range(0, self.data.shape[0]):
            if i - 3 < 0:
                ROE[i] = np.nan
            elif self.data.iloc[i, 1] != self.data.iloc[i - 3, 1]:
                ROE.iloc[i] = np.nan
            else:
                ROE.iloc[i] = np.sum(self.data["net_inc_q"].iloc[i - 3 : i]) / self.data["sh_equity"].iloc[i]

            # For calculating valuation ratios in the next subpart, calculate per share items in advance
        # Earnings Per Share
        EPS = self.data["eps_incl_ex"].to_frame("EPS")

        # Book Per Share
        BPS = (self.data["com_eq"] / self.data["sh_outstanding"]).to_frame("BPS")  # Need to check units

        # Dividend Per Share
        DPS = self.data["div_per_sh"].to_frame("DPS")

        # Liquidity ratios
        # Current ratio
        cur_ratio = (self.data["cur_assets"] / self.data["cur_liabilities"]).to_frame("cur_ratio")

        # Quick ratio
        quick_ratio = ((self.data["cash_eq"] + self.data["receivables"]) / self.data["cur_liabilities"]).to_frame(
            "quick_ratio"
        )

        # Cash ratio
        cash_ratio = (self.data["cash_eq"] / self.data["cur_liabilities"]).to_frame("cash_ratio")

        # Efficiency ratios
        # Inventory turnover ratio
        inv_turnover = pd.Series(np.empty(self.data.shape[0], dtype=object), name="inv_turnover")
        for i in range(0, self.data.shape[0]):
            if i - 3 < 0:
                inv_turnover[i] = np.nan
            elif self.data.iloc[i, 1] != self.data.iloc[i - 3, 1]:
                inv_turnover.iloc[i] = np.nan
            else:
                inv_turnover.iloc[i] = np.sum(self.data["cogs_q"].iloc[i - 3 : i]) / self.data["inventories"].iloc[i]

        # Receivables turnover ratio
        acc_rec_turnover = pd.Series(np.empty(self.data.shape[0], dtype=object), name="acc_rec_turnover")
        for i in range(0, self.data.shape[0]):
            if i - 3 < 0:
                acc_rec_turnover[i] = np.nan
            elif self.data.iloc[i, 1] != self.data.iloc[i - 3, 1]:
                acc_rec_turnover.iloc[i] = np.nan
            else:
                acc_rec_turnover.iloc[i] = np.sum(self.data["rev_q"].iloc[i - 3 : i]) / self.data["receivables"].iloc[i]

        # Payable turnover ratio
        acc_pay_turnover = pd.Series(np.empty(self.data.shape[0], dtype=object), name="acc_pay_turnover")
        for i in range(0, self.data.shape[0]):
            if i - 3 < 0:
                acc_pay_turnover[i] = np.nan
            elif self.data.iloc[i, 1] != self.data.iloc[i - 3, 1]:
                acc_pay_turnover.iloc[i] = np.nan
            else:
                acc_pay_turnover.iloc[i] = np.sum(self.data["cogs_q"].iloc[i - 3 : i]) / self.data["payables"].iloc[i]

        # Leverage financial ratios
        # Debt ratio
        debt_ratio = (self.data["tot_liabilities"] / self.data["tot_assets"]).to_frame("debt_ratio")

        # Debt to Equity ratio
        debt_to_equity = (self.data["tot_liabilities"] / self.data["sh_equity"]).to_frame("debt_to_equity")

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
