# -*- coding: utf-8 -*-
import os
from typing import Callable
import dataclasses

import pandas as pd

from config.settings import PROJECT_STUFF_DIR


@dataclasses.dataclass(init=False)
class Data:
    def __init__(self, path: str, cb: Callable):
        """
        :param path: relative path to data from PROJECT_STUFF_DIR
        :param cb: callback to load data
        """
        self.path = os.path.join(PROJECT_STUFF_DIR, path)
        self.cb = pd.read_csv

    def get_fundament_data_from_csv(self) -> pd.DataFrame:
        # fundamenatal_data_filename = Path(
        #     os.path.join(PROJECT_STUFF_DIR, "stock/ai4-finance/dji30_fundamental_data.csv"))
        # fundamental_all_data = pd.read_csv(
        #     fundamenatal_data_filename, low_memory=False, index_col=0
        # )  # dtype param make low_memory warning silent
        data_all = self.cb(self.path)
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
