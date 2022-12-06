import os
from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import PROJECT_STUFF_DIR


class Data:
    @staticmethod
    def get_fundament_data_from_csv() -> pd.DataFrame:
        fundamenatal_data_filename = Path(
            os.path.join(PROJECT_STUFF_DIR, "stock/ai4-finance/dji30_fundamental_data.csv"))
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
            "dlcq": "short_debt",  # Debt in current liabilities
            "ltq": "tot_liabilities",  # Liabilities
        }

        # Omit items that will not be used
        fundamental_specified_data = fundamental_all_data[items_naming.keys()]

        # Rename column names for the sake of readability
        fundamental_specified_data = fundamental_specified_data.rename(columns=items_naming)
        fundamental_specified_data["date"] = pd.to_datetime(fundamental_specified_data["date"], format="%Y%m%d")
        # fund_data.sort_values(["date", "tic"], ignore_index=True)
        return fundamental_specified_data
