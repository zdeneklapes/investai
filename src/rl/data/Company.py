# -*- coding: utf-8 -*-
from typing import Optional
import dataclasses
import pandas as pd


@dataclasses.dataclass
class Company:
    symbol: str
    balance_sheet: pd.DataFrame
    income_statement: pd.DataFrame
    cash_flow_statement: pd.DataFrame
    key_metrics: pd.DataFrame
    financial_ratios: pd.DataFrame
    growth: pd.DataFrame
    dcf: Optional[pd.DataFrame] = None
