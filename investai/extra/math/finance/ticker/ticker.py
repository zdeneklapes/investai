# -*- coding: utf-8 -*-
import dataclasses
import enum
from typing import Optional

import pandas as pd


@dataclasses.dataclass
class Ticker:
    """Ticker class to store all the data for a ticker"""

    ticker: Optional[str] = None
    enterprise_value: Optional[pd.DataFrame] = None
    balance_sheet: Optional[pd.DataFrame] = None
    income: Optional[pd.DataFrame] = None
    cash_flow: Optional[pd.DataFrame] = None
    key_metrics: Optional[pd.DataFrame] = None
    financial_ratios: Optional[pd.DataFrame] = None
    growth: Optional[pd.DataFrame] = None
    data_detailed: Optional[pd.DataFrame] = None
    quotes: Optional[pd.DataFrame] = None
    profile: Optional[pd.DataFrame] = None
    dividends: Optional[pd.DataFrame] = None
    ratings: Optional[pd.DataFrame] = None
    dcf: Optional[pd.DataFrame] = None

    class Names(enum.Enum):
        symbol: str = "symbol"
        profile: str = "profile"
        quotes: str = "quotes"
        enterprise_value: str = "enterprise_value"
        balance_sheet: str = "balance_sheet"
        income: str = "income"
        cash_flow: str = "cash_flow"
        key_metrics: str = "key_metrics"
        financial_ratios: str = "financial_ratios"
        growth: str = "growth"
        data_detailed: str = "data_detailed"
        dividends: str = "dividends"
        ratings: str = "ratings"
        dcf: str = "dcf"

        @classmethod
        def list(cls):
            return [e.value for e in cls]
