# -*- coding: utf-8 -*-
import enum
import json
from typing import Optional
import dataclasses
import pandas as pd


@dataclasses.dataclass
class CompanyInfo:
    symbol: Optional[str] = None
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

    def __json__(self):
        return {
            "symbol": self.symbol,
        }


class CompanyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, complex):
            return [o.real, o.imag]
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)
