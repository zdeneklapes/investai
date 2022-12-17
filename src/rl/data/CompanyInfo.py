# -*- coding: utf-8 -*-
import json
from typing import Optional
import dataclasses
import pandas as pd


@dataclasses.dataclass
class CompanyInfo:
    symbol: str
    balance_sheet: pd.DataFrame
    income_statement: pd.DataFrame
    cash_flow_statement: pd.DataFrame
    key_metrics: pd.DataFrame
    financial_ratios: pd.DataFrame
    growth: pd.DataFrame
    prices_detailed: pd.DataFrame
    dcf: Optional[pd.DataFrame] = None

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
