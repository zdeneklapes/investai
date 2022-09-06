import os
from os import path
import sys
from typing import Optional

import yfinance as yf
import pandas as pd
import finnhub
from tqdm import tqdm

from shared.types import param_type
from stock.stockDataset import StockDataset
from stock.stockAnalyses import StockAnalyses


class MyInterests:
    def __init__(self, margin):
        self.margin_on_products = margin


def main(*, hyperparams: param_type, args: param_type):
    # TODO (continue_from_here)

    x = StockDataset(hyperparams=hyperparams)  # TODO: Bad parsing stock datas
    yahoo_data = x.load_all_stock_symbols(amount=5)  # Bad handle stocks which can't be GET from yahoo finance

    # TODO (DEBUG)
    print(f"{yahoo_data.symbols=}")
    f = yahoo_data.income_statement(frequency='a', trailing=True)

    # if type(f) == str:
    #     tics = f.replace('Income Statement data unavailable for', '')
    #     tics = tics.split(',')
    #     tics = list(map(lambda x: x.strip() ,tics))
    #
    #     yahoo_data = x.load_all_stock_symbols(amount=5, remove_tickers=tics)
    #     print(f"{yahoo_data.symbols=}")
    #     f = yahoo_data.income_statement()
    #
    #     print(f)
    # x.test()
