import os
from os import path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from time import time
from datetime import date, datetime, timedelta

import pandas as pd
import finnhub
from tqdm import tqdm
import yfinance as yf
from yahooquery import Ticker

from shared.types import param_type

class AllStocks:
    def __init__(self):
        all_stocks = None


    def get_all_stock_StockSymbol(self, *, exchanges_filename: str = '') -> Optional[pd.DataFrame]:
        """
        Should save all stocks on the World into .csv file.

        Returns:
            Pandas.DataFrame : Jo
        """
        ss = StockSymbol(api_key=os.getenv('STOCK_SYMBOL_API_KEY'))
        result = pd.DataFrame()
        loop = tqdm(ss.market_list)

        #
        for market in loop:
            exchange = market['abbreviation']
            exchange_stocks = ss.get_symbol_list(market=exchange)  # finnhub_client.stock_symbols(exchange)
            result = pd.concat([result, pd.DataFrame(exchange_stocks)])
            loop.set_description(desc=f"Loading exchange: {market['market']}, symbols: {len(exchange_stocks)}")

        #
        result.to_csv(path_or_buf=f'{self.saved_file}')
        return result


    def get_all_stock_Finnhub(self, *, exchanges_filename: str = '') -> Optional[pd.DataFrame]:
        """ TODO """
        csv = pd.read_csv(exchanges_filename)
        finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))

        result = pd.DataFrame()
        loop = tqdm(csv['code'].tolist())
        for market in loop:
            exchange_stocks = finnhub_client.stock_symbols(exchange=market)
            result = pd.concat([result, pd.DataFrame(exchange_stocks)])
            loop.set_description(desc=f"Loading exchange: {market}, symbols: {len(exchange_stocks)}")

        result.to_csv(path_or_buf=f'{self.__get_all_stock2.__name__}.csv')
        return result


class StockDataset:
    def __init__(self, hyperparams: param_type):
        self.hyperparams = hyperparams
        self.saved_file = path.expanduser(self.hyperparams['datasets_home']['server']['exchanges'])
        self.exchange_file = path.expanduser(
            path.join(self.hyperparams['datasets_home']['server']['finnhub'], 'finnhub_exchanges.csv'))

    def load_all_stock_symbols(self, amount, remove_tickers=None) -> Ticker:
        tickers = pd.read_csv(filepath_or_buffer=self.saved_file, usecols=['symbol']).loc[:amount]
        print(tickers)
        if remove_tickers:
            tickers.drop(tickers[(tickers['symbol'] == remove_tickers)].index, inplace=True)
            print(tickers)
        tics = Ticker(symbols=tickers['symbol'])
        return tics

        # income_statement = tics.income_statement(trailing=True)
        # income_statement.to_csv('foo.csv')
        # f = income_statement['asOfDate'] == str(date(2021, 6, 30))
        # print(income_statement[f]['TotalRevenue'])

    def get_stats(self, ticker):
        info = yf.Tickers(ticker).tickers[ticker].info
        print(f"{ticker} {info['currentPrice']} {info['marketCap']}")

    def test(self):
        ticker_list = ['AAPL', 'ORCL', 'PREM.L', 'UKOG.L', 'KOD.L', 'TOM.L', 'VELA.L', 'MSFT', 'AMZN', 'GOOG']

        with ThreadPoolExecutor() as executor:
            executor.map(self.get_stats, ticker_list)
