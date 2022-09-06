import sys
from os import path
import math
from typing import List, Dict, Optional, Union
from datetime import date, datetime

import finnhub
import yfinance as yf
import tqdm
import pandas as pd
import pandas_datareader as pdr

from src.shared.utils import parse_yaml
from src.shared.exitcode import ExitCode
from src.shared.logging_setup import LOGGER_STREAM
from src.credentials import FINNHUB_API_KEY


def count_pe_ratio(*, stocks: list) -> dict[str, float]:
    if type(stocks) is not list: stocks = [stocks]

    tickers = yf.Tickers(stocks)
    result: dict[str, float] = {}
    for key, val in tickers.tickers.items():
        if key == "": break
        current_price = float(tickers.tickers[key].info["currentPrice"])
        trailing_pe = float(tickers.tickers[key].info["trailingPE"])
        market_cap = float(tickers.tickers[key].info["marketCap"])
        count_shares = int(market_cap / current_price)
        result[key] = trailing_pe

    return result


def get_stocks_data(*, stocks, start, end):
    """
    TODO
    """
    stocks_data = pdr.DataReader(name=stocks, data_source='yahoo', start=start, end=end)
    LOGGER_STREAM.info(stocks_data)




def main(*, params: dict[str, str]):
    # stocks = ['AAPL', 'MSFT']
    # LOGGER_STREAM.info(count_pe_ratio(stocks=stocks)
    # get_stocks_data(stocks=stocks, start=date(2000, 1, 1), end=datetime.now())

    #
    # item is dict, e.g.: {"as": list(...)}
    for item in get_all_stocks(exchanges_filename=path.join(params['dataset_finnhub_dir'], 'finnhub_exchanges.csv')):
        key: str = list(item)[0]
        value: list = item[key]
        print(key)
        print(value[0])


    #


if __name__ == "__main__":
    if len(sys.argv) == 2 and path.isfile(sys.argv[1]):
        config_yaml = parse_yaml(filename=sys.argv[1])
        main(params=config_yaml)
        sys.exit(ExitCode.OK)
    else:
        LOGGER_STREAM.error('Bad script argument, expected *.yaml file')
        sys.exit(ExitCode.BAD_ARGS)
