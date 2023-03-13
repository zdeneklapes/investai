# -*- coding: utf-8 -*-

import pandas as pd

from project_configs.project_dir import ProjectDir


def dji30_tickers(): pass


def sp500_tickers():
    prj_dir = ProjectDir(__file__)
    SP500 = prj_dir.data.indexes.joinpath("SP500")
    SP500.mkdir(parents=True, exist_ok=True)
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    table[0].to_csv(SP500.joinpath("info.csv"))
    table[1].to_csv(SP500.joinpath("history.csv"))


def nasdaq100_tickers(): pass


def hsi50_tickers(): pass


def sse100_tickers(): pass


def ftse100_tickers(): pass


def csi300_tickers(): pass


def cac40_tickers(): pass


def dax30_tickers(): pass


def mdax50_tickers(): pass


def sdax50_tickers(): pass


def lq45_tickers(): pass


def sri_kehati_tickers(): pass


def fx_tickers(): pass


if __name__ == "__main__":
    sp500_tickers()
