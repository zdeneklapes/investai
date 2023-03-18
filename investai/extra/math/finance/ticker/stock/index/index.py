# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
from shared.projectstructure import ProjectStructure


class Index:
    def dji30_tickers(self):
        pass

    def sp500_tickers(self):
        prj_dir = ProjectStructure(Path(__file__))
        SP500 = prj_dir.data.indexes.joinpath("SP500")
        SP500.mkdir(parents=True, exist_ok=True)
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)
        table[0].to_csv(SP500.joinpath("info.csv"))
        table[1].to_csv(SP500.joinpath("history.csv"))

    def nasdaq100_tickers():
        pass

    def hsi50_tickers(self):
        pass

    def sse100_tickers(self):
        pass

    def ftse100_tickers(self):
        pass

    def csi300_tickers(self):
        pass

    def cac40_tickers(self):
        pass

    def dax30_tickers(self):
        pass

    def mdax50_tickers(self):
        pass

    def sdax50_tickers(self):
        pass

    def lq45_tickers(self):
        pass

    def sri_kehati_tickers(self):
        pass

    def fx_tickers(self):
        pass


if __name__ == "__main__":
    index = Index()
    index.sp500_tickers()
