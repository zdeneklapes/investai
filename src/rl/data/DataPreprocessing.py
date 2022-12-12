# -*- coding: utf-8 -*-
import dataclasses

import pandas as pd
from meta.data_processors.yahoofinance import Yahoofinance

from common.Args import Args
from rl.data.types import TimeInterval


@dataclasses.dataclass(init=False)
class DataPreprocessing(Yahoofinance):
    def __init__(self, start_date: str, end_date: str, ticker_list: list = None, time_interval: TimeInterval = "1d"):
        super().__init__("yahoofinance", start_date, end_date, time_interval)
        self.ticker_list = ticker_list

    def retrieve_data(self, args: Args) -> pd.DataFrame:
        if args.input_dataset:
            return pd.read_csv(args.input_dataset)
        else:
            return self.preprocess_data()

    def preprocess_data(self) -> pd.DataFrame:
        ...
