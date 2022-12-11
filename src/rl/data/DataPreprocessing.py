# -*- coding: utf-8 -*-
from typing import Callable
import dataclasses

import pandas as pd
from meta.data_processors.yahoofinance import Yahoofinance

from common.Args import Args
from rl.data.types import TimeInterval


@dataclasses.dataclass(init=False)
class DataPreprocessing(Yahoofinance):
    def __init__(
        self,
        start_date: str,
        end_date: str,
        cb_readfile: Callable = pd.read_csv,
        ticker_list: list = None,
        time_interval: TimeInterval = "1d",
    ):
        super().__init__("yahoofinance", start_date, end_date, time_interval)
        self.cb_readfile: Callable = cb_readfile
        self.ticker_list = ticker_list

    def save_preprocessed_data(self, df: pd.DataFrame, filepath: str) -> None:
        if not df or not filepath:
            raise ValueError("df, filepath can't be None")

        df.to_csv(filepath)

    def load_data(self, filepath: str) -> pd.DataFrame:
        return self.cb_readfile(filepath)

    def retrieve_data(self, args: Args) -> pd.DataFrame:
        if args.input_dataset:
            self.load_data(args.input_dataset)
        else:
            return self.preprocess_data()

    def preprocess_data(self) -> pd.DataFrame:
        ...
