# -*- coding: utf-8 -*-
import os
import dataclasses

import pandas as pd
from meta.data_processors.yahoofinance import Yahoofinance

from configuration.settings import ProjectDir
from common.Args import Args
from common.utils import now_time
from rl.data.types import TimeInterval


@dataclasses.dataclass(init=False)
class DataPreprocessing(Yahoofinance):
    def __init__(self, start_date: str, end_date: str, ticker_list: list = None, time_interval: TimeInterval = "1d"):
        super().__init__("yahoofinance", start_date, end_date, time_interval)
        self.ticker_list = ticker_list

    def save_dataset(self, df: pd.DataFrame):
        filename = os.path.join(ProjectDir.DATASET.AI4FINANCE, f"dji30_ta_data_{now_time()}.json")
        df.to_json(filename)
        print(f"Saved dataset to {filename}")

    def retrieve_data(self, args: Args) -> pd.DataFrame:
        if args.input_dataset:
            return pd.read_json(args.input_dataset)
        else:
            df = self.preprocess_data()  # pylint: disable=assignment-from-no-return
            if args.save_dataset:
                self.save_dataset(df)
            return df

    def preprocess_data(self) -> pd.DataFrame:
        ...
