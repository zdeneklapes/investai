# -*- coding: utf-8 -*-
import dataclasses
from enum import Enum
from pathlib import Path

import pandas as pd
from meta.data_processors.yahoofinance import Yahoofinance

from configuration.settings import ProjectDir
from common.Args import Args
from common.utils import now_time
from rl.data.types import TimeInterval


@dataclasses.dataclass(init=False)
class DataBase(Yahoofinance):
    class DataType(Enum):
        TRAIN = "train"
        TEST = "test"

    def __init__(self, start_date: str, end_date: str, ticker_list: list = None, time_interval: TimeInterval = "1d"):
        super().__init__("yahoofinance", start_date, end_date, time_interval)
        self.ticker_list = ticker_list

    def save_dataset(self, df: pd.DataFrame):
        raise NotImplementedError("This method has bad implementation.")
        # filename = os.path.join(ProjectDir.DATASET.AI4FINANCE, f"dji30_ta_data_{now_time()}.json")
        # df.to_json(filename)
        # print(f"Saved dataset to {filename}")

    def get_filename(self, prj_dir: ProjectDir, name: str = "data") -> Path:
        filename = prj_dir.DATASET.AI4FINANCE.joinpath(f"{name}_{now_time()}.json")
        return filename

    def load_data(self, args: Args) -> pd.DataFrame:
        if args.input_dataset:
            self.dataframe = pd.read_json(args.input_dataset)
            # return pd.read_json(args.input_dataset)
        else:
            df = self.preprocess_data()  # pylint: disable=assignment-from-no-return
            if args.save_dataset:
                self.save_dataset(df)
            return df

    def data_split(self, df, start, end, target_date_col="date"):
        """
        Source AI4Finance
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def get_data_type(self, df: pd.DataFrame, data_type: DataType, date_divider: str) -> pd.DataFrame:
        ##
        if data_type == DataBase.DataType.TRAIN:
            return self.data_split(df, self.start_date, date_divider)
        elif data_type == DataBase.DataType.TEST:
            return self.data_split(df, date_divider, self.end_date)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def preprocess_data(self) -> pd.DataFrame:
        ...
