# -*- coding: utf-8 -*-
import os
import dataclasses
from pathlib import Path

import tqdm
import pandas as pd
from meta.data_processors.yahoofinance import Yahoofinance

from project_configs.project_dir import ProjectDir
from shared.Args import Args
from shared.utils import now_time


@dataclasses.dataclass(init=False)
class DataBase(Yahoofinance):
    from rl.data.types import TimeInterval, FileType, DataType

    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        ticker_list: list = None,
        time_interval: TimeInterval = "1d",
    ):
        super().__init__(data_source, start_date, end_date, time_interval)
        self.ticker_list = ticker_list

    def check_dir_exists(self, _filename: Path):
        directory = _filename.parent
        if os.path.isdir(directory):
            print(f"OK: Dir exist for: {_filename}")
        else:
            raise NotADirectoryError(f"ERROR: Dir does not exist {directory}")

    def get_file_type(self, filename: Path) -> FileType:
        from rl.data.types import FileType

        if filename.suffix == ".json":
            return FileType.JSON
        elif filename.suffix == ".csv":
            return FileType.CSV
        else:
            raise ValueError(f"Unknown file type: {filename}")

    def save_dataset(self, df: pd.DataFrame, _filename: Path):
        from rl.data.types import FileType

        self.check_dir_exists(_filename)

        print(f"Saving raw_data into: {_filename}")

        ##
        if self.get_file_type(_filename) == FileType.JSON:
            df.to_json(_filename.as_posix())
        elif self.get_file_type(_filename) == FileType.CSV:
            df.to_csv(_filename)
        else:
            raise ValueError(f"type: FileType must be {FileType.list()}")

        print(f"Data saved to json: {_filename}")

    def get_filename(self, prj_dir: ProjectDir, name: str = "raw_data") -> Path:
        filename = prj_dir.data.ai4finance.joinpath(f"{name}_{now_time()}.json")
        return filename

    def load_data(self, args: Args) -> pd.DataFrame:
        if args.input_dataset:
            self.dataframe = pd.read_json(args.input_dataset)
        else:
            raise ValueError("No input dataset provided")
        # else:
        #     df = self.preprocess_data()  # pylint: disable=assignment-from-no-return
        #     if args.save_dataset:
        #         self.save_dataset(df)
        #     return df

    def data_split(self, df, start, end, target_date_col="date"):
        """
        Source AI4Finance
        split the dataset into training or testing using date
        :param raw_data: (df) pandas dataframe, start, end
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
            raise ValueError(f"Unknown raw_data type: {data_type}")

    def preprocess_data(self) -> pd.DataFrame:
        ...


def get_missing_counts(df: pd.DataFrame):
    grouped_by_df = df.groupby(by=["date"])
    grouped_by_size = grouped_by_df.size()
    counts = pd.Series([])
    for i in grouped_by_size.unique():
        occurences = (grouped_by_size == i).sum()
        counts = counts.append(pd.Series(occurences))
        print(f"{i} : {occurences}")


class MissingDataHandler:
    def __init__(self):
        pass

    def get_missing_tic(self, df: pd.DataFrame, date: str, _tickers: list) -> list:
        df_date = df[df["date"] == date]
        missing_tics: set = set(_tickers).difference(set(df_date["tic"].values))
        missing_tics: list = list(missing_tics)
        return [missing_tics]

    def get_previous_value(self, df: pd.DataFrame, date: str, missing_tic: str):
        df_before_date_binary = df["date"] < date
        df_before_date = df[df_before_date_binary]
        previous_value_of_missing_tic = df_before_date["tic"] == missing_tic[0]
        df_missing_tic_previous_value = df_before_date[previous_value_of_missing_tic]
        if df_missing_tic_previous_value.empty:
            return None
        else:
            return df_missing_tic_previous_value.iloc[-1]

    def get_following_value(self, df: pd.DataFrame, date: str, missing_tic: str):
        df_before_date_binary = df["date"] > date
        df_before_date = df[df_before_date_binary]
        previous_value_of_missing_tic_binary = df_before_date["tic"] == missing_tic[0]
        df_missing_tic_previous_value = df_before_date[previous_value_of_missing_tic_binary]
        if df_missing_tic_previous_value.empty:
            return None
        else:
            return df_missing_tic_previous_value.iloc[0]

    def fill_missed_tic_gaps(self, df: pd.DataFrame, _tickers: list) -> pd.DataFrame:
        grouped_by_df = df.groupby(by=["date"])
        grouped_by_size = grouped_by_df.size()
        max_size = grouped_by_size.max()
        # print(grouped_by_size.unique())

        # for each date where some tics are missed
        for date in tqdm.tqdm(grouped_by_size[grouped_by_size < max_size].index):
            for tic in self.get_missing_tic(df, date, _tickers):  # TODO: Improve performance
                # TODO: Improve performance
                filled_value = self.get_previous_value(df, date, missing_tic=tic)
                filled_value = (
                    filled_value if filled_value is not None else self.get_following_value(df, date, missing_tic=tic)
                )

                #
                if filled_value is None:
                    raise ValueError("None value of filling value")
                else:
                    filled_value["date"] = date  # update date we want to fill the gap
                    df = df.append(filled_value)  # TODO: Improve performance

        return df


def get_count_miss_tics(df: pd.DataFrame):
    grouped_by_df = df.groupby(by=["date"])
    grouped_by_size = grouped_by_df.size()
    counts = pd.Series([])
    for i in grouped_by_size.unique():
        occurences = (grouped_by_size == i).sum()
        counts = counts.append(pd.Series(occurences))
        print(f"{i} : {occurences}")


def print_info(df: pd.DataFrame):
    ##
    print(f"{df.shape=}")
    print(f"Start date: {df['date'].iloc[0]}")
    print(f"End date: {df['date'].iloc[-1]}")

    print(f"Stocks: {df['tic'].unique().size}")

    ##
    grouped_data_size = df.groupby(by=["date"]).size()
    if grouped_data_size.nunique() == 1:
        print(f"OK: All dates have same row count: {grouped_data_size.unique()}")
    else:
        print(f"ERROR: Different size for each date: {grouped_data_size.unique()}")
