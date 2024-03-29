# -*- coding: utf-8 -*-
from pathlib import Path
import dataclasses

import pandas as pd

from shared.program import Program


@dataclasses.dataclass(init=False)
class Memory:
    """Memory class for storing the history of the agent performance in the environment"""

    def __init__(self, program: Program, df: pd.DataFrame = pd.DataFrame()):
        self.program = program
        self.df = df

    def concat(self, memory: pd.DataFrame):
        """Append memory
        :param reward: Portfolio return
        :param action: Action
        :param date: Date
        """
        self.df = pd.concat([self.df, memory], axis=0, ignore_index=True)

    def to_datetime(self, column: str = "date", format: str = "%Y-%m-%d"):
        """Convert column to datetime"""
        self.df[column] = pd.to_datetime(self.df[column], format=format)

    def save_json(self, file_path: str):
        """Save df to csv file
        :param file_path: Path to save the df
        """
        if "i" in self.program.args.project_verbose: self.program.log.info(f"Saving df to: {file_path}")
        self.df.to_json(file_path, index=True)

    def load_json(self, file_path: Path):
        """Save df to csv file
        :param file_path: Path to save the df
        """
        if "i" in self.program.args.project_verbose: self.program.log.info(f"Loading df from: {file_path}")
        self.df = pd.read_json(file_path)
        # if "date" in self.df.columns: self.to_datetime()

    def save_csv(self, file_path: str):
        """Save df to csv file
        :param file_path: Path to save the df
        """
        if "i" in self.program.args.project_verbose: self.program.log.info(f"Saving df to: {file_path}")
        self.df.to_csv(file_path, index=True)

    def load_csv(self, file_path: str):
        """Save df to csv file
        :param file_path: Path to save the df
        """
        if "i" in self.program.args.project_verbose: self.program.log.info(f"Loading df from: {file_path}")
        self.df = pd.read_csv(file_path, index_col=0)
        # if "date" in self.df.columns: self.to_datetime()
