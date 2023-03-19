# -*- coding: utf-8 -*-
from pathlib import Path

import attrs
import pandas as pd


@attrs.define
class Memory:
    """Memory class for storing the history of the agent performance in the environment"""

    df: pd.DataFrame

    def concat(self, memory: pd.DataFrame):
        """Append memory
        :param reward: Portfolio return
        :param action: Action
        :param date: Date
        """
        self.df = pd.concat([self.df, memory], axis=0, ignore_index=True)

    def save(self, file_path: str):
        """Save memory to csv file
        :param file_path: Path to save the memory
        """
        self.df.to_json(file_path, index=True)

    def load(self, file_path: Path):
        """Save memory to csv file
        :param file_path: Path to save the memory
        """
        self.df = pd.read_json(file_path)
