# -*- coding: utf-8 -*-
from pathlib import Path

import attrs
import pandas as pd


@attrs.define
class Memory:
    """Memory class for storing the history of the agent performance in the environment"""
    df: pd.DataFrame

    def append(self, reward, action, date):
        """Append memory
        :param reward: Portfolio return
        :param action: Action
        :param date: Date
        """
        df_new = pd.DataFrame({
            "reward": [reward],
            "action": [action],
            "date": [date]
        })
        self.df = pd.concat([self.df, df_new], axis=0, ignore_index=True)

    @property
    def _initial_portfolio_value(self) -> int:
        """Initial portfolio value"""
        return self.df["portfolio_value"].iloc[0]

    def save_memory(self, save_path: Path):
        """Save memory to csv file
        :param save_path: Path to save the memory
        """
        self.df.to_json(save_path.as_posix(), index=True)

    def load_memory(self, save_path: Path):
        """Save memory to csv file
        :param save_path: Path to save the memory
        """
        self.df.from_json(save_path.as_posix(), index=True)
