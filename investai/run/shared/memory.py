# -*- coding: utf-8 -*-
from pathlib import Path

import attrs
import pandas as pd


@attrs.define
class Memory:
    """Memory class for storing the history of the agent performance in the environment"""
    df: pd.DataFrame

    def append(self, portfolio_value, portfolio_return, action, date):
        """Append memory
        :param portfolio_value: Portfolio value
        :param portfolio_return: Portfolio return
        :param action: Action
        :param date: Date
        """
        df_new = pd.DataFrame({
            "portfolio_value": [portfolio_value],
            "portfolio_return": [portfolio_return],
            "action": [action],
            "date": [date]
        })
        self.df = pd.concat([self.df, df_new], axis=0, ignore_index=True)

    @property
    def _initial_portfolio_value(self) -> int:
        """Initial portfolio value"""
        return self.df["portfolio_value"].iloc[0]

    @property
    def _current_portfolio_value(self) -> int:
        """Current portfolio value"""
        return self.df["portfolio_value"].iloc[-1]

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
