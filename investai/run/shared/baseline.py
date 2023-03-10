# -*- coding: utf-8 -*-
import attr

from rl.plot.plot import get_baseline, backtest_stats


@attr.define
class Baseline:
    ticker: str
    start: str
    end: str

    def __post_init__(self):
        self.baseline_df = get_baseline(self.ticker, self.start, self.end)
        self.stats = backtest_stats(self.baseline_df, value_col_name="close")

        print(self.baseline_df.index.min())
        print(self.baseline_df.index.max())
