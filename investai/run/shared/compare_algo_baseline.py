# -*- coding: utf-8 -*-
from typing import List

from rl.examples.baseline import Baseline
from rl.examples.learned_algorithm import LearnedAlgorithm
from rl.plot.plot import backtest_plot


class CompareAlgoBaseline:
    def __init__(self, algos: List[LearnedAlgorithm], baseline: Baseline):
        self.algos: List[LearnedAlgorithm] = algos
        self.baseline: Baseline = baseline

    def plot_baseline(self):
        backtest_plot(
            self.algos[0].df_account_value,
            baseline_ticker="SPY",
            baseline_start=self.baseline.start,
            baseline_end=self.baseline.end,
            value_col_name="account_value",
        )
