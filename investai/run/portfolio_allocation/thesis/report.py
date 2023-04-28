# -*- coding: utf-8 -*-
import inspect

import pandas as pd
import empyrical  # noqa
import pyfolio as pf  # noqa
import seaborn as sns  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.axes._axes import Axes  # noqa
from matplotlib.axis import Axis  # noqa
import numpy as np  # noqa

from run.portfolio_allocation.thesis.wandbapi import WandbAPI
from shared.program import Program
from shared.reload import reload_module  # noqa
from extra.math.finance.shared.baseline import Baseline  # noqa
from run.shared.memory import Memory

MARKET_OPEN_DAYS = 252


# TODO: Add comparison in the exact same date as AI4Finance
# TODO: Training and testing with the best data


class Report(Memory, WandbAPI):
    def __init__(self, program: Program):
        super().__init__(program)
        self.filepath = self.program.args.history_path.as_posix()

    def initialize_stats(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        self.load_csv(self.filepath)
        self.returns_pivot_df = self.df.pivot(columns=["id"], values=["reward"])
        self.returns_pivot_df = pd.concat(
            [pd.DataFrame(data=[[0] * self.returns_pivot_df.columns.__len__()], columns=self.returns_pivot_df.columns),
             self.returns_pivot_df]).reset_index(
            drop=True)
        self.cumprod_returns_df = (self.returns_pivot_df + 1).cumprod()
        self.baseline = Baseline(self.program)
        self.baseline.load_csv(self.program.args.baseline_path.as_posix())
        self.returns_pivot_df.index = pd.to_datetime(self.baseline.df["date"].iloc[-self.returns_pivot_df.shape[0]:],
                                                     format="%Y-%m-%d")
        self.cumprod_returns_df.index = pd.to_datetime(
            self.baseline.df["date"].iloc[-self.cumprod_returns_df.shape[0]:],
            format="%Y-%m-%d"
        )
        self.baseline_columns = ["^DJI", "^GSPC", "^IXIC", "^RUT", "maximum_sharpe_0_1", "minimum_variance_0_1", ]
        self.returns_pivot_df.columns = self.returns_pivot_df.columns.droplevel(0)
        self.cumprod_returns_df.columns = self.cumprod_returns_df.columns.droplevel(0)

        self.id_min = self.cumprod_returns_df.drop(self.baseline_columns, axis=1).iloc[-1].idxmin()
        self.id_max = self.cumprod_returns_df.drop(self.baseline_columns, axis=1).iloc[-1].idxmax()

        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def stats(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")

        # Min Stats
        min_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df[self.id_min])
        min_stats["Beta"] = empyrical.beta(self.returns_pivot_df[self.id_min],
                                           self.returns_pivot_df["^DJI"])
        min_stats["Alpha"] = empyrical.alpha(self.returns_pivot_df[self.id_min],
                                             self.returns_pivot_df["^DJI"])

        # Max Stats
        max_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df[self.id_max])
        max_stats["Beta"] = empyrical.beta(self.returns_pivot_df[self.id_max],
                                           self.returns_pivot_df["^DJI"])
        max_stats["Alpha"] = empyrical.alpha(self.returns_pivot_df[self.id_max],
                                             self.returns_pivot_df["^DJI"])

        # DJI Stats
        dji_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df["^DJI"])
        dji_stats["Beta"] = empyrical.beta(self.returns_pivot_df["^DJI"],
                                           self.returns_pivot_df["^DJI"])
        dji_stats["Alpha"] = empyrical.alpha(self.returns_pivot_df["^DJI"],
                                             self.returns_pivot_df["^DJI"])

        # GSPC Stats
        gspc_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df["^GSPC"])
        gspc_stats["Beta"] = empyrical.beta(self.returns_pivot_df["^GSPC"],
                                            self.returns_pivot_df["^DJI"])
        gspc_stats["Alpha"] = empyrical.alpha(self.returns_pivot_df["^GSPC"],
                                              self.returns_pivot_df["^DJI"])

        # IXIC Stats
        ixic_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df["^IXIC"])
        ixic_stats["Beta"] = empyrical.beta(self.returns_pivot_df["^IXIC"],
                                            self.returns_pivot_df["^DJI"])
        ixic_stats["Alpha"] = empyrical.alpha(self.returns_pivot_df["^IXIC"],
                                              self.returns_pivot_df["^DJI"])

        # RUT Stats
        rut_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df["^RUT"])
        rut_stats["Beta"] = empyrical.beta(self.returns_pivot_df["^RUT"],
                                           self.returns_pivot_df["^DJI"])
        rut_stats["Alpha"] = empyrical.alpha(self.returns_pivot_df["^RUT"],
                                             self.returns_pivot_df["^DJI"])

        # Maximum Sharpe ratio
        max_sharpe_ratio: pd.Series = pf.timeseries.perf_stats(
            self.returns_pivot_df["maximum_sharpe_0_1"])
        max_sharpe_ratio["Beta"] = empyrical.beta(self.returns_pivot_df["maximum_sharpe_0_1"],
                                                  self.returns_pivot_df["^DJI"])
        max_sharpe_ratio["Alpha"] = empyrical.alpha(self.returns_pivot_df["maximum_sharpe_0_1"],
                                                    self.returns_pivot_df["^DJI"])

        # Minimum variance
        min_variance: pd.Series = pf.timeseries.perf_stats(
            self.returns_pivot_df["minimum_variance_0_1"])
        min_variance["Beta"] = empyrical.beta(self.returns_pivot_df["minimum_variance_0_1"],
                                              self.returns_pivot_df["^DJI"])
        min_variance["Alpha"] = empyrical.alpha(self.returns_pivot_df["minimum_variance_0_1"],
                                                self.returns_pivot_df["^DJI"])

        # AI4Finance Stats
        ai4finance_stats: pd.Series = pd.Series({  # noqa
            "Annual return": 0.09,
            "Cumulative returns": None,
            "Annual volatility": 0.232,
            "Sharpe ratio": 0.49,
            "Calmar ratio": 0.24,
            "Stability": 0.04,
            "Max drawdown": -0.375,
            "Omega ratio": 1.14,
            "Sortino ratio": 0.67,
            "Skew": None,
            "Kurtosis": None,
            "Tail ratio": 1.03,
            "Daily value at risk": -0.028,
            "Alpha": 0.03,
            "Beta": 0.68,
        })

        # Stats All
        stats = pd.concat(
            [min_stats, max_stats, dji_stats, gspc_stats, ixic_stats, rut_stats, max_sharpe_ratio, min_variance],
            axis=1)
        stats.columns = [self.id_min, self.id_max, "^DJI", "^GSPC", "^IXIC", "^RUT", "Max Sharpe Ratio", "Min Variance"]
        stats.rename(index={"Cumulative returns": "Cum. returns"}, inplace=True)
        stats = stats.round(3)
        stats.to_csv(self.program.args.folder_figure.joinpath("stats.csv"))

        # Remove duplicate values
        for i, row in stats.iterrows():
            duplicate_mask = row.duplicated()
            if any(duplicate_mask):
                stats.loc[i, duplicate_mask] += 0.001

        styler: pd.io.formats.style.Styler = stats.style

        # styler.highlight_max(axis=1, props="color:#00F000; font-weight: bold ;")
        def highlight_max(s: pd.Series):
            val_max = s.max()
            return ["color: #00F000; font-weight: bold" if v == val_max else "" for v in s.values]

        styler.apply(highlight_max, axis=1)
        styler.applymap_index(lambda v: "font-weight: bold;", axis="index")
        styler.applymap_index(lambda v: "font-weight: bold;", axis="columns")
        latex = styler.to_latex(
            column_format="*{9}{|m{0.08\\linewidth}|}",
            caption="Performance metrics of the models vs. indexes and strategies",
            hrules=True,
            position="ht!",
            position_float="centering",
            convert_css=True,
            label="tab:stats",
        )
        latex = latex.replace(r"\\", r"\\[0.5cm]")
        latex = latex.replace(r"^", r"\^")
        print(latex)

        # Stats RL vs. AI4Finance

        # # Stats RL
        # stats = pd.concat([min_stats, max_stats, ai4finance_stats, ], axis=1)
        # stats.columns = ["The lowest Total Reward", "The highest Total Reward", "AI4Finance"]
        # stats.rename(index={"Cumulative returns": "Cum. returns"}, inplace=True)
        # stats.round(3).to_csv(self.program.args.folder_figure.joinpath("stats_models.csv"))
        #
        # # Stategies and indexes
        # stats = pd.concat([dji_stats, gspc_stats, ixic_stats, rut_stats, max_sharpe_ratio, min_variance], axis=1)
        # stats.columns = ["^DJI", "^GSPC", "^IXIC", "^RUT", "Max Sharpe Ratio", "Min Variance"]
        # stats.rename(index={"Cumulative returns": "Cum. returns"}, inplace=True)
        # stats.round(3).to_csv(self.program.args.folder_figure.joinpath("stats_stategies.csv"))

        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def returns_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        baselines_df = self.cumprod_returns_df[self.baseline_columns]

        returns_min_df = self.cumprod_returns_df[self.id_min]
        returns_max_df = self.cumprod_returns_df[self.id_max]

        plot_df = pd.concat([returns_min_df, returns_max_df, baselines_df], axis=1)
        _: Axes = sns.lineplot(data=plot_df)
        plt.savefig(self.program.args.folder_figure.joinpath("returns.png"))
        plt.clf()  # Clear the current figure
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def annual_returns_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        # Min Annual returns
        pf.plot_annual_returns(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("annual_returns_min.png"))
        plt.clf()

        # Max Annual returns
        pf.plot_annual_returns(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("annual_returns_max.png"))
        plt.clf()

        # DJI Annual returns
        pf.plot_annual_returns(self.returns_pivot_df[f"{self.id_baseline}_^DJI"])
        plt.savefig(self.program.args.folder_figure.joinpath("annual_returns_dji.png"))
        plt.clf()

        # IXIC Annual returns
        pf.plot_annual_returns(self.returns_pivot_df[f"{self.id_baseline}_^IXIC"])
        plt.savefig(self.program.args.folder_figure.joinpath("annual_returns_ixic.png"))
        plt.clf()

        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def monthly_return_heatmap_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        # Min Monthly returns heatmap
        pf.plot_monthly_returns_heatmap(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_heatmap_min.png"))
        plt.clf()
        plt.cla()

        # Max Monthly returns heatmap
        pf.plot_monthly_returns_heatmap(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_heatmap_max.png"))
        plt.clf()
        plt.cla()

        # DJI Monthly returns heatmap
        pf.plot_monthly_returns_heatmap(self.returns_pivot_df[f"{self.id_baseline}_^DJI"])
        plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_heatmap_dji.png"))
        plt.clf()
        plt.cla()

        # IXIC Monthly returns heatmap
        pf.plot_monthly_returns_heatmap(self.returns_pivot_df[f"{self.id_baseline}_^IXIC"])
        plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_heatmap_ixic.png"))
        plt.clf()
        plt.cla()
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def return_quantiles_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        # Min Return quantiles
        pf.plot_return_quantiles(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("return_quantiles_min.png"))
        plt.clf()
        plt.cla()

        # Max Return quantiles
        pf.plot_return_quantiles(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("return_quantiles_max.png"))
        plt.clf()
        plt.cla()

        # DJI Return quantiles
        pf.plot_return_quantiles(self.returns_pivot_df[f"{self.id_baseline}_^DJI"])
        plt.savefig(self.program.args.folder_figure.joinpath("return_quantiles_dji.png"))
        plt.clf()
        plt.cla()
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def rolling_beta_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        # Min Rolling beta
        pf.plot_rolling_beta(self.returns_pivot_df[self.id_min], self.returns_pivot_df[f"{self.id_baseline}_^DJI"])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_beta_min.png"))
        plt.clf()
        plt.cla()

        # Max Rolling beta
        pf.plot_rolling_beta(self.returns_pivot_df[self.id_max], self.returns_pivot_df[f"{self.id_baseline}_^DJI"])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_beta_max.png"))
        plt.clf()
        plt.cla()

        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def rolling_sharpe_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        # Min Rolling sharpe
        pf.plot_rolling_sharpe(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_sharpe_min.png"))
        plt.gcf()
        plt.cla()

        # Max Rolling sharpe
        pf.plot_rolling_sharpe(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_sharpe_max.png"))
        plt.gcf()
        plt.cla()

        # DJI Rolling sharpe
        pf.plot_rolling_sharpe(self.returns_pivot_df[f"{self.id_baseline}_^DJI"])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_sharpe_dji.png"))
        plt.gcf()
        plt.cla()
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def drawdown_underwater_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")

        # Min Drawdown underwater
        pf.plot_drawdown_underwater(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_underwater_min.png"))
        plt.clf()
        plt.cla()

        # Max Drawdown underwater
        pf.plot_drawdown_underwater(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_underwater_max.png"))
        plt.clf()
        plt.cla()

        # DJI Drawdown underwater
        pf.plot_drawdown_underwater(self.returns_pivot_df[f"{self.id_baseline}_^DJI"])
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_underwater_dji.png"))
        plt.clf()
        plt.cla()

        # GSPC
        pf.plot_drawdown_underwater(self.returns_pivot_df[f"{self.id_baseline}_^IXIC"])
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_underwater_ixic.png"))
        plt.clf()
        plt.cla()
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def drawdown_periods_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")

        # Min Drawdown periods
        pf.plot_drawdown_periods(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_periods_min.png"))
        plt.clf()
        plt.cla()

        # Max Drawdown periods
        pf.plot_drawdown_periods(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_periods_max.png"))
        plt.clf()
        plt.cla()

        # DJI Drawdown periods
        pf.plot_drawdown_periods(self.returns_pivot_df[f"{self.id_baseline}_^DJI"])
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_periods_dji.png"))
        plt.clf()
        plt.cla()
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")


class ReportTest:
    def __init__(self):
        self.program = Program()
        self.program.args.history_path = self.program.args.folder_model.joinpath("history.csv")
        self.program.args.baseline_path = self.program.args.folder_baseline.joinpath("baseline.csv")
        self.wandbstats = Report(program=self.program)
        self.wandbstats.initialize_stats()

    def stats_test(self): self.func_ret = self.wandbstats.stats()

    def returns_figure_test(self): self.func_ret = self.wandbstats.returns_figure()

    def annual_returns_figure_test(self): self.func_ret = self.wandbstats.annual_returns_figure()

    def monthly_return_heatmap_test(self): self.func_ret = self.wandbstats.monthly_return_heatmap_figure()

    def return_quantiles_test(self): self.func_ret = self.wandbstats.return_quantiles_figure()

    def rolling_beta_test(self): self.func_ret = self.wandbstats.rolling_beta_figure()

    def rolling_sharpe_test(self): self.func_ret = self.wandbstats.rolling_sharpe_figure()

    def drawdown_underwater_test(self): self.func_ret = self.wandbstats.drawdown_underwater_figure()

    def drawdown_periods_test(self): self.func_ret = self.wandbstats.drawdown_periods_figure()


def t():
    report_test = ReportTest()
    report_test.drawdown_periods_test()
    return report_test


def main():
    program: Program = Program()
    if program.args.project_verbose > 0: program.log.info("Start report")
    report: Report = Report(program=program)
    if program.args.report_download_history: report.download_test_history()
    if program.args.report_figure:
        report.initialize_stats()
        report.stats()
        # report.returns_figure()
        # report.annual_returns_figure()
        # report.monthly_return_heatmap_figure()
        # report.return_quantiles_figure()
        # report.rolling_beta_figure()
        # report.rolling_sharpe_figure()
        # report.drawdown_underwater_figure()
        # wandbstats.drawdown_periods_figure() # TODO: Fix this
    if program.args.project_verbose > 0: program.log.info("End report")
    return None


if __name__ == "__main__":
    main()
    # t()
