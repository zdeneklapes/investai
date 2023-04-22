# -*- coding: utf-8 -*-
import wandb
from itertools import chain
import inspect

from tqdm import tqdm
import pandas as pd
import empyrical  # noqa
import pyfolio as pf  # noqa
import seaborn as sns  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.axes._axes import Axes  # noqa
from matplotlib.axis import Axis  # noqa

from shared.program import Program
from shared.reload import reload_module  # noqa
from extra.math.finance.shared.baseline import Baseline  # noqa
from run.shared.memory import Memory

MARKET_OPEN_DAYS = 252


class Report(Memory):
    def __init__(self, program: Program = None):
        self.program = program
        self.filepath = self.program.args.history_path.as_posix()

    def download_test_history(self):
        if self.program.args.project_verbose > 0: self.program.log.info("START download_test_history")
        api = wandb.Api()
        runs = [
            api.runs(self.program.args.wandb_entity + "/" + self.program.args.wandb_project, filters={"group": group})
            for group in [
                # "sweep-nasfit-2",
                # "sweep-nasfit-3",
                # "sweep-nasfit-4"
                # "sweep-nasfit-5"
                "sweep-nasfit-6"
            ]
        ]
        log_keys = [
            "test/reward/^DJI",
            "test/reward/^GSPC",
            "test/reward/^IXIC",
            "test/reward/^RUT",
            "test/reward/maximum_sharpe_0_1",
            "test/reward/maximum_quadratic_utility_0_1",
            "test/reward/minimum_variance_0_1",
        ]

        #
        all_samples = 100_000
        iterations = tqdm(enumerate(list(chain(*runs))))
        final_df = pd.DataFrame()
        baseline_done = False

        # all rewards
        for idx, run in iterations:  # type: int,  wandb.apis.public.Run
            if run.state == "running" or run.state == "failed":
                continue

            # Baselines
            if not baseline_done:
                baseline_done = True
                for log_key in tqdm(log_keys):
                    key_history_df = run.history(samples=all_samples, keys=[log_key])
                    key_history_df.rename(columns={log_key: "reward"}, inplace=True)
                    key_history_df['id'] = run.id + "_" + log_key.split("/")[-1]
                    key_history_df['group'] = run._attrs['group']
                    key_history_df['algo'] = "baseline"
                    final_df = pd.concat([final_df, key_history_df])

            # Models
            model_key = "test/reward/model"
            key_history_df = run.history(samples=all_samples, keys=[model_key])
            key_history_df.rename(columns={model_key: "reward"}, inplace=True)
            key_history_df['id'] = run.id
            key_history_df['group'] = run._attrs['group']
            key_history_df['algo'] = run._attrs['config']['algo']
            final_df = pd.concat([final_df, key_history_df])

        #
        assert final_df.groupby(['id']).size().unique().size == 1, "All runs should have the same number of samples"
        final_df.to_csv(self.filepath, index=True)
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"History downloaded and saved to {self.filepath}")
        self.df = final_df
        if self.program.args.project_verbose > 0: self.program.log.info("End download_test_history")
        return self.df

    def initialize_stats(self):
        if self.program.args.project_verbose > 0: self.program.log.info("START initialize_stats")
        self.load_csv(self.filepath)
        self.returns_pivot_df = self.df.pivot(columns=['id'], values=['reward'])
        self.returns_pivot_df = pd.concat(
            [pd.DataFrame(data=[[0] * self.returns_pivot_df.columns.__len__()], columns=self.returns_pivot_df.columns),
             self.returns_pivot_df]).reset_index(
            drop=True)
        self.cumprod_returns_df = (self.returns_pivot_df + 1).cumprod()

        # TODO: Change index/_step by date
        self.baseline = Baseline(self.program)
        self.baseline.load_csv(self.program.args.baseline_path.as_posix())

        self.returns_pivot_df.index = pd.to_datetime(self.baseline.df['date'].iloc[-self.returns_pivot_df.shape[0]:],
                                                     format="%Y-%m-%d")
        self.cumprod_returns_df.index = pd.to_datetime(
            self.baseline.df['date'].iloc[-self.cumprod_returns_df.shape[0]:],
            format="%Y-%m-%d"
        )

        self.baseline_columns = self.returns_pivot_df.filter(regex="zy8buea3.*").columns.map(lambda x: x[1]).drop(
            'zy8buea3')

        self.returns_pivot_df.columns = self.returns_pivot_df.columns.droplevel(0)
        self.cumprod_returns_df.columns = self.cumprod_returns_df.columns.droplevel(0)
        self.cumprod_returns_df_without_baseline = self.cumprod_returns_df.drop(columns=self.baseline_columns)
        self.idx_min = self.cumprod_returns_df_without_baseline.iloc[-1].argmin()
        self.idx_max = self.cumprod_returns_df_without_baseline.iloc[-1].argmax()
        self.id_min = self.cumprod_returns_df_without_baseline.iloc[-1].index[self.idx_min]
        self.id_max = self.cumprod_returns_df_without_baseline.iloc[-1].index[self.idx_max]

        # TODO: Get config for idx_max
        # TODO: Get config for idx_min
        if self.program.args.project_verbose > 0: self.program.log.info("END initialize_stats")

    def stats(self):
        # Min Stats
        min_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df[self.id_min])
        min_stats['Beta'] = empyrical.beta(self.returns_pivot_df[self.id_min], self.returns_pivot_df['zy8buea3_^DJI'])
        min_stats.to_csv(self.program.args.folder_figure.joinpath("stats_min_baseline.csv"))

        # Max Stats
        max_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df[self.id_max])
        max_stats['Beta'] = empyrical.beta(self.returns_pivot_df[self.id_max], self.returns_pivot_df['zy8buea3_^DJI'])
        max_stats.to_csv(self.program.args.folder_figure.joinpath("stats_max_baseline.csv"))

    def returns_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info("START plot_returns")
        baselines_df = self.cumprod_returns_df[self.baseline_columns]

        # Min Cumulated returns
        returns_min_df = self.cumprod_returns_df[self.id_min]
        compare_min_baselines_df = pd.concat([returns_min_df, baselines_df], axis=1)
        _: Axes = sns.lineplot(data=compare_min_baselines_df)
        plt.savefig(self.program.args.folder_figure.joinpath("returns_min_baseline.png"))
        plt.clf()  # Clear the current figure

        # ##############################################
        # Max Cumulated returns
        returns_max_df = self.cumprod_returns_df[self.id_max]
        compare_max_baselines_df = pd.concat([returns_max_df, baselines_df], axis=1)
        _: Axes = sns.lineplot(data=compare_max_baselines_df)
        plt.savefig(self.program.args.folder_figure.joinpath("plot_returns_max_baseline.png"))
        plt.clf()  # Clear the current figure

        if self.program.args.project_verbose > 0: self.program.log.info("END plot_returns")

    def annual_returns_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info("START plot_annual_returns")
        # Min Annual returns
        pf.plot_annual_returns(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("annual_returns_min_model.png"))
        plt.clf()

        # Max Annual returns
        pf.plot_annual_returns(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("annual_returns_max_model.png"))
        plt.clf()

        if self.program.args.project_verbose > 0: self.program.log.info("END plot_annual_returns")

    def monthly_returns_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info("START plot_monthly_returns")
        # Min Monthly returns
        pf.plot_monthly_returns_heatmap(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_heatmap_min_model.png"))
        plt.clf()

        # Max Monthly returns
        pf.plot_monthly_returns_heatmap(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_heatmap_max_model.png"))
        plt.clf()

        if self.program.args.project_verbose > 0: self.program.log.info("END plot_monthly_returns")

    def monthly_return_heatmap_figure(self):
        # Min Monthly returns heatmap
        pf.plot_monthly_returns_heatmap(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_heatmap_min_baseline.png"))
        plt.clf()
        plt.cla()

        # Max Monthly returns heatmap
        pf.plot_monthly_returns_heatmap(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_heatmap_max_baseline.png"))
        plt.clf()
        plt.cla()

    def return_quantiles_figure(self):
        # Min Return quantiles
        pf.plot_return_quantiles(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("return_quantiles_min_baseline.png"))
        plt.clf()
        plt.cla()

        # Max Return quantiles
        pf.plot_return_quantiles(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("return_quantiles_max_baseline.png"))
        plt.clf()
        plt.cla()

    def rolling_beta_figure(self):
        # Min Rolling beta
        pf.plot_rolling_beta(self.returns_pivot_df[self.id_min], self.returns_pivot_df['zy8buea3_^DJI'])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_beta_min_baseline.png"))
        plt.clf()
        plt.cla()

        # Max Rolling beta
        pf.plot_rolling_beta(self.returns_pivot_df[self.id_min], self.returns_pivot_df['zy8buea3_^DJI'])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_beta_max_baseline.png"))
        plt.clf()
        plt.cla()

    def rolling_sharpe_figure(self):
        # Min Rolling sharpe
        pf.plot_rolling_sharpe(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_sharpe_min_baseline.png"))
        plt.gcf()
        plt.cla()

        # Max Rolling sharpe
        pf.plot_rolling_sharpe(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_sharpe_max_baseline.png"))
        plt.gcf()
        plt.cla()

    def drawdown_underwater_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

        # Min Drawdown underwater
        pf.plot_drawdown_underwater(self.returns_pivot_df[self.id_min])
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_underwater_min_baseline.png"))
        plt.clf()
        plt.cla()

        # Max Drawdown underwater
        pf.plot_drawdown_underwater(self.returns_pivot_df[self.id_max])
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_underwater_max_baseline.png"))
        plt.clf()
        plt.cla()
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def plot_drawdown(self):
        pass


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

    def monthly_returns_test(self): self.func_ret = self.wandbstats.monthly_returns_figure()

    def monthly_return_heatmap_test(self): self.func_ret = self.wandbstats.monthly_return_heatmap_figure()

    def return_quantiles_test(self): self.func_ret = self.wandbstats.return_quantiles_figure()

    def rolling_beta_test(self): self.func_ret = self.wandbstats.rolling_beta_figure()

    def rolling_sharpe_test(self): self.func_ret = self.wandbstats.rolling_sharpe_figure()

    def drawdown_underwater_test(self): self.func_ret = self.wandbstats.drawdown_underwater_figure()


def t():
    report_test = ReportTest()
    report_test.drawdown_underwater_test()
    return report_test


def main():
    program = Program()
    if program.args.project_verbose > 0: program.log.info("Start report")
    wandbstats = Report(program=program)
    if program.args.report_download_history: wandbstats.download_test_history()
    if program.args.report_figure:
        wandbstats.initialize_stats()
        wandbstats.stats()
        wandbstats.figure_returns()
        wandbstats.figure_details()
    if program.args.project_verbose > 0: program.log.info("End report")
    return None


if __name__ == "__main__":
    main()
