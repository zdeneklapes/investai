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
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
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
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")
        return self.df

    def initialize_stats(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        self.load_csv(self.filepath)
        self.returns_pivot_df = self.df.pivot(columns=['id'], values=['reward'])
        self.returns_pivot_df = pd.concat(
            [pd.DataFrame(data=[[0] * self.returns_pivot_df.columns.__len__()], columns=self.returns_pivot_df.columns),
             self.returns_pivot_df]).reset_index(
            drop=True)
        self.cumprod_returns_df = (self.returns_pivot_df + 1).cumprod()
        self.baseline = Baseline(self.program)
        self.baseline.load_csv(self.program.args.baseline_path.as_posix())
        self.returns_pivot_df.index = pd.to_datetime(self.baseline.df['date'].iloc[-self.returns_pivot_df.shape[0]:],
                                                     format="%Y-%m-%d")
        self.cumprod_returns_df.index = pd.to_datetime(
            self.baseline.df['date'].iloc[-self.cumprod_returns_df.shape[0]:],
            format="%Y-%m-%d"
        )
        self.id_baseline = "fi9bu9a5"
        self.baseline_columns = self.returns_pivot_df.filter(
            regex="fi9bu9a5.*").columns.map(lambda x: x[1]).drop(
            [self.id_baseline, f"{self.id_baseline}_maximum_quadratic_utility_0_1"]
        )
        b_columns = self.baseline_columns.append(pd.Index([f"{self.id_baseline}_maximum_quadratic_utility_0_1"]))
        self.returns_pivot_df.columns = self.returns_pivot_df.columns.droplevel(0)
        self.cumprod_returns_df.columns = self.cumprod_returns_df.columns.droplevel(0)
        self.cumprod_returns_df_without_baseline = self.cumprod_returns_df.drop(columns=b_columns)
        self.idx_min = self.cumprod_returns_df_without_baseline.iloc[-1].argmin()
        self.idx_max = self.cumprod_returns_df_without_baseline.iloc[-1].argmax()
        self.id_min = self.cumprod_returns_df_without_baseline.iloc[-1].index[self.idx_min]
        self.id_max = self.cumprod_returns_df_without_baseline.iloc[-1].index[self.idx_max]
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def stats(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        # Min Stats
        min_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df[self.id_min])
        min_stats['Beta'] = empyrical.beta(self.returns_pivot_df[self.id_min],
                                           self.returns_pivot_df[f'{self.id_baseline}_^DJI'])
        min_stats.to_csv(self.program.args.folder_figure.joinpath("stats_min.csv"))

        # Max Stats
        max_stats: pd.Series = pf.timeseries.perf_stats(self.returns_pivot_df[self.id_max])
        max_stats['Beta'] = empyrical.beta(self.returns_pivot_df[self.id_max],
                                           self.returns_pivot_df[f'{self.id_baseline}_^DJI'])
        max_stats.to_csv(self.program.args.folder_figure.joinpath("stats_max.csv"))
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def returns_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        baselines_df = self.cumprod_returns_df[self.baseline_columns]

        # Min Cumulated returns
        returns_min_df = self.cumprod_returns_df[self.id_min]
        compare_min_baselines_df = pd.concat([returns_min_df, baselines_df], axis=1)
        _: Axes = sns.lineplot(data=compare_min_baselines_df)
        plt.savefig(self.program.args.folder_figure.joinpath("returns_min.png"))
        plt.clf()  # Clear the current figure

        # Max Cumulated returns
        returns_max_df = self.cumprod_returns_df[self.id_max]
        compare_max_baselines_df = pd.concat([returns_max_df, baselines_df], axis=1)
        _: Axes = sns.lineplot(data=compare_max_baselines_df)
        plt.savefig(self.program.args.folder_figure.joinpath("returns_max.png"))
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
        plt.savefig(self.program.args.folder_figure.joinpath("return_quantiles_max.png"))
        plt.clf()
        plt.cla()
        if self.program.args.project_verbose > 0: self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")

    def rolling_beta_figure(self):
        if self.program.args.project_verbose > 0: self.program.log.info(
            f"START {inspect.currentframe().f_code.co_name}")
        # Min Rolling beta
        pf.plot_rolling_beta(self.returns_pivot_df[self.id_min], self.returns_pivot_df[f'{self.id_baseline}_^DJI'])
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_beta_min.png"))
        plt.clf()
        plt.cla()

        # Max Rolling beta
        pf.plot_rolling_beta(self.returns_pivot_df[self.id_max], self.returns_pivot_df[f'{self.id_baseline}_^DJI'])
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
        plt.savefig(self.program.args.folder_figure.joinpath("rolling_sharpe_max.png"))
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
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_underwater_max.png"))
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
        plt.savefig(self.program.args.folder_figure.joinpath("drawdown_periods_max.png"))
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
    program = Program()
    if program.args.project_verbose > 0: program.log.info("Start report")
    wandbstats = Report(program=program)
    if program.args.report_download_history: wandbstats.download_test_history()
    if program.args.report_figure:
        wandbstats.initialize_stats()
        wandbstats.stats()
        wandbstats.returns_figure()
        wandbstats.annual_returns_figure()
        wandbstats.monthly_return_heatmap_figure()
        wandbstats.return_quantiles_figure()
        wandbstats.rolling_beta_figure()
        wandbstats.rolling_sharpe_figure()
        wandbstats.drawdown_underwater_figure()
        # wandbstats.drawdown_periods_figure()
    if program.args.project_verbose > 0: program.log.info("End report")
    return None


if __name__ == "__main__":
    main()
    # t()
