# -*- coding: utf-8 -*-
import wandb
from itertools import chain

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

    def get_summary(self):
        if self.program.args.project_verbose > 0: self.program.log.info("START get_summary")
        # Sharpe
        print("Sharpe")
        print(empyrical.sharpe_ratio(self.returns_pivot_df[self.id_min]))
        print(empyrical.sharpe_ratio(self.returns_pivot_df[self.id_max]))

        # Beta
        print("Beta")
        print(empyrical.beta(self.returns_pivot_df[self.id_min], self.returns_pivot_df['zy8buea3_^DJI']))
        print(empyrical.beta(self.returns_pivot_df[self.id_max], self.returns_pivot_df['zy8buea3_^DJI']))

        # Max drawdown
        print("Max drawdown")
        print(empyrical.max_drawdown(self.returns_pivot_df[self.id_min]))
        print(empyrical.max_drawdown(self.returns_pivot_df[self.id_max]))
        if self.program.args.project_verbose > 0: self.program.log.info("END get_summary")

    def plot_returns(self):
        if self.program.args.project_verbose > 0: self.program.log.info("START plot_returns")
        baselines_df = self.cumprod_returns_df[self.baseline_columns]

        # Min Cumulated returns
        returns_min_df = self.cumprod_returns_df[self.id_min]
        compare_min_baselines_df = pd.concat([returns_min_df, baselines_df], axis=1)
        _: Axes = sns.lineplot(data=compare_min_baselines_df)
        plt.savefig(self.program.args.folder_figure.joinpath("returns_min_baseline.png"))
        plt.clf()  # Clear the current figure

        # Min Stats
        min_stats: pd.Series = pf.show_perf_stats(self.returns_pivot_df[self.id_min])
        min_stats.to_csv(self.program.args.folder_figure.joinpath("stats_min_baseline.csv"))

        # ##############################################
        # Max Cumulated returns
        returns_max_df = self.cumprod_returns_df[self.id_max]
        compare_max_baselines_df = pd.concat([returns_max_df, baselines_df], axis=1)
        _: Axes = sns.lineplot(data=compare_max_baselines_df)
        plt.savefig(self.program.args.folder_figure.joinpath("plot_returns_max_baseline.png"))
        plt.clf()  # Clear the current figure

        # Max Stats
        max_stats: pd.Series = pf.show_perf_stats(self.returns_pivot_df[self.id_max])
        max_stats.to_csv(self.program.args.folder_figure.joinpath("stats_max_baseline.csv"))

        if self.program.args.project_verbose > 0: self.program.log.info("END plot_returns")

    def plot_details(self):
        if self.program.args.project_verbose > 0: self.program.log.info("START plot_details")

        # Annual returns
        # pf.plot_annual_returns(returns_pivot_df[id_min])
        # plt.savefig(self.program.args.folder_figure.joinpath("annual_returns_min_baseline.png"))

        # Monthly returns
        # pf.plot_monthly_returns_dist(returns_pivot_df[id_min])
        # plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_dist_min_baseline.png"))

        # Monthly returns heatmap
        # pf.plot_monthly_returns_heatmap(returns_pivot_df[id_min])
        # plt.savefig(self.program.args.folder_figure.joinpath("monthly_returns_heatmap_min_baseline.png"))

        # Return quantiles
        # pf.plot_return_quantiles(returns_pivot_df[id_min])
        # plt.savefig(self.program.args.folder_figure.joinpath("return_quantiles_min_baseline.png"))

        # Rolling beta
        # pf.plot_rolling_beta(returns_pivot_df[id_min], returns_pivot_df['zy8buea3_^DJI'])
        # plt.savefig(self.program.args.folder_figure.joinpath("rolling_beta_min_baseline.png"))

        # Rolling sharpe
        # pf.plot_rolling_sharpe(returns_pivot_df[id_min])
        # plt.savefig(self.program.args.folder_figure.joinpath("rolling_sharpe_min_baseline.png"))

        # Drawdown underwater
        # pf.plot_drawdown_underwater(returns_pivot_df[id_min])
        # plt.savefig(self.program.args.folder_figure.joinpath("drawdown_underwater_min_baseline.png"))

        if self.program.args.project_verbose > 0: self.program.log.info("END plot_details")

    def plot_drawdown(self):
        pass


class ReportTest:
    def __init__(self):
        self.program = Program()
        self.program.args.history_path = self.program.args.folder_model.joinpath("history.csv")
        self.program.args.baseline_path = self.program.args.folder_baseline.joinpath("baseline.csv")
        self.wandbstats = Report(program=self.program)

    def plot_returns_test(self):
        self.wandbstats.initialize_stats()
        return {"program": self.program,
                "wandbstats": self.wandbstats,
                "plot": self.wandbstats.plot_returns(), }


def main():
    program = Program()
    if program.args.project_verbose > 0: program.log.info("Start report")
    wandbstats = Report(program=program)
    if program.args.report_download_history: wandbstats.download_test_history()
    if program.args.report_figure:
        wandbstats.initialize_stats()
        wandbstats.plot_returns()
        # wandbstats.plot_details()
    if program.args.project_verbose > 0: program.log.info("End report")
    return None


if __name__ == "__main__":
    main()
