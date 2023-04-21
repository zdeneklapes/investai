# -*- coding: utf-8 -*-
import wandb
from itertools import chain
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import empyrical
import matplotlib.pyplot as plt
import pyfolio as pf

from shared.program import Program
from shared.reload import reload_module  # noqa


class Reload:
    def __init__(self, program: Program = None):
        self.program = program
        self.filepath = self.program.args.folder_model.joinpath("wandb_history.csv")

    def download_test_history(self):
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
        runs_with_history = defaultdict(dict)
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
                    df = run.history(samples=all_samples, keys=[log_key])
                    df.rename(columns={log_key: "reward"}, inplace=True)
                    df['id'] = run.id + "_" + log_key.split("/")[-1]
                    df['group'] = run._attrs['group']
                    df['algo'] = "baseline"
                    final_df = pd.concat([final_df, df])

            # Models
            model_key = "test/reward/model"
            df = run.history(samples=all_samples, keys=[model_key])
            df.rename(columns={model_key: "reward"}, inplace=True)
            df['id'] = run.id
            df['group'] = run._attrs['group']
            df['algo'] = run._attrs['config']['algo']
            final_df = pd.concat([final_df, df])

        #
        assert final_df.groupby(['id']).size().unique().size == 1, "All runs should have the same number of samples"
        final_df.to_csv(self.filepath, index=True)
        if self.program.args.project_verbose > 0:
            self.program.log.info(f"History downloaded and saved to {self.filepath}")
        return runs_with_history

    def create_stats(self):
        history_df: pd.DataFrame = pd.read_csv(self.filepath, index_col=0)
        pivot_df = history_df.pivot(columns=['id'], values=['reward'])
        pivot_df = pd.concat(
            [pd.DataFrame(data=[[0] * pivot_df.columns.__len__()], columns=pivot_df.columns), pivot_df]).reset_index(
            drop=True)
        cumprod_df = (pivot_df + 1).cumprod()
        idx_min = cumprod_df.iloc[-1].argmin()
        idx_max = cumprod_df.iloc[-1].argmax()
        id_min = cumprod_df.iloc[-1].index[idx_min]
        id_max = cumprod_df.iloc[-1].index[idx_max]

        # TODO: Change index/_step by date

        # Sharpe
        print("Sharpe")
        print(empyrical.sharpe_ratio(pivot_df[id_min]))
        print(empyrical.sharpe_ratio(pivot_df[id_max]))

        # Beta
        print("Beta")
        print(empyrical.beta(pivot_df[id_min], pivot_df[('reward', 'zy8buea3_^DJI')]))
        print(empyrical.beta(pivot_df[id_max], pivot_df[('reward', 'zy8buea3_^DJI')]))

        # Max drawdown
        print("Max drawdown")
        print(empyrical.max_drawdown(pivot_df[id_min]))
        print(empyrical.max_drawdown(pivot_df[id_max]))

        # Plot
        print()

        plt.subplot(4, 1, 1)
        pf.plotting.plot_rolling_returns(pivot_df[id_min], pivot_df[('reward', 'zy8buea3_^DJI')])
        plt.subplot(4, 1, 2)
        pf.plotting.plot_rolling_returns(pivot_df[id_max], pivot_df[('reward', 'zy8buea3_^DJI')])

        # Daily, Non-Cumulative Returns
        plt.subplot(4, 1, 3)
        pf.plotting.plot_returns(pivot_df[id_min])
        plt.subplot(4, 1, 4)
        pf.plotting.plot_returns(pivot_df[id_max])
        plt.tight_layout()
        plt.show()
        # pf.plotting.plot_returns(bt_returns)
        # pf.plot_annual_returns(bt_returns)
        # pf.plot_monthly_returns_dist(bt_returns)
        # pf.plot_monthly_returns_heatmap(bt_returns)
        # pf.plot_return_quantiles(bt_returns)
        # pf.plot_rolling_beta(bt_returns)
        # pf.plot_rolling_sharpe(bt_returns)
        # pf.plot_drawdown_periods(bt_returns)
        # pf.plot_drawdown_underwater(bt_returns)


def t():
    program = Program()
    wandbstats = Reload(program=program)
    # foo = wandbstats.download_test_history()
    foo = wandbstats.create_stats()
    return foo


if __name__ == "__main__":
    t()
