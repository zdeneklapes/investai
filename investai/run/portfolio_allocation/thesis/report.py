# -*- coding: utf-8 -*-
import wandb
from itertools import chain
from collections import defaultdict

from tqdm import tqdm
import pandas as pd

from shared.program import Program
from shared.utils import reload_module  # noqa


class WandbStats:
    def __init__(self, program: Program = None):
        self.program = program

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
        keys = [
            "test/reward/^DJI",
            "test/reward/^GSPC",
            "test/reward/^IXIC",
            "test/reward/^RUT",
            "test/reward/maximum_sharpe_0_1",
            "test/reward/maximum_quadratic_utility_0_1",
            "test/reward/minimum_variance_0_1",
            "test/reward/model",
        ]
        all_samples = 100_000
        runs_with_history = defaultdict(dict)
        iterations = tqdm(enumerate(list(chain(*runs))))
        final_df = pd.DataFrame()
        for idx_r, run in iterations:  # type: int,  wandb.apis.public.Run
            if run.state == "running" or run.state == "failed":
                continue
            filepath = self.program.args.folder_history.joinpath(f"{run.id}.csv")
            if self.program.args.project_verbose > 0:
                iterations.set_description(f"History will be saved to {filepath}.csv")
            df = pd.DataFrame()
            for idx_k, key in tqdm(enumerate(keys), leave=False):
                if idx_k == 0:
                    df = run.history(samples=all_samples, keys=[key])
                else:
                    new_df = run.history(samples=all_samples, keys=[key])
                    df[key] = new_df[key]
            df['id'] = run.id
            df['group'] = run._attrs['group']
            final_df = pd.concat([final_df, df])
        assert final_df.groupby(['id']).size().unique().size == 1, "All runs should have the same number of samples"
        final_df.to_csv(self.program.args.folder_history.joinpath("wandb_test_history.csv"))
        return runs_with_history

    def create_stats(self):
        history_df: pd.DataFrame = pd.read_csv(self.program.args.folder_history.joinpath("wandb_test_history.csv"))
        for id, df in history_df.groupby(by=['id']):
            print(type(df))
        # empyrical.sharpe_ratio(bt_returns)
        # empyrical.beta(bt_returns,benchmark_rets)
        # empyrical.max_drawdown(bt_returns)
        # pf.plotting.plot_rolling_returns(bt_returns)
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
    wandbstats = WandbStats(program=program)
    # foo = wandbstats.download_test_history()
    foo = wandbstats.create_stats()
    return foo


if __name__ == "__main__":
    t()
