# -*- coding: utf-8 -*-
import inspect
from itertools import chain
from typing import List
from pathlib import Path

import pandas as pd
import wandb
from tqdm import tqdm

from extra.math.finance.shared.baseline import Baseline
from shared.program import Program


class WandbAPI:
    def __init__(self, program: Program = None):
        self.program = program
        self.baseline = Baseline(self.program)
        self.baseline.load_csv(self.program.args.baseline_path.as_posix())

    def download_test_history(self, groups: List[str] = None, log_keys: List[str] = None,
                              samples=100_000, ) -> pd.DataFrame:
        # Functions
        def get_run_baseline(last_n_samples: int, log_key: str, algo: str) -> pd.DataFrame:
            series: pd.Series = self.baseline.df[log_key].tail(last_n_samples)
            df: pd.DataFrame = series.to_frame()
            df.rename(columns={log_key: "reward"}, inplace=True)
            df['id'] = log_key
            df['group'] = df['algo'] = "baseline"
            return df

        def get_run_model(run: wandb.apis.public.Run, log_key: str) -> pd.DataFrame:
            df = run.history(samples=samples, keys=[log_key])
            df.rename(columns={log_key: "reward"}, inplace=True)
            df['id'] = run.id
            df['group'] = run._attrs['group']
            df['algo'] = run._attrs['config']['algo']
            return df

        # Logics
        if self.program.args.project_verbose > 0:
            self.program.log.info(f"START {inspect.currentframe().f_code.co_name}")
        if log_keys is None:
            log_keys = [
                "^DJI",
                "^GSPC",
                "^IXIC",
                "^RUT",
                "maximum_sharpe_0_1",
                "maximum_quadratic_utility_0_1",
                "minimum_variance_0_1",
            ]
        if groups is None:
            groups = [
                # "sweep-nasfit-2",
                # "sweep-nasfit-3",
                # "sweep-nasfit-4"
                # "sweep-nasfit-5"
                "sweep-nasfit-6"
            ]

        api = wandb.Api()
        runs = [
            api.runs(self.program.args.wandb_entity + "/" + self.program.args.wandb_project, filters={"group": group})
            for group in groups
        ]

        iterations = tqdm(enumerate(list(chain(*runs))))

        # Models
        runs_data = [get_run_model(run, "test/reward/model") for _, run in iterations if
                     run.state not in ["running", "failed"]]
        final_df = pd.concat(runs_data)

        # Baselines
        n_samples = final_df.groupby(['id']).size()[0]
        final_dfs = [get_run_baseline(n_samples, log_key, "baseline") for log_key in tqdm(log_keys)]
        final_df = pd.concat([final_df, *final_dfs])

        #
        assert final_df.groupby(['id']).size().unique().size == 1, "All runs should have the same number of samples"
        if not (self.program.args.project_debug):
            final_df.to_csv(self.program.args.history_path.as_posix(), index=True)
            if self.program.args.project_verbose > 0:
                self.program.log.info(f"History downloaded and saved to {self.program.args.history_path.as_posix()}")
        if self.program.args.project_verbose > 0:
            self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")
        return final_df


class TestWandbAPI:
    def __init__(self, program: Program = Program()):
        self.program = program

    def run_tests(self):
        return self.test_main()

    def test_main(self):
        self.program.args.baseline_path = Path("out/baseline/baseline.csv")
        self.program.args.project_debug = True
        return main(self.program)


def test():
    return TestWandbAPI().run_tests()  # Just to can easily run test from ipython


def main(program: Program):
    return WandbAPI(program).download_test_history()


if __name__ == "__main__":
    program = Program()
    if program.args.project_debug:  # Just to can run test from CLI
        test()
    else:
        main(program)
