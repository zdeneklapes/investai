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
from shared.reload import reload_module  # noqa
from shared.utils import log_artifact


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
        if "i" in self.program.args.project_verbose:
            self.program.log.info(f"START {inspect.currentframe().f_code.co_name}")
        if log_keys is None: log_keys = ["^DJI", "^GSPC", "^IXIC", "^RUT", "maximum_sharpe_0_1", "minimum_variance_0_1"]

        # Models 1
        api = wandb.Api()
        runs = [
            api.runs(self.program.args.wandb_entity + "/" + self.program.args.wandb_project, filters={"group": group})
            for group in [
                "sweep-nasfit-6",
            ]
        ]
        iterations = tqdm(enumerate(list(chain(*runs))))
        runs_data = []
        for idx, run in iterations:
            if run.state not in ["running", "failed"]:
                runs_data.append(get_run_model(run, "test/reward/model"))
        df = pd.concat(runs_data)

        # Models 2
        api = wandb.Api()
        runs = [
            api.runs(self.program.args.wandb_entity + "/" + self.program.args.wandb_project, filters={"group": group})
            for group in [
                "run-nasfit-robust-1",
                "run-nasfit-robust-2",
                "run-nasfit-robust-3",
                "sweep-nasfit-7",
            ]
        ]
        iterations = tqdm(enumerate(list(chain(*runs))))
        runs_data = []
        for idx, run in iterations:
            if run.state not in ["running", "failed"]:
                runs_data.append(get_run_model(run, "test/reward_1/model"))
        df = pd.concat([df, *runs_data])

        # Baselines
        n_samples = df.groupby(['id']).size()[0]
        baseline_dfs = [get_run_baseline(n_samples, log_key, "baseline") for log_key in tqdm(log_keys)]
        df = pd.concat([df, *baseline_dfs])

        # set Index
        df.index = df.groupby(['id']).cumcount()

        # Check
        assert df.groupby(['id']).size().unique().size == 1, "All runs should have the same number of samples"

        df.to_csv(self.program.args.history_path.as_posix(), index=True)
        if "i" in self.program.args.project_verbose:
            self.program.log.info(f"History saved to {self.program.args.history_path.as_posix()}")
        if "i" in self.program.args.project_verbose:
            self.program.log.info(f"END {inspect.currentframe().f_code.co_name}")
        if self.program.is_wandb_enabled(check_init=False):
            log_artifact(self.program.args,
                         self.program.args.history_path.as_posix(),
                         self.program.args.history_path.name.split('.')[0],
                         "history",
                         {"path": self.program.args.history_path.as_posix()})
        return df


class TestWandbAPI:
    def __init__(self, program: Program = Program()):
        self.program = program

    def run_tests(self):
        return self.test_main()

    def test_main(self):
        self.program.args.baseline_path = Path("out/baseline/baseline.csv")
        self.program.args.project_verbose = "id"
        self.program.args.project_debug = True
        return main(self.program)


def test():
    return TestWandbAPI().run_tests()  # Just to can easily run test from ipython


def main(program: Program):
    return WandbAPI(program).download_test_history()


if __name__ == "__main__":
    program = Program()
    if program.args.project_verbose:
        test()
    else:
        main(program)
