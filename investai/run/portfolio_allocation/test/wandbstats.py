# -*- coding: utf-8 -*-
import wandb
from pprint import pprint

from shared.program import Program
from shared.utils import calculate_sharpe_ratio, reload_module  # noqa


class WandbStats:
    def __init__(self, program: Program = None):
        self.program = program

    def create_graphs(self):
        api = wandb.Api()
        runs = [
            api.runs(self.program.args.wandb_entity + "/" + self.program.args.wandb_project, filters={"group": group})
            for group in [
                # "sweep-nasfit-2",
                # "sweep-nasfit-3",
                "sweep-nasfit-4"
            ]
        ]
        test_log_keys = [
            "test/reward/^DJI",
            "test/reward/^GSPC",
            "test/reward/^IXIC",
            "test/reward/^RUT",
            "test/reward/maximum_sharp_0_1",
            "test/reward/maximum_quadratic_utility_0_1",
            "test/reward/minimum_variance_0_1",
            "test/reward/model",
        ]
        for runs_group in runs:
            for run in runs_group:
                history = run.scan_history(keys=test_log_keys)
                for key in test_log_keys:
                    print(key)
                    pprint(history[key])
                    print()


class TestWandbStats:
    def __init__(self, program: Program):
        self.program = program

    def test_create_graphs(self):
        wandbstats = WandbStats(program=self.program)
        wandbstats.create_graphs()


def t():
    program = Program()
    wandbstats = TestWandbStats(program=program)
    wandbstats.test_create_graphs()


def main():
    program = Program()
    wandbstats = WandbStats(program=program)
    wandbstats.create_graphs()


if __name__ == "__main__":
    # main()
    t()
