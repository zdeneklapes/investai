# -*- coding: utf-8 -*-
import wandb

from shared.program import Program


class WandbStats:
    def __init__(self, program: Program = None):
        self.program = program

    def create_graphs(self):
        runs = wandb.Api().runs(self.program.args.wandb_entity + "/" + self.program.args.wandb_project)
        for run in runs:
            for artifacts in run.logged_artifacts():
                models = [artifact for artifact in artifacts if artifact.type == 'model']
                for model in models:
                    print(model.name)

                # TODO: create stats for all models


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
