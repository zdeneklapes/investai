# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

from shared.program import Program


class Robustness:
    def __init__(self, program: Program = Program()):
        self.program = program

    def find_the_model(self):
        history_df = pd.read_csv(self.program.args.history_path.as_posix())
        returns_pivot_df = history_df.pivot(columns=['id'], values=['reward'])
        returns_pivot_df = pd.concat(  # Set first rows on 0
            [pd.DataFrame(data=[[0] * returns_pivot_df.columns.__len__()], columns=returns_pivot_df.columns),
             returns_pivot_df]
        ).reset_index(drop=True)
        cumprod_returns_df = (returns_pivot_df + 1).cumprod()
        best_model = cumprod_returns_df.iloc[-1].max()  # Gets the highest value from last row
        print(best_model)

        # Download the trained models from wandb parameters for the best model

        # get parameters for the best models, (must be train on the same algorithm and same data)

    def call_training(self):
        # get data from find_the_model

        # Compare if all models have the same (similar) test/total_reward (trained
        # on the same data, with same hyperparameters)

        pass

    def test_model(self, model):
        # total_rewards = {}
        for i in range(1000):
            # Model test
            # Store total reward
            pass

    def test_robustness(self):
        trained_models = self.call_training()

        # Multiple times call testing period for each model
        for model in trained_models:
            self.test_model(model)


class TestRobustness:
    def __init__(self, program: Program = Program()):
        self.program = program

    def run_tests(self):
        return self.test_main()

    def test_main(self):
        self.program.args.baseline_path = Path("out/baseline/baseline.csv")
        self.program.args.project_debug = True
        return main(self.program)


def test():
    return TestRobustness().run_tests()  # Just to can easily run test from ipython


def main(program: Program):
    return Robustness(program).find_the_model()


if __name__ == "__main__":
    program = Program()
    if program.args.project_debug:  # Just to can run test from CLI
        test()
    else:
        main(program)
