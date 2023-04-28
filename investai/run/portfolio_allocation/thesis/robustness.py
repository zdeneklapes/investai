# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path


from shared.program import Program
from shared.reload import reload_module  # noqa


class Robustness:
    def __init__(self, program: Program = Program()):
        self.program = program

    def find_the_best_model_id(self) -> str:
        history_df = pd.read_csv(self.program.args.history_path.as_posix(), index_col=0)
        returns_pivot_df = history_df.pivot(columns=['id'], values=['reward'])
        returns_pivot_df.columns = returns_pivot_df.columns.droplevel(0)
        # Set first rows on 0
        returns_pivot_df = pd.concat(
            [pd.DataFrame(data=[[0] * returns_pivot_df.columns.__len__()], columns=returns_pivot_df.columns),
             returns_pivot_df]
        ).reset_index(drop=True)
        # Portfolio Return
        cumprod_returns_df = (returns_pivot_df + 1).cumprod()
        # ID
        best_model_id = cumprod_returns_df.iloc[-1].idxmax()
        return best_model_id

    def get_best_model_from_wandb(self):
        id = self.find_the_best_model_id()
        print(id)

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
        self.program.args.history_path = Path("out/model/history.csv")
        self.program.args.project_debug = True
        return main(self.program)


def test():
    """Easily run test from ipython"""
    return TestRobustness().run_tests()


def main(program: Program):
    """Main function"""
    return Robustness(program).find_the_best_model_id()


if __name__ == "__main__":
    program = Program()
    test() if program.args.project_debug else main(program=program)  # Test or Main
