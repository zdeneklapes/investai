# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

import wandb

from shared.program import Program
from shared.reload import reload_module  # noqa
import json
from run.shared.sb3.sweep_configuration import sweep_configuration
from run.portfolio_allocation.thesis.train import Train


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

    def get_hyperparameters_best_model_from_wandb(self) -> Tuple[Dict, str]:
        """
        Get hyperparameters from the best model in wandb
        :returns: hyperparameters: dict, algo: str
        """
        id = self.find_the_best_model_id()
        wandb_api = wandb.Api()
        run: wandb.apis.public.Run = wandb_api.run(
            f"{self.program.args.wandb_entity}/{self.program.args.wandb_project}/{id}")
        config = json.loads(run.json_config)
        keys = config.keys() & sweep_configuration['parameters'].keys()
        hyperparameters = {key: config[key]['value'] for key in keys}
        return hyperparameters, config['algo']['value']

    def get_dataset_best_model_from_wandb(self, type: str) -> Path | None:
        id = self.find_the_best_model_id()
        wandb_api = wandb.Api()
        run: wandb.apis.public.Run = wandb_api.run(
            f"{self.program.args.wandb_entity}/{self.program.args.wandb_project}/{id}")
        for artifact in run.logged_artifacts():  # type: wandb.apis.public.Artifact
            if artifact.type == type:
                artifact_dir = artifact.download(root=self.program.args.wandb_dir.as_posix())
                return Path(artifact_dir)
        return None

    def set_hyperparameters_to_program(self, hyperparameters: Dict):
        for key, value in hyperparameters.items():
            setattr(self.program.args, key, value)

    def call_training(self):
        dataset_path = self.get_dataset_best_model_from_wandb(type="dataset")
        hyperparameters, algorithm = self.get_hyperparameters_best_model_from_wandb()
        self.set_hyperparameters_to_program(hyperparameters)

        # Train
        Train(program=program, dataset_path=dataset_path, algorithm=algorithm).train()

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
    return Robustness(program).get_hyperparameters_best_model_from_wandb()


if __name__ == "__main__":
    program = Program()
    test() if program.args.project_debug else main(program=program)  # Test or Main
