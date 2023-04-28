# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

import wandb

from shared.program import Program
from shared.reload import reload_module  # noqa
import json
from run.shared.sb3.sweep_configuration import sweep_configuration
from run.portfolio_allocation.thesis.train import main as train_main
from run.portfolio_allocation.thesis.test import Test
from run.shared.memory import Memory


class Robustness:
    def __init__(self, program: Program = Program()):
        self.program = program

    def history_to_pivot(self, history_path: str):
        history_df = pd.read_csv(history_path, index_col=0)
        returns_pivot_df = history_df.pivot(columns=["id"], values=["reward"])
        returns_pivot_df.columns = returns_pivot_df.columns.droplevel(0)
        # Set first rows on 0
        returns_pivot_df = pd.concat(
            [pd.DataFrame(data=[[0] * returns_pivot_df.columns.__len__()], columns=returns_pivot_df.columns),
             returns_pivot_df]
        ).reset_index(drop=True)
        # Portfolio Return
        cumprod_returns_df = (returns_pivot_df + 1).cumprod()
        return cumprod_returns_df

    def find_the_best_model_id(self) -> str:
        cumprod_returns_df = self.history_to_pivot(self.program.args.history_path.as_posix())
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
        keys = config.keys() & sweep_configuration["parameters"].keys()
        hyperparameters = {key: config[key]["value"] for key in keys}
        hyperparameters["total_timesteps"] = config["_total_timesteps"]["value"]
        return hyperparameters, config["algo"]["value"]

    def get_artifact_best_model_from_wandb(self, type: str) -> Path | None:
        id = self.find_the_best_model_id()
        wandb_api = wandb.Api()
        run: wandb.apis.public.Run = wandb_api.run(
            f"{self.program.args.wandb_entity}/{self.program.args.wandb_project}/{id}")
        for artifact in run.logged_artifacts():  # type: wandb.apis.public.Artifact
            if artifact.type == type:
                artifact_dir = artifact.download(root=self.program.args.folder_dataset.joinpath("artifacts").as_posix())
                return Path(artifact_dir) / artifact.file().split("/")[-1]
        return None

    def set_hyperparameters_to_program(self, hyperparameters: Dict):
        for key, value in hyperparameters.items():
            setattr(self.program.args, key, value)

    def call_training(self):
        hyperparameters, algorithm = self.get_hyperparameters_best_model_from_wandb()
        self.program.args.dataset_paths = [self.get_artifact_best_model_from_wandb(type="dataset")]
        self.program.args.algorithms = [algorithm]
        self.set_hyperparameters_to_program(hyperparameters)
        # Train
        train_main(self.program)

    def test_best_models(self):
        history_df = pd.read_csv(self.program.args.history_path.as_posix(), index_col=0)
        returns_pivot_df = history_df.pivot(columns=["group", "id"], values=["reward"])
        returns_pivot_df.columns = returns_pivot_df.columns.droplevel(0)
        cumprod_returns_df = (returns_pivot_df[("run-nasfit-robust-1")] + 1).cumprod()
        best_model_id = cumprod_returns_df.iloc[-1]
        return best_model_id

    def test_multiple_times_one_best_model(self):
        model_path = self.get_artifact_best_model_from_wandb(type="model")
        hyperparameters, algorithm = self.get_hyperparameters_best_model_from_wandb()
        dataset_path = self.get_artifact_best_model_from_wandb(type="dataset")
        memory: Memory = Test(program=self.program, dataset_path=dataset_path).test(model_path=model_path.as_posix(),
                                                                                    algorithm=algorithm)
        # TODO: Evaluate memory


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
    return Robustness(program).test_best_models()


if __name__ == "__main__":
    program = Program()
    test() if program.args.project_debug else main(program=program)  # Test or Main
