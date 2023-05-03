# -*- coding: utf-8 -*-
"""Test robustness of the trained model."""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

import wandb

from shared.program import Program
from shared.reload import reload_module  # noqa
import json
from run.shared.sb3.sweep_configuration import sweep_configuration
from run.portfolio_allocation.thesis.train import main as train_main
from run.portfolio_allocation.thesis.report import print_stats_latex


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
        best_model_id = cumprod_returns_df.iloc[-1].idxmax()
        if "i" in self.program.args.project_verbose: self.program.log.info(f"Best model id: {best_model_id}")
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
        history_df = pd.read_csv(self.program.args.history_path, index_col=0)
        id = self.find_the_best_model_id()
        returns_pivot_df = history_df.pivot(columns=["group", "id"], values=["reward"])
        returns_pivot_df.columns = returns_pivot_df.columns.droplevel(0)
        # Set first rows on 0
        returns_pivot_df = pd.concat(
            [pd.DataFrame(data=[[0] * returns_pivot_df.columns.__len__()], columns=returns_pivot_df.columns),
             returns_pivot_df]
        ).reset_index(drop=True)
        groups_df = returns_pivot_df["run-nasfit-robust-3"]
        cumprod_returns_df = (groups_df + 1).cumprod()
        returns_robust_df: pd.Series = cumprod_returns_df.iloc[-1]
        best_model_return = (returns_pivot_df['sweep-nasfit-6', id] + 1).cumprod().iloc[-1]
        returns_robust_df[id] = best_model_return
        df = returns_robust_df.to_frame()
        df.columns = ["test/total_reward"]
        mean = df["test/total_reward"].mean().round(3)
        print_stats_latex(
            df,
            2,
            0.2,
            f"Robust Test: Trained models using hyperparameters from the best model trained using sweep. "
            f"The mean is {mean} and the spead between the highest and the lowest is reward "
            f"is {(df['test/total_reward'].max() - df['test/total_reward'].min()).round(3)}.",
            "tab:robust",
            file_path=self.program.args.folder_figure.joinpath("robust.tex"),
            highlight=False,
            vspace="0cm")


class TestRobustness:
    def __init__(self, program: Program = Program()):
        self.program = program

    def run_tests(self):
        return self.test_main()

    def test_main(self):
        self.program.args.history_path = Path("out/model/history.csv")
        self.program.args.project_verbose = "id"
        self.program.args.test = 5
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
