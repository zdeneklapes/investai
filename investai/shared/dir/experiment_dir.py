# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path
from typing import Optional


class ExperimentDir:
    def __init__(self, root: Path = None):
        if not root:
            from shared.utils import find_git_root
            root = find_git_root(Path(__file__).parent)

        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.root: Path = root
        self.out: Path = self.root.joinpath("out")
        self.datasets: Path = self.out.joinpath("datasets")
        self.models: Path = self.out.joinpath("models")
        self.algo: Optional[Path] = None
        self.tensorboard: Optional[Path] = None

        #
        self.create_dirs()

    def set_algo(self, algo_name: str):
        self.algo = self.models.joinpath(algo_name)
        self.tensorboard = self.algo.joinpath("tensorboard")

    def get_last_algorithm_index(self, algo: str) -> int:
        """Get the index of the last trained model. If no model is trained, return 0"""
        models = [i for i in os.listdir(self.models.as_posix()) if algo in i]
        if len(models) == 0:
            return 0
        else:
            models.sort()
            last_algo = int(models[-1].split("_")[1])
            return last_algo

    def create_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)

    def create_specific_dirs(self):
        self.algo.mkdir(parents=True, exist_ok=True)
        self.tensorboard.mkdir(parents=True, exist_ok=True)
        # self.chart.mkdir(parents=True, exist_ok=True)

    def delete_out_dir(self):
        shutil.rmtree(self.out)  # Force delete even if dir is not empty
