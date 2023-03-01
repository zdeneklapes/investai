# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path


class ExperimentDir:
    def __init__(self, root: Path = None):
        if not root:
            root = Path(__file__).parent.parent.parent

        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.root: Path = root
        self.out: Path = self.root.joinpath("out")
        self.datasets: Path = self.out.joinpath("datasets")
        self.models: Path = self.out.joinpath("models")
        self.algo: Path | None = None
        self.tensorboard: Path | None = None
        # self.try_number: Path | None = None
        # self.algo = self.models.joinpath(algo)
        # self.chart: Path | None = None

    def set_algo(self, algo: str):
        last_algo_index = self._get_last_algorithm_index(algo)
        self.algo = self.models.joinpath(algo + f"_{last_algo_index + 1}")
        self.tensorboard = self.algo.joinpath("tensorboard")

    def _get_last_algorithm_index(self, algo: str) -> int:
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
