# -*- coding: utf-8 -*-
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
        self.try_number: Path | None = None
        self.algo: Path | None = None
        self.tensorboard: Path | None = None
        self.chart: Path | None = None

    def _get_next_algo_folder_id(self) -> str:
        try:
            return str(len(list(self.algo.iterdir())) + 1)
        except Exception:
            return "1"

    def add_attributes_for_models(self, algo: str):
        self.algo = self.models.joinpath(algo)
        self.try_number = self.algo.joinpath(self._get_next_algo_folder_id())
        self.tensorboard = self.try_number.joinpath("tensorboard")
        self.chart = self.try_number.joinpath("chart")

    def create_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)

    def create_specific_dirs(self):
        self.algo.mkdir(parents=True, exist_ok=True)
        self.tensorboard.mkdir(parents=True, exist_ok=True)
        self.chart.mkdir(parents=True, exist_ok=True)

    def delete_out_dir(self):
        shutil.rmtree(self.out)  # Force delete even if dir is not empty
