# -*- coding: utf-8 -*-
import shutil
from pathlib import Path


class ExperimentDir:
    def __init__(self, root: Path = None):
        if not root:
            root = Path(__file__).parent.parent.parent

        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.root = root
        self.out = self.root.joinpath("out")
        self.datasets = self.out.joinpath("datasets")
        self.models = self.out.joinpath("models")
        self.algo = None
        self.tensorboard = None
        self.results = None

    def _get_next_algo_folder_id(self) -> str:
        try:
            return str(len(list(self.algo.iterdir())) + 1)
        except Exception:
            return "1"

    def add_attributes_for_models(self, algo: str, try_number: str = None):
        self.algo = self.models.joinpath(algo)
        if not try_number:
            try_number = self._get_next_algo_folder_id()
        self.tensorboard = self.algo.joinpath(try_number).joinpath("tensorboard")
        self.results = self.algo.joinpath(try_number).joinpath("results")

    def create_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)

    def create_specific_dirs(self):
        self.algo.mkdir(parents=True, exist_ok=True)
        self.tensorboard.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    def delete_out_dir(self):
        shutil.rmtree(self.out)  # Force delete even if dir is not empty
