# -*- coding: utf-8 -*-
import shutil
from pathlib import Path
from shared.utils import find_git_root


class ExperimentDir:
    def __init__(self, root: Path = None):
        if not root:
            root = find_git_root(Path(__file__).parent)

        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.root: Path = root
        self.out: Path = self.root.joinpath("out")
        self.datasets: Path = self.out.joinpath("datasets")
        self.models: Path = self.out.joinpath("models")
        self.tensorboard: Path = self.models.joinpath("tensorboard")

        #
        self.create_dirs()

    def create_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        self.tensorboard.mkdir(parents=True, exist_ok=True)

    def delete_out_dir(self):
        shutil.rmtree(self.out)  # Force delete even if dir is not empty