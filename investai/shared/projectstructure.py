# -*- coding: utf-8 -*-
from pathlib import Path

from shared.utils import find_git_root


# Root
class ProjectStructure:
    def __init__(self, root: Path = None):
        if not root:
            root = find_git_root(Path(__file__).parent)
        elif not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")
        self.root = root
        self.tickers = self.root.joinpath("data/tickers")
        self.out: Path = self.root.joinpath("out")
        self.datasets: Path = self.out.joinpath("datasets")
        self.baselines: Path = self.out.joinpath("baselines")
        self.models: Path = self.out.joinpath("models")
        self.tensorboard: Path = self.models.joinpath("tensorboard")
        self.wandb: Path = self.models.joinpath("wandb")

        #
        self.create_dirs()

    def create_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.tickers.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.baselines.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        self.tensorboard.mkdir(parents=True, exist_ok=True)
        self.wandb.mkdir(parents=True, exist_ok=True)
