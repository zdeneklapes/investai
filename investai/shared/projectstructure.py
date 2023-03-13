# -*- coding: utf-8 -*-
from pathlib import Path

from shared.utils import find_git_root


# Root
class ProjectStructure:
    class _DataDir:
        class _StockDir:
            def __init__(self, root):
                self.root = root.joinpath("stock")
                #
                self.ai4finance = self.root.joinpath("ai4finance")
                self.finnhub = self.root.joinpath("finnhub")
                self.numerai = self.root.joinpath("numerai")

        def __init__(self, root: Path):
            self.root = root.joinpath("raw_data")
            #
            self.exchanges = self.root.joinpath("exchanges")
            self.indexes = self.root.joinpath("indexes")
            self.stock = self._StockDir(self.root)
            self.test_tickers = self.root.joinpath("test_tickers")
            self.tickers = self.root.joinpath("tickers")

    def __init__(self, root: Path = None):
        if not root:
            root = find_git_root(Path(__file__).parent)
        elif not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.root = root
        self.data = self._DataDir(self.root)
        self.out: Path = self.root.joinpath("out")
        self.datasets: Path = self.out.joinpath("datasets")
        self.models: Path = self.out.joinpath("models")
        self.tensorboard: Path = self.models.joinpath("tensorboard")
        self.wandb: Path = self.models.joinpath("wandb")

        #
        self.create_dirs()

    def create_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.data.root.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        self.tensorboard.mkdir(parents=True, exist_ok=True)
        self.wandb.mkdir(parents=True, exist_ok=True)
