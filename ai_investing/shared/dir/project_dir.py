# -*- coding: utf-8 -*-
from pathlib import Path

from shared.dir.experiment_dir import ExperimentDir
from shared.utils import find_git_root


# Root
class ProjectDir:
    class _DataDir:
        class _StockDir:
            def __init__(self, root):
                self.root = root.joinpath("stock")
                #
                self.ai4finance = self.root.joinpath("ai4finance")
                self.finnhub = self.root.joinpath("finnhub")
                self.numerai = self.root.joinpath("numerai")

        def __init__(self, root: Path):
            self.root = root.joinpath("data")
            #
            self.exchanges = self.root.joinpath("exchanges")
            self.indexes = self.root.joinpath("indexes")
            self.stock = self._StockDir(self.root)
            self.test_tickers = self.root.joinpath("test_tickers")
            self.tickers = self.root.joinpath("tickers")

    def __init__(self, root: Path = None):
        if not root:
            root = find_git_root(Path(__file__).parent)

        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.root = root
        self.data = self._DataDir(self.root)


if __name__ == "__main__":
    experiment_dir = ExperimentDir(Path(__file__).parent)
    experiment_dir.delete_out_dir()
