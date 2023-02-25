# -*- coding: utf-8 -*-
from pathlib import Path

from project_configs.experiment_dir import ExperimentDir


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

    class _ModelDir:
        def __init__(self, root: Path):
            self.root = root.joinpath("models")
            #
            self.experiments = self.root.joinpath("experiments")

    def __init__(self, root: Path = None):
        if not root:
            root = Path(__file__).parent.parent.parent

        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        if root.name != "ai-investing":
            raise ValueError(f"Project root directory is not \"ai-investing\", but \"{root.as_posix()}\"")

        self.root = root
        #
        self.parent = root.parent
        self.data = self._DataDir(self.root)
        self.model = self._ModelDir(self.root)

    def check_and_create_dirs(self):
        self.model.root.mkdir(parents=True, exist_ok=True)
        print(f"Created {self.model.root}")


if __name__ == "__main__":
    experiment_dir = ExperimentDir(Path(__file__).parent)
    experiment_dir.delete_out_dir()
