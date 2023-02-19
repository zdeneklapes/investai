# -*- coding: utf-8 -*-
from pathlib import Path
import shutil


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

    def add_attributes_for_models(self, algo: str, try_number: str):
        self.algo.joinpath(algo).mkdir(parents=True, exist_ok=True)
        self.tensorboard = self.algo.joinpath(try_number).joinpath("tensorboard")
        self.results = self.algo.joinpath(try_number).joinpath("results")

    def create_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)

    def create_specific_dirs(self, algo: str, try_number: str):
        tensorboard_dir = self.tensorboard(try_number)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        results_dir = self.results(algo, try_number)
        results_dir.mkdir(parents=True, exist_ok=True)

    def delete_out_dir(self):
        shutil.rmtree(self.out)  # Force delete even if dir is not empty


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
