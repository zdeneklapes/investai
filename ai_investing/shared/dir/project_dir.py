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

    def __init__(self, file_dir: str):
        root = self._find_git_root(Path(file_dir).parent)

        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.root = root
        self.data = self._DataDir(self.root)

    def _find_git_root(self, path):
        path = Path(path).resolve()
        if (path / '.git').is_dir():
            return path
        if path == path.parent:
            raise Exception('Not a Git repository')
        return self._find_git_root(path.parent)


if __name__ == "__main__":
    experiment_dir = ExperimentDir(Path(__file__).parent)
    experiment_dir.delete_out_dir()
