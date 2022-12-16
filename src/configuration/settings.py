# -*- coding: utf-8 -*-
from pathlib import Path


# Root
class ProjectDir:
    class _DatasetDir:
        def __init__(self, root: Path):
            self.root = root.joinpath("dataset")
            self.stock = self.root.joinpath("stock")
            self.ai4finance = self.stock.joinpath("ai4finance")
            self.financial_modeling_prep = self.root.joinpath("financialmodelingprep")

    class _ModelDir:
        def __init__(self, root: Path):
            self.root = root.joinpath("trained_models")

    def __init__(self, root: Path = None):
        if not root:
            root = Path(__file__).parent.parent.parent

        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.root = root
        self.parent = root.parent
        self.dataset = self._DatasetDir(self.root)
        self.model = self._ModelDir(self.root)

    def check_and_create_dirs(self):
        # print("Checking and creating directories...")
        # self.DATASET.ROOT.mkdir(parents=True, exist_ok=True)
        # print(f"Created {self.DATASET.ROOT}")
        # self.DATASET.STOCK.mkdir(parents=True, exist_ok=True)
        # print(f"Created {self.DATASET.STOCK}")
        # self.DATASET.AI4FINANCE.mkdir(parents=True, exist_ok=True)
        # print(f"Created {self.DATASET.AI4FINANCE}")
        self.model.root.mkdir(parents=True, exist_ok=True)
        print(f"Created {self.model.root}")
