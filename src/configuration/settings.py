# -*- coding: utf-8 -*-
from pathlib import Path


# Root
class ProjectDir:
    class _DatasetDir:
        def __init__(self, root: Path):
            self.ROOT = root.joinpath("dataset")
            self.STOCK = self.ROOT.joinpath("stock")
            self.AI4FINANCE = self.STOCK.joinpath("ai4finance")

    class _ModelDir:
        def __init__(self, root: Path):
            self.ROOT = root.joinpath("trained_models")

    def __init__(self, root: Path = None):
        if not root:
            root = Path(__file__).parent.parent.parent

        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.ROOT = root
        self.PARENT = root.parent
        self.DATASET = self._DatasetDir(self.ROOT)
        self.MODEL = self._ModelDir(self.ROOT)

    def check_and_create_dirs(self):
        # print("Checking and creating directories...")
        # self.DATASET.ROOT.mkdir(parents=True, exist_ok=True)
        # print(f"Created {self.DATASET.ROOT}")
        # self.DATASET.STOCK.mkdir(parents=True, exist_ok=True)
        # print(f"Created {self.DATASET.STOCK}")
        # self.DATASET.AI4FINANCE.mkdir(parents=True, exist_ok=True)
        # print(f"Created {self.DATASET.AI4FINANCE}")
        self.MODEL.ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Created {self.MODEL.ROOT}")
