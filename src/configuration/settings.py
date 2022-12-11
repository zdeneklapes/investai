# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


# Root
class ProjectDir:
    class _DatasetDir:
        ROOT = ROOT.joinpath("dataset")
        STOCK = ROOT.joinpath("stock")
        AI4FINANCE = STOCK.joinpath("ai4finance")

    class _ModelDir:
        ROOT = ROOT.joinpath("trained_models")

    PARENT = ROOT.parent
    SRC = Path(__file__).parent.parent
    DATASET = _DatasetDir
    MODEL = _ModelDir
