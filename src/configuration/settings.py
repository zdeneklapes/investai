# -*- coding: utf-8 -*-
from pathlib import Path


# Root
class ProjectDir:
    ROOT = Path(__file__).parent.parent.parent
    ROOT_PARENT = Path(__file__).parent.parent.parent.parent
    SRC = Path(__file__).parent.parent


class DatasetDir:
    ROOT = ProjectDir.ROOT.joinpath("dataset")
    STOCK = ROOT.joinpath("stock")
    AI4FINANCE = STOCK.joinpath("ai4finance")


class ModelDir:
    ROOT = ProjectDir.ROOT.joinpath("trained_models")
