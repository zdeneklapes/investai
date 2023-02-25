# -*- coding: utf-8 -*-
from os import getenv

import attr
import pandas as pd

from project_configs.project_dir import ProjectDir
from project_configs.experiment_dir import ExperimentDir


@attr.define
class Program:
    prj_dir: ProjectDir
    exp_dir: ExperimentDir
    DEBUG: bool = False
    TRAIN_DATE_START = '2019-01-01'
    TRAIN_DATE_END = '2020-01-01'
    TEST_DATE_START = '2020-01-01'
    TEST_DATE_END = '2021-01-01'
    DATASET_PATH = 'out/dataset.csv'
    DEBUG = getenv('DEBUG', False)
    dataset: pd.DataFrame = attr.field(init=False)
