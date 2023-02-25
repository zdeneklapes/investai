# -*- coding: utf-8 -*-
from os import getenv
from typing import Union

from argparse import Namespace
import attr

from project_configs.project_dir import ProjectDir
from project_configs.experiment_dir import ExperimentDir


@attr.define
class Program:
    project_dir: ProjectDir
    experiment_dir: ExperimentDir
    args: Union[vars, Namespace] = attr.field(default=None)
    train_date_start: str = attr.field(default='2019-01-01')
    train_date_end: str = attr.field(default='2020-01-01')
    test_date_start: str = attr.field(default='2020-01-01')
    test_date_end: str = attr.field(default='2021-01-01')
    dataset_path: str = attr.field(default='out/dataset.csv')
    debug = getenv('DEBUG', None)
