# -*- coding: utf-8 -*-
from os import getenv
from typing import Union

from argparse import Namespace
import attr

from shared.dir.project_dir import ProjectDir
from shared.dir.experiment_dir import ExperimentDir


@attr.define
class Program:
    project_dir: ProjectDir
    experiment_dir: ExperimentDir
    args: Union[vars, Namespace] = attr.field(default=None)
    train_date_start: str = attr.field(default='xxx-xx-xx')
    train_date_end: str = attr.field(default='xxx-xx-xx')
    test_date_start: str = attr.field(default='xxx-xx-xx')
    test_date_end: str = attr.field(default='xxx-xx-xx')
    debug = getenv('DEBUG', None)
