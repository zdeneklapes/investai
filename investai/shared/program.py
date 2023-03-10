# -*- coding: utf-8 -*-
from os import getenv
from typing import Union
from pathlib import Path
from argparse import Namespace
import attr

from shared.dir.project_dir import ProjectDir
from shared.dir.experiment_dir import ExperimentDir
from shared.arguments import parse_arguments


@attr.define
class Program:
    project_dir: ProjectDir = attr.field(default=ProjectDir())
    experiment_dir: ExperimentDir = attr.field(default=ExperimentDir(Path(__file__).parent))
    args: Union[vars, Namespace] = attr.field(default=parse_arguments()[1])
    train_date_start: str = attr.field(default='xxx-xx-xx')
    train_date_end: str = attr.field(default='xxx-xx-xx')
    test_date_start: str = attr.field(default='xxx-xx-xx')
    test_date_end: str = attr.field(default='xxx-xx-xx')
    debug = getenv('DEBUG', None)
