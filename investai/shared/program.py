# -*- coding: utf-8 -*-
from os import getenv
from typing import Union
from argparse import Namespace
import attr

from shared.projectstructure import ProjectStructure
from shared.arguments import parse_arguments
from shared.utils import get_logger


@attr.define
class Program:
    project_structure: ProjectStructure = attr.field(default=ProjectStructure())
    args: Union[vars, Namespace] = attr.field(default=parse_arguments()[1])
    train_date_start: str = attr.field(default='xxx-xx-xx')
    train_date_end: str = attr.field(default='xxx-xx-xx')
    test_date_start: str = attr.field(default='xxx-xx-xx')
    test_date_end: str = attr.field(default='xxx-xx-xx')
    debug = getenv('DEBUG', None)

    def __attrs_post_init__(self):
        self.logger = get_logger(self.project_structure.out.joinpath('run.log').as_posix())
