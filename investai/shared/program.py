# -*- coding: utf-8 -*-
from os import getenv
from typing import Union
from argparse import Namespace
import attr

from shared.projectstructure import ProjectStructure
from shared.arguments import parse_arguments
from loguru import logger


@attr.define
class Program:
    project_structure: ProjectStructure = attr.field(default=ProjectStructure())
    args: Union[vars, Namespace] = attr.field(default=parse_arguments()[1])
    train_date_start: str = attr.field(default='xxx-xx-xx')
    train_date_end: str = attr.field(default='xxx-xx-xx')
    test_date_start: str = attr.field(default='xxx-xx-xx')
    test_date_end: str = attr.field(default='xxx-xx-xx')
    debug = getenv('DEBUG', None)
    logger: logger = attr.field(default=logger)

    def init_logger(self, file_path: str):
        from loguru import logger
        logger.add(file_path,
                   rotation="1 MB",
                   backtrace=True,
                   diagnose=True,
                   enqueue=True)
        return logger

    def __attrs_post_init__(self):
        self.logger = self.init_logger(self.project_structure.out.joinpath('run.log').as_posix())

    def is_wandb_enabled(self):
        return self.args.wandb or self.args.wandb_sweep
