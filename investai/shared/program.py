# -*- coding: utf-8 -*-
from typing import Union
from argparse import Namespace
import attr

from torch.utils.tensorboard import SummaryWriter

from shared.arguments import parse_arguments
from loguru import logger


@attr.define
class Program:
    args: Union[vars, Namespace] = attr.field(default=parse_arguments()[1])
    log: logger = attr.field(default=logger)

    def init_logger(self, file_path: str):
        from loguru import logger
        open(file_path, 'w').close()  # clear log file
        logger.add(file_path,
                   rotation="1 MB",
                   backtrace=True,
                   diagnose=True,
                   enqueue=True)
        return logger

    def __attrs_post_init__(self):
        self.log = self.init_logger(self.args.folder_out.joinpath('run.log').as_posix())
        writer = SummaryWriter(self.args.folder_tensorboard.as_posix())
        writer.add_text(
            'hyperparameters',
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key} |{value}|" for key, value in vars(self.args).items()])),
        )

    def is_wandb_enabled(self):
        return self.args.wandb or self.args.wandb_sweep
