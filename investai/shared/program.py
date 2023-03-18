# -*- coding: utf-8 -*-
from argparse import Namespace

from loguru import logger
from shared.arguments import parse_arguments
from shared.utils import find_git_root
from torch.utils.tensorboard import SummaryWriter


class Program:
    def __init__(self):
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=find_git_root(__file__).joinpath(".env").as_posix())
        self.args: Namespace = parse_arguments()[1]
        self.log: logger = logger
        self.__post_init__()

    def __init_logger__(self, file_path: str):
        from loguru import logger

        open(file_path, "w").close()  # clear log file
        logger.add(file_path, rotation="1 MB", backtrace=True, diagnose=True, enqueue=True)
        return logger

    def __post_init__(self):
        self.log = self.__init_logger__(self.args.folder_out.joinpath("run.log").as_posix())
        writer = SummaryWriter(self.args.folder_tensorboard.as_posix())
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key} |{value}|" for key, value in vars(self.args).items()])),
        )

    def is_wandb_enabled(self):
        return self.args.wandb or self.args.wandb_sweep
