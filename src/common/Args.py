# -*- coding: utf-8 -*-
from typing import Dict, List, Any
import dataclasses
import argparse
import os
from operator import itemgetter

from config.settings import PROJECT_STUFF_DIR


class _LoadFromFile(argparse.Action):
    """Source: <https://stackoverflow.com/a/27434050/14471542>"""

    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


@dataclasses.dataclass(init=False)
class Args:
    class Names:
        TRAIN = "train"
        TEST = "test"
        CREATE_DATASET = "create-dataset"
        DATASET = "dataset"
        MODELS = "models"
        CONFIG = "config"

    def __init__(self):
        args = self.argument_parser()
        self.validator(args)

        #
        self.train: bool = args[Args.Names.TRAIN]
        self.test: bool = args[Args.Names.TEST]
        self.create_dataset: bool = args[Args.Names.CREATE_DATASET]
        self.dataset: str = args[Args.Names.DATASET]
        self.models: List[str] = args[Args.Names.MODELS]
        self.config: str = args[Args.Names.CONFIG]

    def argument_parser(self) -> Dict[str, Any]:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            f"--{Args.Names.TRAIN}",
            dest=f"{Args.Names.TRAIN}",
            help="Will train models based on hyper parameters",
            action="store_true",
        )
        parser.add_argument(
            f"--{Args.Names.TEST}",
            dest=f"{Args.Names.TEST}",
            help="Will test trained models.",
            action="store_true",
        )
        parser.add_argument(
            f"--{Args.Names.CREATE_DATASET}",
            dest=f"{Args.Names.CREATE_DATASET}",
            help=f"Prepare and save dataset as csv into: {PROJECT_STUFF_DIR}",
            action="store_true",
        )
        parser.add_argument(
            f"--{Args.Names.DATASET}",
            dest=f"{Args.Names.DATASET}",
            help="Use already prepared dataset.",
            nargs="?",  # 1 optional argument
            type=str,
        )
        parser.add_argument(
            f"--{Args.Names.MODELS}",
            dest=f"{Args.Names.MODELS}",
            help="Already trained model",
            nargs="+",  # 1 or more arguments
            type=str,
            default=[],
        )
        parser.add_argument(f"--{Args.Names.CONFIG}", dest=f"{Args.Names.CONFIG}", type=open, action=_LoadFromFile)
        cli_args = vars(parser.parse_args())
        return cli_args

    def validator(self, args: vars):
        parser = argparse.ArgumentParser()

        if not any(args.values()):
            parser.error("No argument provided")

        # All
        if not any(itemgetter(Args.Names.TRAIN, Args.Names.TEST, Args.Names.DATASET, Args.Names.CREATE_DATASET)(args)):
            parser.error("Please choose at least one action to do.")

        # model
        for model in args[Args.Names.MODELS]:
            if not os.path.exists(model):
                parser.error("Model not found")

        # Dataset
        if args[Args.Names.DATASET] is not None and not os.path.exists(args[Args.Names.DATASET]):
            parser.error("Dataset not found")
