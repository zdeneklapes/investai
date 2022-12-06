# -*- coding: utf-8 -*-
from typing import Dict, List, Any
import dataclasses
import argparse
import os
from operator import itemgetter

from config.settings import PROJECT_STUFF_DIR


@dataclasses.dataclass
class Args:
    train: bool
    test: bool
    create_dataset: bool
    dataset: str
    models: List[str]
    config: str


class _LoadFromFile(argparse.Action):
    """Source: <https://stackoverflow.com/a/27434050/14471542>"""

    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


def argument_parser():
    class Names:
        TRAIN = "train"
        TEST = "test"
        CREATE_DATASET = "create-dataset"
        DATASET = "dataset"
        MODELS = "models"
        CONFIG = "config"

    def get_argparse() -> Dict[str, Any]:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            f"--{Names.TRAIN}",
            dest=f"{Names.TRAIN}",
            help="Will train models based on hyper parameters",
            action="store_true",
        )
        parser.add_argument(
            f"--{Names.TEST}",
            dest=f"{Names.TEST}",
            help="Will test trained models.",
            action="store_true",
        )
        parser.add_argument(
            f"--{Names.CREATE_DATASET}",
            dest=f"{Names.CREATE_DATASET}",
            help=f"Prepare and save dataset as csv into: {PROJECT_STUFF_DIR}",
            action="store_true",
        )
        parser.add_argument(
            f"--{Names.DATASET}",
            dest=f"{Names.DATASET}",
            help="Use already prepared dataset.",
            nargs="?",  # 1 optional argument
            type=str,
        )
        parser.add_argument(
            f"--{Names.MODELS}",
            dest=f"{Names.MODELS}",
            help="Already trained model",
            nargs="+",  # 1 or more arguments
            type=str,
            default=[],
        )
        parser.add_argument(f"--{Names.CONFIG}", dest=f"{Names.CONFIG}", type=open, action=_LoadFromFile)
        cli_args = vars(parser.parse_args())
        return cli_args

    def validator(_args: vars):
        parser = argparse.ArgumentParser()

        if not any(_args.values()):
            parser.error("No argument provided")

        # All
        if not any(itemgetter(Names.TRAIN, Names.TEST, Names.DATASET, Names.CREATE_DATASET)(_args)):
            parser.error("Please choose at least one action to do.")

        # model
        for model in _args[Names.MODELS]:
            if not os.path.exists(model):
                parser.error("Model not found")

        # Dataset
        if _args[Names.DATASET] is not None and not os.path.exists(_args[Names.DATASET]):
            parser.error("Dataset not found")

    args = get_argparse()
    validator(args)
    return Args(**args)
