# -*- coding: utf-8 -*-
import os
from operator import itemgetter
from typing import Dict, Any
import argparse

from config.settings import DATA_DIR


class ArgNames:
    TRAIN = "train"
    TEST = "test"
    CREATE_DATASET = "create-dataset"
    DATASET = "dataset"
    MODEL = "model"
    FROM_FILE = "from-file"


class LoadFromFile(argparse.Action):
    """Source: <https://stackoverflow.com/a/27434050/14471542>"""

    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


class ArgumentParser:
    pass


def argument_validator(args: vars):
    parser = argparse.ArgumentParser()

    if not any(args.values()):
        parser.error("No argument provided")

    # All
    if not any(itemgetter(ArgNames.TRAIN, ArgNames.TEST, ArgNames.DATASET, ArgNames.CREATE_DATASET)(args)):
        parser.error("Please choose at least one action to do.")

    # model
    for model in args["model"]:
        if not os.path.exists(model):
            parser.error("Model not found")

    # Dataset
    if args["dataset"] is not None:
        if not os.path.exists(args["dataset"]):
            parser.error("Dataset not found")


def argument_parser() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        f"--{ArgNames.TRAIN}",
        dest=f"{ArgNames.TRAIN}",
        action="store_true",
        help="Will train models based on hyper parameters",
    )
    parser.add_argument(
        f"--{ArgNames.TEST}", dest=f"{ArgNames.TEST}", action="store_true", help="Will test trained models."
    )
    parser.add_argument(
        f"--{ArgNames.CREATE_DATASET}",
        dest=f"{ArgNames.CREATE_DATASET}",
        action="store_true",
        help=f"Prepare and save dataset as csv into: {DATA_DIR}",
    )
    parser.add_argument(
        f"--{ArgNames.DATASET}", dest=f"{ArgNames.DATASET}", nargs="?", help="Used already prepared dataset."
    )
    parser.add_argument(
        f"--{ArgNames.MODEL}", dest=f"{ArgNames.MODEL}", nargs="*", help="Already trained model", default=[]
    )
    parser.add_argument(f"--{ArgNames.FROM_FILE}", dest=f"{ArgNames.FROM_FILE}", type=open, action=LoadFromFile)
    cli_args = vars(parser.parse_args())
    argument_validator(cli_args)
    return cli_args