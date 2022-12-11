# -*- coding: utf-8 -*-
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import argparse
import enum

from configuration.settings import ModelDir


@dataclass
class Args:
    train: Optional[bool] = False
    test: Optional[bool] = False
    create_dataset: Optional[bool] = False
    input_dataset: Optional[str] = None
    default_dataset: Optional[bool] = None
    models: List[str] = field(default_factory=list)
    config: Optional[str] = None
    stable_baseline: Optional[bool] = False
    ray: Optional[bool] = False

    def validate(self):
        """TODO"""
        # parser = argparse.ArgumentParser()
        #
        # if not any(_args.values()):
        #     parser.error("No argument provided")
        #
        # # All
        # if not any(itemgetter(Names.TRAIN, Names.TEST, Names.INPUT_DATASET[1], Names.CREATE_DATASET[0])(_args)):
        #     parser.error("Please choose at least one action to do.")
        #
        # # model
        # for model in _args[Names.MODELS]:
        #     if not os.path.exists(model):
        #         parser.error("Model not found")
        #
        # # Dataset
        # if _args[Names.INPUT_DATASET] is not None and not os.path.exists(_args[Names.INPUT_DATASET]):
        #     parser.error("Dataset not found")


class _LoadFromFile(argparse.Action):
    """Source: <https://stackoverflow.com/a/27434050/14471542>"""

    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


def argument_parser():
    class Names(enum.Enum):
        #
        config = "configuration"
        #
        train = "train"
        test = "test"
        #
        create_dataset = "create-dataset"
        input_dataset = "input-dataset"
        default_dataset = "default-dataset"
        #
        models = "models"
        #
        stable_baseline = "stable-baselines3"
        ray = "ray"

    def get_argparse() -> Dict[str, Any]:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            f"--{Names.train.value}",
            dest=f"{Names.train.name}",
            help="Will train models based on hyper parameters",
            action="store_true",
        )
        parser.add_argument(
            f"--{Names.test.value}",
            dest=f"{Names.test.name}",
            help="Will test trained models.",
            action="store_true",
        )
        parser.add_argument(
            f"--{Names.create_dataset.value}",
            dest=f"{Names.create_dataset.name}",
            help=f"Prepare and save dataset as csv into: {ModelDir.ROOT}",
            action="store_true",
        )
        parser.add_argument(
            f"--{Names.input_dataset.value}",
            dest=f"{Names.input_dataset.name}",
            help="Use already prepared dataset.",
            nargs="?",  # 1 optional argument
            type=str,
        )
        parser.add_argument(
            f"--{Names.default_dataset.value}",
            dest=f"{Names.default_dataset.name}",
            help="Default preprocessed dataset will be used",
            action="store_true",
        )
        parser.add_argument(
            f"--{Names.models.value}",
            dest=f"{Names.models.name}",
            help="Already trained model",
            nargs="+",  # 1 or more arguments
            type=str,
            default=[],
        )
        parser.add_argument(
            f"--{Names.stable_baseline.value}",
            dest=f"{Names.stable_baseline.name}",
            help="Use stable-baselines3",
            action="store_true",
        )
        parser.add_argument(
            f"--{Names.ray.value}",
            dest=f"{Names.ray.name}",
            help="Use ray-rllib",
            action="store_true",
        )
        parser.add_argument(f"--{Names.config.value}", dest=f"{Names.config.name}", type=open, action=_LoadFromFile)
        cli_args = vars(parser.parse_args())
        return cli_args

    args = Args(**get_argparse())
    args.validate()
    return args
