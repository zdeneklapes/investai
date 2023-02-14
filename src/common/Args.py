# -*- coding: utf-8 -*-
import argparse
from argparse import Namespace
from typing import Dict, Any, Tuple


#
# @dataclass
# class Args:
#     train: Optional[bool] = False
#     test: Optional[bool] = False
#     dataset: Optional[bool] = False
#     save_dataset: Optional[bool] = False
#     input_dataset: Optional[str] = None
#     default_dataset: Optional[bool] = None
#     models: List[str] = field(default_factory=list)
#     config: Optional[str] = None
#     stable_baseline: Optional[bool] = False
#     ray: Optional[bool] = False
#
#     def validate(self):
#         # TODO
#         pass
#

class _LoadArgumentsFromFile(argparse.Action):
    """Source: <https://stackoverflow.com/a/27434050/14471542>"""

    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


def get_argparse() -> Tuple[Dict[str, Any], Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Will train models based on hyper parameters",
                        action="store_true", )
    parser.add_argument("--test", help="Will test trained models.",
                        action="store_true", )
    parser.add_argument("--dataset", help="Will test trained models.",
                        action="store_true", )
    parser.add_argument("--save_dataset", help="Prepare and save dataset as csv into: {ProjectDir().model.root}",
                        action="store_true", )
    parser.add_argument("--input_dataset", help="Use already prepared dataset.",
                        nargs="?", )  # 1 optional argument type=str, )
    parser.add_argument("--default_dataset", help="Default preprocessed dataset will be used",
                        action="store_true", )
    parser.add_argument("--models", help="Already trained model",
                        nargs="+", )  # 1 or more arguments type=str, default=[], )
    parser.add_argument("--stable_baseline", help="Use stable-baselines3",
                        action="store_true", )
    parser.add_argument("--ray", help="Use ray-rllib",
                        action="store_true", )
    parser.add_argument("--config", help="Configuration file",
                        type=open, action=_LoadArgumentsFromFile)
    return vars(parser.parse_args()), parser.parse_args()

# class Names(enum.Enum):
#     #
#     config = "configuration"
#     #
#     train = "train"
#     test = "test"
#     #
#     save_dataset = "save-dataset"
#     input_dataset = "input-dataset"
#     default_dataset = "default-dataset"
#     #
#     models = "m", "models"
#     #
#     stable_baseline = "sb3", "stable-baselines3"
#     ray = "r", "ray"
#
#
# def argument_parser():
#     args = Args(**get_argparse())
#     args.validate()
#     return args
