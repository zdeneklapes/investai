# -*- coding: utf-8 -*-
from os import path
import sys
import cProfile
import pstats
import argparse
from argparse import Namespace
from typing import Dict, Tuple

from common.baseexitcode import BaseExitCode


class Util:
    @staticmethod
    def check_paths(*, params: Dict[str, str]):
        for key, val in params.items():
            if key.find("dir") != -1 and not path.isdir(str(val)):
                # TODO: LOGGER_STREAM.error(f'Bad directory|{key}: {val}')
                print(f"Bad directory|{key}: {val}")  # TODO: Remove this
                sys.exit(BaseExitCode.BAD_PARAMS)


def now_time(_format: str = "%Y-%m-%dT%H-%M-%S") -> str:
    import datetime

    return datetime.datetime.now().strftime(_format)


def line_profiler_stats(func):
    def wrapper(*args, **kwargs):
        import line_profiler

        time_profiler = line_profiler.LineProfiler()
        try:
            return time_profiler(func)(*args, **kwargs)
        finally:
            time_profiler.print_stats()

    return wrapper


def profileit(func):
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        ps = pstats.Stats(prof).sort_stats("cumtime")
        ps.print_stats()
        return retval

    return wrapper


def cProfile_decorator(sort_by: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            try:
                return func(*args, **kwargs)
            finally:
                pr.disable()
                pr.print_stats(sort=sort_by)

        return wrapper

    return decorator


# This function reload the module
def reload_module(module):
    import importlib
    importlib.reload(module)


class _LoadArgumentsFromFile(argparse.Action):
    """Source: <https://stackoverflow.com/a/27434050/14471542>"""

    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


def get_argparse() -> Tuple[vars, Namespace]:
    """
    Parse arguments from command line or file
    :return: Tuple[vars, Namespace]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Will train models based on hyper parameters",
                        action="store_true", )
    parser.add_argument("--test", help="Will test trained models.",
                        action="store_true", )
    parser.add_argument("--dataset", help="Will test trained models.",
                        nargs="?",
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
