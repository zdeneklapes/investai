# -*- coding: utf-8 -*-
import os
from os import path
import sys
import cProfile
import pstats
import argparse
from argparse import Namespace
from typing import Dict, Tuple
from distutils.util import strtobool

from shared.baseexitcode import BaseExitCode


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
    # nargs="?": 1 optional argument
    # nargs="*": 0 or more arguments
    # nargs="+": 1 or more arguments

    parser = argparse.ArgumentParser()
    # Project arguments
    parser.add_argument("--train", help="Will train models based on hyper parameters", action="store_true", )
    parser.add_argument("--test", help="Will test trained models", action="store_true", )
    parser.add_argument("--dataset", help="Will test trained models", nargs="?", default="dataset.csv")
    parser.add_argument("--prepare-dataset", help="Prepare and save dataset as csv", action="store_true", )
    parser.add_argument("--models", help="Already trained model",
                        nargs="+", )  # 1 or more arguments type=str, default=[], )
    parser.add_argument("--stable_baseline", help="Use stable-baselines3", action="store_true", )
    parser.add_argument("--ray", help="Use ray-rllib", action="store_true", )
    parser.add_argument("--config", help="Configuration file", type=open, action=_LoadArgumentsFromFile)
    parser.add_argument("--debug", help="Debug mode", action="store_true", )

    # Training arguments
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256, help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2, help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1, help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3, help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
                        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="the number of steps to run in the environment for each training step")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="the coefficient of the entropy")
    parser.add_argument("--learning-rate", type=float, default=7e-4, help="the learning rate of the optimizer")

    return vars(parser.parse_args()), parser.parse_args()
