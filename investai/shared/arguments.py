# -*- coding: utf-8 -*-
import argparse
import os
from argparse import Namespace
from typing import Tuple
from distutils.util import strtobool


class _LoadArgumentsFromFile(argparse.Action):
    """Source: <https://stackoverflow.com/a/27434050/14471542>"""

    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


def parse_arguments() -> Tuple[vars, Namespace]:
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
    parser.add_argument("--dataset-name", "-dn", help="Will test trained models", nargs="?", default="dataset.csv")
    parser.add_argument("--dataset-split", type=float, default=0.8,
                        help="Define what percentage of the dataset is used for training")
    parser.add_argument("--models", help="Already trained model",
                        nargs="+", )  # 1 or more arguments type=str, default=[], )
    parser.add_argument("--stable_baseline", help="Use stable-baselines3", action="store_true", )
    parser.add_argument("--ray", help="Use ray-rllib", action="store_true", )

    parser.add_argument("--config-file", help="Configuration file", type=open, action=_LoadArgumentsFromFile)
    parser.add_argument("--debug", help="Debug mode", action="store_true", default=os.environ.get("DEBUG", False))

    # Wandb arguments
    parser.add_argument("--wandb-project", type=str, default=None, help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-model-save-freq", type=int, default=0,
                        help="Save model every x steps (0 = no checkpoint)")
    parser.add_argument("--wandb-gradient-save-freq", type=int, default=100,
                        help="Save gradient every x steps (0 = no checkpoint)")
    parser.add_argument("--wandb-verbose", type=int, default=2, help="Verbosity level 0: not output 1: info 2: debug")

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
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000, help="total timesteps of the experiments")
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
    parser.add_argument("--entropy-coefficient", type=float, default=0.01, help="the coefficient of the entropy")
    parser.add_argument("--learning-rate", type=float, default=7e-4, help="the learning rate of the optimizer")

    # Environment arguments
    parser.add_argument("--initial-cash", type=int, default=100_000, help="Initial amount of money")
    parser.add_argument("--reward-scaling", type=float, help="Reward scaling")
    parser.add_argument("--transaction-cost", type=float, default=0.5, help="Transaction cost in $")
    parser.add_argument("--start-data-from-index", type=int, default=0, help="Start data from index")

    return vars(parser.parse_args()), parser.parse_args()
