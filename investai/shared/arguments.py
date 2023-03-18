# -*- coding: utf-8 -*-
import argparse
import os
from argparse import Namespace
from distutils.util import strtobool  # noqa
from pathlib import Path
from pprint import pprint  # noqa
from typing import Tuple

from shared.utils import find_git_root


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

    # TODO: Use this: type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True
    # nargs="?": 1 optional argument
    # nargs="*": 0 or more arguments
    # nargs="+": 1 or more arguments

    def postprocess_folder_arguments(args: Namespace):
        def create_dirs(args):
            args.folder_root.mkdir(parents=True, exist_ok=True)
            args.folder_ticker.mkdir(parents=True, exist_ok=True)
            args.folder_out.mkdir(parents=True, exist_ok=True)
            args.folder_dataset.mkdir(parents=True, exist_ok=True)
            args.folder_baseline.mkdir(parents=True, exist_ok=True)
            args.folder_model.mkdir(parents=True, exist_ok=True)
            args.folder_tensorboard.mkdir(parents=True, exist_ok=True)

        #
        if args.folder_ticker is None:
            args.folder_ticker = args.folder_root.joinpath("data/ticker")
        if args.folder_out is None:
            args.folder_out: Path = args.folder_root.joinpath("out")
        if args.folder_dataset is None:
            args.folder_dataset: Path = args.folder_out.joinpath("dataset")
        if args.folder_baseline is None:
            args.folder_baseline: Path = args.folder_out.joinpath("baseline")
        if args.folder_model is None:
            args.folder_model: Path = args.folder_out.joinpath("model")
        if args.folder_tensorboard is None:
            args.folder_tensorboard: Path = args.folder_model.joinpath("tensorboard")

        #
        create_dirs(args)
        return args

    parser = argparse.ArgumentParser()
    BOOL_AS_STR_ARGUMENTS_for_parser_add_argument = dict(
        type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True
    )

    # Project arguments
    parser.add_argument(
        "--train", **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument, help="Will train models based on hyper parameters"
    )
    parser.add_argument("--test", **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument, help="Will test trained models")
    parser.add_argument("--dataset-path", "-dp", help="Will test trained models", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--dataset-split-coef",
        type=float,
        default=0.6,
        help="Define what percentage of the dataset is used for training",
    )
    parser.add_argument("--baseline-path", "-pb", help="Baseline path", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--portfolio-allocation-env", help="Portfolio allocation environment", nargs="?", type=int, default=1
    )
    parser.add_argument("--config-file", help="Configuration file", type=open, action=_LoadArgumentsFromFile)
    parser.add_argument("--debug", help="Debug mode", **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument)
    parser.add_argument(
        "--project-verbose", type=int, default=0, help="Verbosity level 0: not output 1: info 2: debug, default: 0"
    )

    # Project structure arguments
    parser.add_argument(
        "--folder-root", "-pr", help="Project root", nargs="?", type=Path, default=find_git_root(Path(__file__).parent)
    )
    parser.add_argument("--folder-ticker", help="Path to ticker data folder", nargs="?", type=Path, default=None)
    parser.add_argument("--folder-out", help="Path to output data folder", nargs="?", type=Path, default=None)
    parser.add_argument("--folder-dataset", help="Path to datasets folder", nargs="?", type=Path, default=None)
    parser.add_argument("--folder-baseline", help="Path to baselines folder", nargs="?", type=Path, default=None)
    parser.add_argument("--folder-model", help="Path to models folder", nargs="?", type=Path, default=None)
    parser.add_argument("--folder-tensorboard", help="Path to tensorboard folder", nargs="?", type=Path, default=None)

    # W&B arguments
    parser.add_argument(
        "--wandb",
        help="Wandb logging",
        **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument,
    )
    parser.add_argument(
        "--wandb-sweep",
        help="Wandb logging",
        **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument,
    )
    parser.add_argument(
        "--wandb-sweep-count",
        help="Wandb sweep count",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--wandb-model-save",
        help="Save model",
        action="store_true",
    )
    parser.add_argument(
        "--wandb-model-save-freq",
        help="Save model every x steps (0 = no checkpoint)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--wandb-gradient-save-freq",
        help="Save gradient every x steps (0 = no checkpoint)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--wandb-verbose",
        help="Verbosity level 0: not output 1: info 2: debug, default: 0",
        type=int,
        default=0,
    )

    # W&B Environment variables
    parser.add_argument(
        "--wandb-entity",
        help="Wandb entity",
        default=os.environ.get("WANDB_ENTITY", None),
    )
    parser.add_argument(
        "--wandb-project",
        help="Wandb project name",
        default=os.environ.get("WANDB_PROJECT", "portfolio-allocation"),
    )
    parser.add_argument(
        "--wandb-tags",
        help="Wandb tags",
        default=os.environ.get("WANDB_TAGS", None),
    )
    parser.add_argument(
        "--wandb-job-type",
        help="Wandb job type",
        default=os.environ.get("WANDB_JOB_TYPE", None),
    )
    parser.add_argument(
        "--wandb-run-group",
        help="Wandb run group",
        default=os.environ.get("WANDB_RUN_GROUP", None),
    )
    parser.add_argument(
        "--wandb-mode",
        help="Wandb mode",
        default=os.environ.get("WANDB_MODE", None),
    )
    parser.add_argument(
        "--wandb-dir",
        help="Wandb directory",
        type=Path,
        default=os.environ.get("WANDB_DIR", None),
    )

    # Training arguments
    parser.add_argument(
        "--algorithms",
        help="the algorithm to use",
        type=str,
        default=["ppo"],
        choices=["ppo", "a2c", "sac", "td3", "dqn", "ddpg"],
        nargs="+",
    )
    parser.add_argument(
        "--torch-deterministic",
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
        action="store_true",
    )
    parser.add_argument(
        "--track",
        help="if toggled, this experiment will be tracked with Weights and Biases",
        action="store_true",
    )
    parser.add_argument(
        "--capture-video",
        help="whether to capture videos of the agent performances (check out `videos` folder)",
        action="store_true",
    )

    # Environment arguments
    parser.add_argument(
        "--initial-cash",
        help="Initial amount of money",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--reward-scaling",
        help="Reward scaling",
        type=float,
    )
    parser.add_argument(
        "--transaction-cost",
        help="Transaction cost in $",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--start-index",
        help="Start index",
        type=int,
        default=0,
    )

    # Algorithm arguments
    parser.add_argument("--env-id", type=str, default="", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000, help="total timesteps of the experiments")
    parser.add_argument("--policy-noise", type=float, default=0.2, help="the scale of policy noise")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency of training policy (delayed)")
    parser.add_argument("--exploration-noise", type=float, default=0.1, help="the scale of exploration noise")
    parser.add_argument(
        "--noise-clip",
        type=float,
        default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--policy",
        help="the policy model to use",
        type=str,
        choices=[
            "MlpPolicy",
            "MlpLstmPolicy",
            "MlpLnLstmPolicy",
            "CnnPolicy",
            "CnnLstmPolicy",
            "CnnLnLstmPolicy",
        ],
        default="MlpPolicy",
    )
    parser.add_argument(
        "--learning-rate",
        help="the learning rate of the optimizer",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--n-steps",
        help="the number of steps to run in each environment per update",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--gamma",
        help="the discount factor gamma",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--gae-lambda",
        help="the lambda for the general advantage estimation",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--ent-coef",
        help="the coefficient of the entropy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--vf-coef",
        help="the coefficient of the value function",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-grad-norm",
        help="the maximum value for the gradient clipping",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--rms-prop-eps",
        help="RMSProp optimizer epsilon (stabilizes square root)",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--use-rms-prop",
        help="whether to use RMSProp (otherwise, use Adam)",
        action="store_true",
    )
    parser.add_argument(
        "--use-sde",
        help="whether to use generalized State Dependent Exploration",
        action="store_true",
    )
    parser.add_argument(
        "--sde-sample-freq",
        help="Sample a new noise matrix every n steps (-1 = only at the beginning of the rollout)",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--normalize-advantage",
        help="whether to normalize the advantage",
        action="store_true",
    )
    parser.add_argument(
        "--tensorboard-log",
        help="the log location for tensorboard (if None, no logging)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--policy-kwargs",
        help="Additional keyword argument to pass to the policy on creation",
        type=str,
        default="",
    )
    parser.add_argument(
        "--verbose",
        help="the verbosity level: 0 none, 1 training information, 2 tensorflow debug",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        help="Random generator seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        help="Device (cpu, cuda, auto)",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "---init-setup-model",
        help="Whether or not to setup the model with default hyperparameters",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for each gradient update",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--n-epochs",
        help="Number of epoch when learning",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--clip-range",
        help="Clipping parameter (policy loss)",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--clip-range-vf",
        help="Clipping parameter (value loss)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--target-kl",
        help="Target KL divergence",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--buffer-size",
        help="Size of the replay buffer",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--learning-starts",
        help="Number of steps before learning for the warm-up",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--tau",
        help="Target smoothing coefficient (1-tau)",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--train-freq",
        help="Update the model every ``train_freq`` steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient-steps",
        help="How many gradient update after each step",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--action-noise",
        help="Action noise type (None, normal, ou)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--replay-buffer-class",
        help="Replay buffer class",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--replay-buffer-kwargs",
        help="Additional keyword argument to pass to the replay buffer on creation",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--optimize-memory-usage",
        help="Enable a memory efficient variant of the replay buffer",
        action="store_true",
    )
    parser.add_argument(
        "--target-update-interval",
        help="Number of gradient update before each target network update",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--target-entropy",
        help="Target entropy to be used by SAC/TD3",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--use-sde-at-warmup",
        help="Use gSDE instead of normal action noise at warmup",
        action="store_true",
    )
    parser.add_argument(
        "--policy-delay",
        help="Number of timesteps to delay the policy updates",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--target-policy-noise",
        help="Noise added to target policy",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--target-noise-clip",
        help="Range to clip target policy noise",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--exploration-fraction",
        help="Fraction of entire training period over which the exploration rate is annealed",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--exploration-initial-eps",
        help="Initial value of random action probability",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--exploration-final-eps",
        help="Final value of random action probability",
        type=float,
        default=0.02,
    )

    args = postprocess_folder_arguments(parser.parse_args())
    return vars(args), args


def main():
    args_dict, args = parse_arguments()
    pprint(args_dict)


if __name__ == "__main__":
    main()
