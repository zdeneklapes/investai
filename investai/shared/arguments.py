# -*- coding: utf-8 -*-
import os
import argparse
from argparse import Namespace
from typing import Tuple
from pprint import pprint  # noqa
from distutils.util import strtobool  # noqa


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

    parser = argparse.ArgumentParser()
    BOOL_AS_STR_ARGUMENTS_for_parser_add_argument = \
        dict(type=lambda x: bool(strtobool(x)),
             default=False,
             nargs="?",
             const=True)

    # Project arguments
    parser.add_argument("--train", **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument,
                        help="Will train models based on hyper parameters")
    parser.add_argument("--test", **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument,
                        help="Will test trained models")
    parser.add_argument("--dataset-path", "-dp", help="Will test trained models", nargs="?", type=str, default=None)
    parser.add_argument("--dataset-split-coef", type=float, default=0.6,
                        help="Define what percentage of the dataset is used for training")
    parser.add_argument("--models", help="Already trained model", nargs="+", )
    parser.add_argument("--stable_baseline", help="Use stable-baselines3", action="store_true", )
    parser.add_argument("--ray", help="Use ray-rllib", action="store_true", )
    parser.add_argument("--config-file", help="Configuration file", type=open, action=_LoadArgumentsFromFile)
    parser.add_argument("--debug",
                        help="Debug mode",
                        **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument)
    parser.add_argument('--project-verbose', type=int, default=0,
                        help="Verbosity level 0: not output 1: info 2: debug, default: 0")

    # Wandb arguments
    parser.add_argument("--wandb", **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument, help="Wandb logging", )
    parser.add_argument("--wandb-sweep", **BOOL_AS_STR_ARGUMENTS_for_parser_add_argument, help="Wandb logging", )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "investai"),
                        help="Wandb project name")
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", None),
                        help="Wandb directory")
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP", None),
                        help="Wandb run group")
    parser.add_argument("--wandb-job-type", default=os.environ.get("WANDB_JOB_TYPE", None),
                        help="Wandb job type")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", None),
                        help="Wandb mode")
    parser.add_argument("--wandb-tags", default=os.environ.get("WANDB_TAGS", None),
                        help="Wandb tags")
    parser.add_argument("--wandb-sweep-count", type=int, default=1, help="Wandb sweep count")
    parser.add_argument("--wandb-model-save", action="store_true", help="Save model")
    parser.add_argument("--wandb-model-save-freq", type=int, default=100,
                        help="Save model every x steps (0 = no checkpoint)")
    parser.add_argument("--wandb-gradient-save-freq", type=int, default=100,
                        help="Save gradient every x steps (0 = no checkpoint)")
    parser.add_argument("--wandb-verbose", type=int, default=0,
                        help="Verbosity level 0: not output 1: info 2: debug, default: 0")

    # Training arguments
    parser.add_argument("--algorithms", type=str, default=["ppo"], choices=["ppo", "a2c", "sac", "td3", "dqn", "ddpg"],
                        nargs="+", help="the algorithm to use")
    parser.add_argument("--torch-deterministic", action="store_true",
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", action="store_true",
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--capture-video", action="store_true",
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Environment arguments
    parser.add_argument("--initial-cash", type=int, default=100_000, help="Initial amount of money")
    parser.add_argument("--reward-scaling", type=float, help="Reward scaling")
    parser.add_argument("--transaction-cost", type=float, default=0.5, help="Transaction cost in $")
    parser.add_argument("--start-data-from-index", type=int, default=0, help="Start raw_data from index")

    # Algorithm arguments
    parser.add_argument("--env-id", type=str, default="", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000, help="total timesteps of the experiments")
    parser.add_argument("--policy-noise", type=float, default=0.2, help="the scale of policy noise")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency of training policy (delayed)")
    parser.add_argument("--exploration-noise", type=float, default=0.1, help="the scale of exploration noise")
    parser.add_argument("--noise-clip", type=float, default=0.5,
                        help="noise clip parameter of the Target Policy Smoothing Regularization")

    # Algorithm specific arguments
    parser.add_argument('--policy', type=str,
                        choices=["MlpPolicy", "MlpLstmPolicy", "MlpLnLstmPolicy",
                                 "CnnPolicy", "CnnLstmPolicy", "CnnLnLstmPolicy", ],
                        default="MlpPolicy", help="the policy model to use")
    parser.add_argument('--learning-rate', type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument('--n-steps', type=int, default=5,
                        help="the number of steps to run in each environment per update")
    parser.add_argument('--gamma', type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument('--ent-coef', type=float, default=0.0, help="the coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5, help="the coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help="the maximum value for the gradient clipping")
    parser.add_argument('--rms-prop-eps', type=float, default=1e-5,
                        help="RMSProp optimizer epsilon (stabilizes square root)")
    parser.add_argument('--use-rms-prop', action="store_true", help="whether to use RMSProp (otherwise, use Adam)")
    parser.add_argument('--use-sde', action="store_true", help="whether to use generalized State Dependent Exploration")
    parser.add_argument('--sde-sample-freq', type=int, default=-1,
                        help="Sample a new noise matrix every n steps (-1 = only at the beginning of the rollout)")
    parser.add_argument('--normalize-advantage', action="store_true", help="whether to normalize the advantage")
    # parser.add_argument('--tensorboard-log', type=str, default=None,
    #                     help="the log location for tensorboard (if None, no logging)")
    # parser.add_argument('--policy-kwargs', type=str, default="",
    #                     help="Additional keyword argument to pass to the policy on creation")
    parser.add_argument('--verbose', type=int, default=0,
                        help="the verbosity level: 0 none, 1 training information, 2 tensorflow debug")
    parser.add_argument('--seed', type=int, default=0, help="Random generator seed")
    parser.add_argument('--device', type=str, default='auto', help="Device (cpu, cuda, auto)")
    parser.add_argument('---init-setup-model', action="store_true",
                        help="Whether or not to setup the model with default hyperparameters")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for each gradient update")
    parser.add_argument('--n-epochs', type=int, default=10, help="Number of epoch when learning")
    parser.add_argument('--clip-range', type=float, default=0.2, help="Clipping parameter (policy loss)")
    parser.add_argument('--clip-range-vf', type=float, default=None, help="Clipping parameter (value loss)")
    parser.add_argument('--target-kl', type=float, default=None, help="Target KL divergence")
    parser.add_argument('--buffer-size', type=int, default=50000, help="Size of the replay buffer")
    parser.add_argument('--learning-starts', type=int, default=1000,
                        help="Number of steps before learning for the warm-up")
    parser.add_argument('--tau', type=float, default=0.005, help="Target smoothing coefficient (1-tau)")
    parser.add_argument('--train-freq', type=int, default=1, help="Update the model every ``train_freq`` steps")
    parser.add_argument('--gradient-steps', type=int, default=-1,
                        help="How many gradient update after each step")
    parser.add_argument('--action-noise', type=str, default=None, help="Action noise type (None, normal, ou)")
    parser.add_argument('--replay-buffer-class', type=str, default=None, help="Replay buffer class")
    parser.add_argument('--replay-buffer-kwargs', type=str, default=None,
                        help="Additional keyword argument to pass to the replay buffer on creation")
    parser.add_argument('--optimize-memory-usage', action="store_true",
                        help="Enable a memory efficient variant of the replay buffer")
    parser.add_argument('--target-update-interval', type=int, default=1,
                        help="Number of gradient update before each target network update")
    parser.add_argument('--target-entropy', type=str, default="auto", help="Target entropy to be used by SAC/TD3")
    parser.add_argument('--use-sde-at-warmup', action="store_true",
                        help="Use gSDE instead of normal action noise at warmup")
    parser.add_argument('--policy-delay', type=int, default=2, help="Number of timesteps to delay the policy updates")
    parser.add_argument('--target-policy-noise', type=float, default=0.2, help="Noise added to target policy")
    parser.add_argument('--target-noise-clip', type=float, default=0.5, help="Range to clip target policy noise")
    parser.add_argument('--exploration-fraction', type=float, default=0.1,
                        help="Fraction of entire training period over which the exploration rate is annealed")
    parser.add_argument('--exploration-initial-eps', type=float, default=1.0,
                        help="Initial value of random action probability")
    parser.add_argument('--exploration-final-eps', type=float, default=0.02,
                        help="Final value of random action probability")

    # TODO: Policy specific arguments

    return vars(parser.parse_args()), parser.parse_args()


def main():
    print(f"==== START {main.__name__}() ====")
    print("=== Parsing arguments ===")
    args_as_dict, args = parse_arguments()
    pprint(f"args_as_dict: {args_as_dict}")
    pprint(args.wandb_sweep)
    print(f"==== END {main.__name__}() ====")


if __name__ == '__main__':
    main()
