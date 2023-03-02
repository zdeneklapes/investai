# -*- coding: utf-8 -*-

# ######################################################################################################################
# Imports
# ######################################################################################################################
#
import sys

#
sys.path.append("./ai_investing/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

#
from pathlib import Path
from typing import Any, Dict

#
import pandas as pd

#
from finrl.config import A2C_PARAMS
from finrl import config
from stable_baselines3.common.logger import configure

#
from project_configs.project_dir import ProjectDir
from project_configs.experiment_dir import ExperimentDir
from rl.envs.StockTradingEnv import StockTradingEnv
from rl.experiments.common.classes import Program
from rl.experiments.common.agents import CustomDRLAgent

# ######################################################################################################################
# Configurations
# ######################################################################################################################
algorithm_name = "a2c"

# Dataset
dataset_name = "ta_orig"
base_cols = ["date", "tic"]
data_cols = ["open", "high", "low", "close", "volume"]
features_cols = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]


# ######################################################################################################################
# Classes
# ######################################################################################################################


# ######################################################################################################################
# Helpers
# ######################################################################################################################
def configure_output(exp_dir: ExperimentDir):
    _logger = configure(exp_dir.out.root.as_posix(), ["stdout", "csv", "log", "tensorboard", "json"])
    return _logger


def get_env_kwargs(dataset: pd.DataFrame) -> Dict[str, Any]:
    # Env
    # state_space = (
    #     1
    #     + 2 * stock_dimension  # portfolio value
    #     + len(features_cols) * stock_dimension  # stock price & stock owned  # len(fundamental ratios) * len(stocks)
    # )

    stock_dimension = len(dataset["tic"].unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    # Parameters for the environment
    ENV_KWARGS = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    return ENV_KWARGS


def get_algorithm_params() -> Dict[str, Any]:
    ALGORITHM_PARAMS = {  # noqa: F841 # pylint: disable=unused-variable
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    return A2C_PARAMS  # ALGORITHM_PARAMS


# ######################################################################################################################
# Train
# ######################################################################################################################
def train(program):
    # Establish the training environment using StockTradingEnv() class
    env_kwargs = get_env_kwargs(program.stock_dataset)
    env_gym = StockTradingEnv(df=program.stock_dataset, **env_kwargs)

    # Agent
    env_train, _ = env_gym.get_sb_env()
    drl_agent = CustomDRLAgent(env=env_train, program=program)

    # Parameter for algorithm
    algorithm_params = get_algorithm_params()
    algorithm = drl_agent.get_model(
        model_kwargs=algorithm_params,
        tensorboard_log=program.exp_dir.out.tensorboard.as_posix(),
        verbose=0,
        device="cpu",
    )

    # Train
    drl_agent.train(
        model=algorithm, tb_log_name=f"tb_run_{algorithm_name}", checkpoint_freq=10_000, total_timesteps=200_000
    )


# ######################################################################################################################
# Train
# ######################################################################################################################
if __name__ == "__main__":
    program = Program(
        prj_dir=ProjectDir(root=Path(__file__).parent.parent.parent.parent.parent),
        exp_dir=ExperimentDir(root=Path(__file__).parent),
        DEBUG=False,
    )
    from rl.experiments.common.utils import get_dataset

    program.stock_dataset = get_dataset(
        pd.read_json(program.exp_dir.out.datasets.joinpath(f"{dataset_name}.csv")), purpose="train"
    )
    program.exp_dir.create_dirs()

    #
    print(f"Start: {program.stock_dataset['date'].min()}, End: {program.stock_dataset['date'].max()}")
    train(program)
