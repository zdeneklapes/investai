# -*- coding: utf-8 -*-

# ######################################################################################################################
# Imports
# ######################################################################################################################
#
from stable_baselines3.common.logger import configure
from rl.experiments.common.classes import Program
from rl.experiments.common.agents import CustomDRLAgent
from rl.envs.StockTradingEnv import StockTradingEnv
from project_configs.project_dir import ProjectDir
from project_configs.experiment_dir import ExperimentDir
from finrl.config import A2C_PARAMS
import pandas as pd
from typing import Any, Dict
from pathlib import Path
import sys

#
sys.path.append("./ai_investing/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

#

#

#

#

# ######################################################################################################################
# Configurations
# ######################################################################################################################
algorithm_name = "a2c"

# Dataset
dataset_name = "experiment_same_bigger_fundamental"
base_cols = ["date", "tic"]
data_cols = ["open", "high", "low", "close", "volume"]
ratios_cols = [
    "operatingProfitMargin",
    "netProfitMargin",
    "returnOnAssets",
    "returnOnEquity",
    "currentRatio",
    "quickRatio",
    "cashRatio",
    "inventoryTurnover",
    "receivablesTurnover",
    "payablesTurnover",
    "debtRatio",
    "debtEquityRatio",
    "priceEarningsRatio",
    "priceBookValueRatio",
    "dividendYield",
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
    stock_dimension = len(dataset["tic"].unique())
    ENV_KWARGS = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": (
            1
            # portfolio value
            + 2 * stock_dimension
            # stock price & stock owned  # len(fa_ratios) * len(stocks)
            + len(ratios_cols) * stock_dimension
        ),
        "stock_dim": stock_dimension,
        "tech_indicator_list": ratios_cols,
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
    env_kwargs = get_env_kwargs(program.datasets)
    env_gym = StockTradingEnv(df=program.datasets, **env_kwargs)

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
        prj_dir=ProjectDir(__file__),
        exp_dir=ExperimentDir(root=Path(__file__).parent),
        DEBUG=False,
    )
    from rl.experiments.common.utils import get_dataset

    program.datasets = get_dataset(
        pd.read_csv(program.exp_dir.out.datasets.joinpath(f"{dataset_name}.csv"), index_col=0), purpose="train"
    )
    program.exp_dir.create_dirs()

    #
    print(f"Start: {program.datasets['date'].min()}, End: {program.datasets['date'].max()}")
    train(program)
