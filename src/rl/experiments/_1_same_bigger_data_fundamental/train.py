# -*- coding: utf-8 -*-

# ######################################################################################################################
# Imports
# ######################################################################################################################
#
import sys
import dataclasses
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Literal, Union

#
from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent, TensorboardCallback
from finrl.config import A2C_PARAMS

#
from stable_baselines3.common.callbacks import ProgressBarCallback, CallbackList
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

#
sys.path.append("./src/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

#
from common.utils import now_time
from configuration.settings import ProjectDir, ExperimentDir
from rl.experiments._1_same_bigger_data_fundamental.StockTradingEnv import StockTradingEnv


# ######################################################################################################################
# Classes
# ######################################################################################################################
@dataclasses.dataclass
class Program:
    prj_dir: ProjectDir
    exp_dir: ExperimentDir
    dataset: pd.DataFrame = None
    DEBUG: bool = False


class CustomDRLAgent(DRLAgent):
    def train_model(
        self, model, tb_log_name="run", total_timesteps=1000000, **kwargs
    ) -> Union[A2C, DDPG, PPO, SAC, TD3]:
        callback_list = CallbackList([TensorboardCallback(), ProgressBarCallback()])
        return model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=callback_list, **kwargs)


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

DATASET_SPLIT_CONSTANT = 0.8


# ######################################################################################################################
# Helpers
# ######################################################################################################################
def configure_output(exp_dir: ExperimentDir):
    _logger = configure(exp_dir.out.root.as_posix(), ["stdout", "csv", "log", "tensorboard", "json"])
    return _logger


def get_env_kwargs(dataset: pd.DataFrame) -> Dict[str, Any]:
    # Env
    stock_dimension = len(dataset["tic"].unique())
    state_space = (
        1
        + 2 * stock_dimension  # portfolio value
        + len(ratios_cols) * stock_dimension  # stock price & stock owned  # len(fundamental ratios) * len(stocks)
    )
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    # Parameters for the environment
    ENV_KWARGS = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
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
    # path to save learned algorithm
    learned_algorithm_path = program.exp_dir.out.algorithms.joinpath(f"{algorithm_name}_{now_time()}")
    if not learned_algorithm_path.parent.exists():  # check directory exists
        raise FileNotFoundError(f"Directory not found: {learned_algorithm_path.parent.as_posix()}")
    else:
        print(f"Learned algorithm will be saved to: {learned_algorithm_path.as_posix()}")

    # Establish the training environment using StockTradingEnv() class
    env_kwargs = get_env_kwargs(program.dataset)
    env_gym = StockTradingEnv(df=program.dataset, **env_kwargs)

    # Agent
    env_train, _ = env_gym.get_sb_env()
    drl_agent = CustomDRLAgent(
        env=env_train,
    )

    # Parameter for algorithm
    algorithm_params = get_algorithm_params()
    algorithm = drl_agent.get_model(
        algorithm_name, tensorboard_log=program.exp_dir.out.tensorboard.as_posix() + "/", model_kwargs=algorithm_params
    )

    # Train
    trained_algorithm = drl_agent.train_model(
        model=algorithm, tb_log_name=f"tb_run_{algorithm_name}", total_timesteps=100000
    )

    # Save
    trained_algorithm.save(learned_algorithm_path.as_posix())


def get_dataset(df, purpose: Literal["train", "test"], split_constant: int = 0.8) -> pd.DataFrame:
    date_split = df.iloc[int(df.index.size * split_constant)]["date"]
    if purpose == "train":
        return df[df["date"] < date_split]
    elif purpose == "test":
        return df[df["date"] >= date_split]
    else:
        raise ValueError(f"Unknown type: {purpose}")


# ######################################################################################################################
# Train
# ######################################################################################################################
if __name__ == "__main__":
    program = Program(
        prj_dir=ProjectDir(root=Path(__file__).parent.parent.parent.parent.parent),
        exp_dir=ExperimentDir(root=Path(__file__).parent),
        DEBUG=False,
    )
    program.dataset = get_dataset(pd.read_csv(program.prj_dir.data.joinpath(f"{dataset_name}.csv")), purpose="train")
    program.exp_dir.check_and_create_dirs()

    #
    train(program)
