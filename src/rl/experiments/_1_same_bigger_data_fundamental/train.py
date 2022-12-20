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
import numpy as np
import torch

#
from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent, TensorboardCallback
from finrl.config import A2C_PARAMS
from finrl.meta.preprocessor.preprocessors import data_split

#
from stable_baselines3.common.callbacks import ProgressBarCallback, CallbackList, BaseCallback
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

#
sys.path.append("./src/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

#
from common.utils import now_time
from configuration.settings import ProjectDir, ExperimentDir
from rl.envs.StockTradingEnv import StockTradingEnv


# ######################################################################################################################
# Classes
# ######################################################################################################################
@dataclasses.dataclass
class Program:
    prj_dir: ProjectDir
    exp_dir: ExperimentDir
    dataset: pd.DataFrame = None
    DEBUG: bool = False


class CheckpointCallback(BaseCallback):
    """
    A custom callback that saves a model every ``save_freq`` steps.
    :param save_freq:
    :param save_path:
    """

    def __init__(self, save_freq: int, save_path: Union[Path]):
        super(CheckpointCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        if not self.save_path.parent.exists():  # check directory exists
            raise FileNotFoundError(f"Directory not found: {self.save_path.parent.as_posix()}")
        else:
            print(f"Learned algorithm will be saved to: {self.save_path.as_posix()}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path.as_posix() + f"_{self.n_calls}")
        return True

    def _on_training_end(self) -> None:
        self.model.save(self.save_path.as_posix() + f"_{self.n_calls}")


class CustomDRLAgent(DRLAgent):
    def __init__(self, env, program: Program):
        super().__init__(env)
        self.program: Program = program

    def train_model(
        self, model, tb_log_name="run", total_timesteps=1000000, checkpoint_freq: int = 10000, **kwargs
    ) -> Union[A2C, DDPG, PPO, SAC, TD3]:
        callback_list = CallbackList(
            [
                TensorboardCallback(),
                ProgressBarCallback(),
                CheckpointCallback(
                    checkpoint_freq, program.exp_dir.out.algorithms.joinpath(f"{algorithm_name}_{now_time()}")
                ),
            ]
        )
        learned_algo = model.learn(
            total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=callback_list, **kwargs
        )
        return learned_algo

    def get_model(
        self,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        device: Union[torch.device, "str"] = "cpu",
        tensorboard_log=None,
    ):
        #
        if "action_noise" in model_kwargs:
            NOISE = {"normal": NormalActionNoise, "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise}
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )

        #
        return A2C(
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            **model_kwargs,
        )


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
    # Establish the training environment using StockTradingEnv() class
    env_kwargs = get_env_kwargs(program.dataset)
    env_gym = StockTradingEnv(df=program.dataset, **env_kwargs)

    # Agent
    env_train, _ = env_gym.get_sb_env()
    drl_agent = CustomDRLAgent(env=env_train, program=program)

    # Parameter for algorithm
    algorithm_params = get_algorithm_params()
    algorithm = drl_agent.get_model(
        model_kwargs=algorithm_params,
        tensorboard_log=program.exp_dir.out.tensorboard.as_posix() + "/",
        verbose=0,
        device="cpu",
    )

    # Train
    drl_agent.train_model(
        model=algorithm, tb_log_name=f"tb_run_{algorithm_name}", checkpoint_freq=10_000, total_timesteps=200_000
    )


def get_dataset(
    df, purpose: Literal["train", "test"], date_col_name: str = "date", split_constant: int = 0.75
) -> pd.DataFrame:
    split_date = df.iloc[int(df.index.size * split_constant)]["date"]
    if purpose == "train":
        return data_split(df, df[date_col_name].min(), split_date, date_col_name)  # df[df["date"] >= date_split]
    elif purpose == "test":
        return data_split(df, split_date, df[date_col_name].max(), date_col_name)  # df[df["date"] >= date_split]
    else:
        raise ValueError(f"Unknown purpose: {purpose}")


# ######################################################################################################################
# Train
# ######################################################################################################################
if __name__ == "__main__":
    program = Program(
        prj_dir=ProjectDir(root=Path(__file__).parent.parent.parent.parent.parent),
        exp_dir=ExperimentDir(root=Path(__file__).parent),
        DEBUG=False,
    )
    program.dataset = get_dataset(
        pd.read_csv(program.exp_dir.out.datasets.joinpath(f"{dataset_name}.csv"), index_col=0), purpose="train"
    )
    program.exp_dir.check_and_create_dirs()

    #
    print(f"Start: {program.dataset['date'].min()}, End: {program.dataset['date'].max()}")
    train(program)
