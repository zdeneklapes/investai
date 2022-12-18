# -*- coding: utf-8 -*-
#
import sys
import dataclasses
import pandas as pd
from pathlib import Path

from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent

##
sys.path.append("./src/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

#
from common.utils import now_time
from configuration.settings import ProjectDir, ExperimentDir
from rl.experiments._1_same_bigger_data_fundamental.GymEnv import StockTradingEnv


@dataclasses.dataclass
class Program:
    prj_dir: ProjectDir
    exp_dir: ExperimentDir
    DEBUG: bool = False


# ######################################################################################################################
# Configurations
# ######################################################################################################################
algorithm_name = "a2c"
learned_algorithm_name = f"{algorithm_name}_{now_time()}"
dataset_name = "experiment_same_bigger_fundamental.csv"

HYPERPARAMS = {
    "n_steps": 5,
    "ent_coef": 0.01,
    "learning_rate": 0.0002,
}

base_cols = list(["date", "tic"])

data_cols = list(["open", "high", "low", "close", "volume"])

ratios_cols = list(
    [
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
)


def configure_output(exp_dir: ExperimentDir):
    _logger = configure(exp_dir.out.root.as_posix(), ["stdout", "csv", "tensorboard"])
    return _logger


# ######################################################################################################################
# Train
# ######################################################################################################################
def train(program):
    # Data
    dataset = pd.read_csv(prj_dir.dataset.experiments.joinpath(dataset_name))

    # Env
    stock_dimension = len(dataset["tic"].unique())
    state_space = 1 + 2 * stock_dimension + len(ratios_cols) * stock_dimension
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

    # Establish the training environment using StockTradingEnv() class
    env_gym = StockTradingEnv(df=dataset, **ENV_KWARGS)

    # Agent
    env_train, _ = env_gym.get_sb_env()
    print(type(env_train))
    drl_agent = DRLAgent(env=env_train)

    # Parameter for algorithm
    ALGORITHM_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    algorithm = drl_agent.get_model(algorithm_name, model_kwargs=ALGORITHM_PARAMS)

    # Set new logger
    algorithm.set_logger(configure_output(program.exp_dir))

    # Train
    trained_algorithm = drl_agent.train_model(model=algorithm, tb_log_name=algorithm_name, total_timesteps=50000)

    # Save
    trained_algorithm.save(prj_dir.model.experiments.joinpath(learned_algorithm_name))


# ######################################################################################################################
# Train
# ######################################################################################################################
if __name__ == "__main__":
    exp_dir = ExperimentDir(root=Path(__file__).parent)
    exp_dir.check_and_create_dirs()
    prj_dir = ProjectDir(root=Path(__file__).parent.parent.parent.parent.parent)
    program = Program(prj_dir, exp_dir, DEBUG=False)
    train(program)
