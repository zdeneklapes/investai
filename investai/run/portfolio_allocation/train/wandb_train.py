# -*- coding: utf-8 -*-
# TODO: Wandb Sweep:
# TODO: Add to action +1 more action from 30 actions increase to 31 actions, because Agent can als decide for cash
# TODO: all algorithms
# TODO: tests
# TODO: next datasets
# TODO: Put into dataset change of price form one index to another index: e.g. 10->15=0.5, 10->5=-0.5
import os
from typing import Union
from pathlib import Path
from copy import deepcopy  # noqa

import wandb
from wandb.sdk.wandb_run import Run
from wandb.sdk.lib.disabled import RunDisabled
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback

from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.portfolio_allocation.envs.portfolioallocationenv import PortfolioAllocationEnv
from run.shared.hyperparameters.sweep_configuration import sweep_configuration
from run.shared.callbacks import WandbCallbackExtendMemory
from run.shared.callbacks import TensorboardCallback
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program


class Train:
    ALGORITHMS = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "td3": TD3,
        "dqn": DQN,
        "ddpg": DDPG,
    }

    def __init__(self, program: Program, dataset: StockFaDailyDataset, algorithm: str):
        self.stock_dataset: StockFaDailyDataset = dataset
        self.program: Program = program
        self.algorithm: str = algorithm
        self.model_path = self.program.project_structure.models.joinpath(f"{self.algorithm}.zip")

    def _init_hyper_parameters(self) -> dict:
        """Hyper parameters"""
        # Algorithm parameters info
        algorithm_init_parameter_count: int = Train.ALGORITHMS[self.algorithm].__init__.__code__.co_argcount
        algorithm_init_parameters = set(
            Train.ALGORITHMS[self.algorithm].__init__.__code__.co_varnames[:algorithm_init_parameter_count]
        )
        sweep_config = {}

        if self.program.args.wandb_sweep:
            # Get parameter for Algo from Wandb sweep configuration
            sweep_config = deepcopy(dict(wandb.config.items()))
            sweep_config_set = set(sweep_config.keys())
            # Remove parameters that are not in algorithm
            for key in sweep_config_set - algorithm_init_parameters:
                del sweep_config[key]
        else:
            # Get parameter for Algo from CLI arguments
            # Because __code__.co_varnames returns all variables inside function and I need only function parameters
            sweep_config = {key: self.program.args.__dict__[key] for key in algorithm_init_parameters if
                            key in self.program.args.__dict__}

        # Return
        sweep_config["tensorboard_log"] = self.program.project_structure.tensorboard.as_posix()
        return sweep_config

    def _init_wandb(self) -> Union[Run, RunDisabled, None]:
        run = wandb.init(
            job_type=self.program.args.wandb_job_type,
            dir=self.program.project_structure.models.as_posix(),
            config=(None
                    if self.program.args.wandb_sweep
                    else self._init_hyper_parameters()),
            project=self.program.args.wandb_project,
            entity=os.environ.get("WANDB_ENTITY"),
            tags=[self.algorithm, "portfolio-allocation"],
            notes=f"Portfolio allocation with {self.algorithm} algorithm.",
            group=self.program.args.wandb_group,
            mode="online",
            allow_val_change=False,
            resume=None,
            force=True,  # True: User must be logged in to W&B, False: User can be logged in or not
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        return run

    def _init_environment(self):
        env = PortfolioAllocationEnv(df=self.stock_dataset.train_dataset,
                                     initial_portfolio_value=self.program.args.initial_cash,
                                     tickers=self.stock_dataset.tickers, features=self.stock_dataset.get_features(),
                                     start_data_from_index=self.program.args.start_data_from_index)
        env = Monitor(
            env,
            Path(self.program.project_structure.wandb).as_posix(),
            allow_early_resets=True)  # stable_baselines3.common.monitor.Monitor
        env = DummyVecEnv([lambda: env])
        return env

    def _init_callbacks(self):
        callbacks = CallbackList([
            TensorboardCallback(),
            ProgressBarCallback(),
            WandbCallbackExtendMemory(
                verbose=self.program.args.wandb_verbose,
                model_save_path=self.model_path.parent.as_posix() if self.program.args.wandb_model_save else None,
                model_save_freq=self.program.args.wandb_model_save_freq if self.program.args.wandb_model_save else 0,
                gradient_save_freq=self.program.args.wandb_gradient_save_freq,
            ),
        ])
        return callbacks

    def _deinit_environment(self, env):
        env.close()

    def _init_model(self, environment, callbacks):
        model = Train.ALGORITHMS[self.algorithm](
            env=environment,
            **self._init_hyper_parameters(),
        )
        model.learn(
            total_timesteps=self.program.args.total_timesteps,
            tb_log_name=f"{self.algorithm}",
            callback=callbacks
        )
        return model

    def _deinit_wandb(self):
        wandb.finish()

    def train(self) -> None:
        # Init
        run = self._init_wandb()
        environment = self._init_environment()
        callbacks = self._init_callbacks()

        # Model training
        model = self._init_model(environment, callbacks)
        model.save(self.model_path.as_posix())

        # Wandb: Log artifacts
        # Log dataset
        artifact = wandb.Artifact("dataset", type="dataset")
        artifact.add_file(self.program.args.dataset_path)
        run.log_artifact(artifact)
        # Log model
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(self.model_path.as_posix())
        run.log_artifact(artifact)

        # Wandb: summary
        wandb.define_metric("total_reward", step_metric="total_timesteps")

        # Deinit
        self._deinit_environment(environment)
        self._deinit_wandb()


def main():
    from dotenv import load_dotenv

    program = Program()
    load_dotenv(dotenv_path=program.project_structure.root.joinpath(".env").as_posix())
    dataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    for algorithm in program.args.algorithms:
        t = Train(program=program, dataset=dataset, algorithm=algorithm)
        if program.args.wandb_sweep:
            sweep_id = wandb.sweep(sweep_configuration,
                                   entity=os.environ.get("WANDB_ENTITY"),
                                   project=program.args.wandb_project, )
            wandb.agent(sweep_id,
                        function=t.train,
                        count=program.args.wandb_sweep_count, )
        else:
            t.train()


if __name__ == '__main__':
    main()
