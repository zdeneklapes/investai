# -*- coding: utf-8 -*-
# TODO: wandb_test , How to test wandb_train
# TODO: Add to action +1 more action from 30 actions increase to 31 actions, because Agent can als decide for cash
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
from run.shared.callbacks import WandbCallbackExtendMemory
from run.shared.callbacks import TensorboardCallback
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program


class WandbTrain:
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
        """Hyperparameters for algorithm"""
        config = {}

        # Because __code__.co_varnames returns all variables inside function and I need only function parameters
        algorithm_init_parameter_count: int = WandbTrain.ALGORITHMS[self.algorithm].__init__.__code__.co_argcount
        algorithm_init_parameters = set(
            WandbTrain.ALGORITHMS[self.algorithm].__init__.__code__.co_varnames[:algorithm_init_parameter_count]
        )

        # Get parameter for Algo from Wandb sweep configuration
        if self.program.args.wandb_sweep:
            config = deepcopy(dict(wandb.config.items()))
            sweep_config_set = set(config.keys())
            # Remove parameters that are not in algorithm
            for key in sweep_config_set - algorithm_init_parameters:
                del config[key]

        # Get parameter for Algo from CLI arguments
        else:
            config = {key: self.program.args.__dict__[key]
                      for key in algorithm_init_parameters
                      if key in self.program.args.__dict__}

        config["tensorboard_log"] = self.program.project_structure.tensorboard.as_posix()
        return config

    def _init_wandb(self) -> Union[Run, RunDisabled, None]:
        run = wandb.init(
            job_type=self.program.args.wandb_job_type,
            config=(None
                    if self.program.args.wandb_sweep
                    else self._init_hyper_parameters()),
            project=self.program.args.wandb_project,
            entity=os.environ.get("WANDB_ENTITY"),
            tags=[self.algorithm, "portfolio-allocation"],
            notes=f"Portfolio allocation with {self.algorithm} algorithm.",
            group=self.program.args.wandb_group,
            mode=self.program.args.wandb_mode,
            allow_val_change=False,
            resume=None,
            force=True,  # True: User must be logged in to W&B, False: User can be logged in or not
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        os.environ["WANDB_DIR"] = self.program.project_structure.models.as_posix()
        return run

    def _init_environment(self):
        env = PortfolioAllocationEnv(df=self.stock_dataset.train_dataset,
                                     initial_portfolio_value=self.program.args.initial_cash,
                                     tickers=self.stock_dataset.tickers,
                                     features=self.stock_dataset.get_features(),
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
        ])

        if self.program.is_wandb_enabled():
            callbacks.callbacks.append(WandbCallbackExtendMemory(
                verbose=self.program.args.wandb_verbose,
                model_save_path=self.model_path.parent.as_posix() if self.program.args.wandb_model_save else None,
                model_save_freq=self.program.args.wandb_model_save_freq if self.program.args.wandb_model_save else 0,
                gradient_save_freq=self.program.args.wandb_gradient_save_freq
            ))

        return callbacks

    def _deinit_environment(self, env):
        env.close()

    def _init_model(self, environment, callbacks):
        model = WandbTrain.ALGORITHMS[self.algorithm](
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
        # Initialize
        if self.program.is_wandb_enabled():
            run = self._init_wandb()
        environment = self._init_environment()
        callbacks = self._init_callbacks()

        # Model training
        model = self._init_model(environment, callbacks)
        model.save(self.model_path.as_posix())

        # Wandb: Log artifacts
        if self.program.is_wandb_enabled():
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

            # Wandb: Deinit
            self._deinit_wandb()

        # Deinit
        self._deinit_environment(environment)


def main():
    from dotenv import load_dotenv

    program = Program()
    load_dotenv(dotenv_path=program.project_structure.root.joinpath(".env").as_posix())
    dataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    for algorithm in program.args.algorithms:
        program.log.info(f"START Training {algorithm} algorithm.")
        WandbTrain(program=program, dataset=dataset, algorithm=algorithm).train()
        program.log.info(f"END Training {algorithm} algorithm.")


if __name__ == '__main__':
    main()
