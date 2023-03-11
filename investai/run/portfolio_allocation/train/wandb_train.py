# -*- coding: utf-8 -*-
# TODO: wandb sweep
# TODO: all algorithms
# TODO: tests
# TODO: next datasets
import os
from copy import deepcopy
from typing import Union
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
import wandb
from wandb.sdk.wandb_run import Run
from wandb.sdk.lib.disabled import RunDisabled
from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from run.shared.tickers import DOW_30_TICKER

from run.shared.algorithm_parameters import STABLE_BASELINE_PARAMETERS
from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.portfolio_allocation.envs.portfolioallocationenv import PortfolioAllocationEnv
from run.shared.callbacks import WandbCallbackExtendMemory
from run.shared.callbacks import TensorboardCallback
from shared.program import Program
from shared.dir.experiment_dir import ExperimentDir


class Train:
    def __init__(self, program: Program, dataset: StockFaDailyDataset):
        self.stock_dataset: StockFaDailyDataset = dataset
        self.program: Program = program
        self.algorithm: str = "ppo"
        self.model_path = self.program.experiment_dir.models.joinpath(f"{self.algorithm}.zip")

    def _init_hyper_parameters(self):
        """Hyper parameters for PPO algorithm"""
        default_hyper_parameters = deepcopy(STABLE_BASELINE_PARAMETERS[self.algorithm])
        intersect_keys = (
            set(self.program.args.__dict__.keys())
            .intersection(set(default_hyper_parameters.keys()))
        )
        default_keys = set(default_hyper_parameters.keys()).difference(intersect_keys)

        cli_config = {key: self.program.args.__dict__[key] for key in intersect_keys}
        default_config: dict = {
            key: default_hyper_parameters[key]
            for key in default_keys
        }

        assert len(set(cli_config.keys()).intersection(set(default_config.keys()))) == 0, \
            "cli_config keys and default_config keys should be unique"

        return cli_config | default_config

    def _init_wandb(self) -> Union[Run, RunDisabled, None]:
        if self.program.args.sweep:
            return wandb.init()
        else:
            run = wandb.init(
                job_type="train",
                dir=self.program.experiment_dir.models.as_posix(),
                config=self._init_hyper_parameters(),
                project=os.environ.get("WANDB_PROJECT"),
                entity=os.environ.get("WANDB_ENTITY"),
                tags=["train", "ppo", "portfolio-allocation"],
                notes="Training PPO on DJI30 stocks",
                group="experiment_1",  # TIP: Can be used environment variable "WANDB_RUN_GROUP", Must be unique
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
        env = Monitor(env, wandb.run.dir, allow_early_resets=True)  # stable_baselines3.common.monitor.Monitor
        # env = Monitor(env, wandb.run.dir) # gym.wrappers.Monitor
        env = DummyVecEnv([lambda: env])
        return env

    def _init_callbacks(self):
        callbacks = CallbackList([
            TensorboardCallback(),
            ProgressBarCallback(),
            WandbCallbackExtendMemory(
                verbose=self.program.args.verbose,
                model_save_path=self.model_path.parent.as_posix() if self.program.args.wandb_model_save else None,
                model_save_freq=self.program.args.wandb_model_save_freq if self.program.args.wandb_model_save else 0,
                gradient_save_freq=self.program.args.wandb_gradient_save_freq,
            ),
        ])
        return callbacks

    def _init_model(self, env, callbacks):
        model = PPO(
            tensorboard_log=self.program.experiment_dir.tensorboard.as_posix(),
            env=env,
            **wandb.config,
        )
        model.learn(total_timesteps=self.program.args.total_timesteps, tb_log_name=f"{self.algorithm}",
                    callback=callbacks)
        return model

    def _deinit_environment(self, env):
        env.close()

    def _deinit_wandb(self):
        wandb.finish()

    def train(self) -> None:
        # Init
        run = self._init_wandb()
        environment = self._init_environment()
        callbacks = self._init_callbacks()

        #
        model = self._init_model(environment, callbacks)  # noqa
        model.save(self.model_path.as_posix())

        # Wandb: Log artifacts
        # Log dataset
        artifact = wandb.Artifact("dataset", type="dataset")
        artifact.add_file(self.program.experiment_dir.datasets.joinpath(self.program.args.dataset_name).as_posix())
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

    program = Program(experiment_dir=ExperimentDir(Path(__file__).parent.parent))
    load_dotenv(dotenv_path=program.project_dir.root.as_posix())
    dataset = StockFaDailyDataset(program, DOW_30_TICKER)
    dataset.load_dataset()
    t = Train(program=program, dataset=dataset)

    if program.args.sweep:
        sweep_id = wandb.sweep({})
        wandb.agent(
            sweep_id,
            function=t.train,
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            count=5
        )
    else:
        t.train()


if __name__ == '__main__':
    main()
