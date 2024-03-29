# -*- coding: utf-8 -*-
"""Train a model with Stable Baselines3"""
import os
import time
from copy import deepcopy  # noqa
from pprint import pprint  # noqa
from pathlib import Path

import wandb

from run.portfolio_allocation.thesis.dataset.stockfadailydataset import StockFaDailyDataset
from run.portfolio_allocation.thesis.test import Test
from run.shared.callback.wandbcallbackextendmemory import WandbCallbackExtendMemory
from run.shared.environmentinitializer import EnvironmentInitializer
from run.shared.sb3.algorithms import ALGORITHM_SB3_STR2CLASS
from run.shared.sb3.sweep_configuration import sweep_configuration
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback


class Train:
    def __init__(self, program: Program, dataset_path: Path, algorithm: str):
        self.dataset: StockFaDailyDataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
        self.dataset.load_csv(file_path=dataset_path.as_posix())
        self.dataset_path: Path = dataset_path
        self.program: Program = program
        self.algorithm: str = algorithm.lower()
        self.model_path: Path = self.program.args.folder_model.joinpath(f"{self.algorithm}.zip")
        if os.path.isfile(self.model_path.as_posix()): os.remove(self.model_path.as_posix())

    def _init_hyper_parameters(self) -> dict:
        """Hyperparameters for algorithm"""
        config = {}

        # Because __code__.co_varnames returns all variables inside function and I need only function parameters
        algorithm_init_parameter_count: int = ALGORITHM_SB3_STR2CLASS[self.algorithm].__init__.__code__.co_argcount
        algorithm_init_parameters = set(
            ALGORITHM_SB3_STR2CLASS[self.algorithm].__init__.__code__.co_varnames[:algorithm_init_parameter_count]
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
            config = {
                key: self.program.args.__dict__[key]
                for key in algorithm_init_parameters
                if key in self.program.args.__dict__
            }
        if "d" in self.program.args.project_verbose: pprint(config)

        config["tensorboard_log"] = self.program.args.folder_tensorboard.as_posix()
        if not hasattr(config, "seed"): config["seed"] = int(time.time())
        return config

    def _init_wandb(self):
        self.program.log.info("Init wandb")
        wandb.init(
            # Environment variables
            entity=self.program.args.wandb_entity,
            project=self.program.args.wandb_project,
            group=self.program.args.wandb_run_group,
            job_type=self.program.args.wandb_job_type,
            mode=self.program.args.wandb_mode,
            tags=self.program.args.wandb_tags,
            dir=self.program.args.wandb_dir.as_posix(),
            # Other
            config=(None if self.program.args.wandb_sweep else self._init_hyper_parameters()),
            notes=f"Portfolio allocation with {self.algorithm} algorithm.",
            allow_val_change=False,
            resume=None,
            force=True,  # True: User must be logged in to W&B, False: User can be logged in or not
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=False,
        )

    def _init_callbacks(self):
        self.program.log.info("Init callbacks")
        callbacks = CallbackList(
            [
                ProgressBarCallback(),
            ]
        )

        if self.program.is_wandb_enabled():
            callbacks.callbacks.append(
                WandbCallbackExtendMemory(
                    verbose=self.program.args.wandb_verbose,
                    model_save_path=self.model_path.parent.as_posix() if self.program.args.wandb_model_save else None,
                    model_save_freq=self.program.args.wandb_model_save_freq
                    if self.program.args.wandb_model_save
                    else 0,
                    gradient_save_freq=self.program.args.wandb_gradient_save_freq,
                    program=self.program,
                )
            )

        return callbacks

    def _deinit_environment(self, env):
        self.program.log.info("Deinit environment")
        env.close()

    def _init_model(self, environment, callbacks):
        model = ALGORITHM_SB3_STR2CLASS[self.algorithm](
            env=environment,
            **self._init_hyper_parameters(),
        )
        model.learn(
            total_timesteps=self.program.args.total_timesteps, tb_log_name=f"{self.algorithm}", callback=callbacks
        )
        return model

    def log_artifact(self, name: str, _type: str, path: str):
        if "i" in self.program.args.project_verbose:
            self.program.log.info(f"Log artifact {name=}, {_type=}, {path=}")
        # Log dataset
        artifact = wandb.Artifact(name, type=_type, metadata={"path": path})
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def _deinit_wandb(self):
        self.program.log.info("Deinit wandb")
        wandb.finish()

    def train_run(self):
        self.program.log.info(f"START Training {self.algorithm} algorithm.")
        # Initialize
        environment = EnvironmentInitializer(self.program,
                                             self.dataset).portfolio_allocation(self.dataset.train_dataset)
        callbacks = self._init_callbacks()

        # Model training
        model = self._init_model(environment, callbacks)
        model.save(self.model_path.as_posix())

        # Wandb: Log artifacts
        if self.program.is_wandb_enabled():
            self.log_artifact(self.algorithm, "model", self.model_path.as_posix())

        if self.program.args.test:
            Test(program=self.program,
                 dataset_path=self.dataset_path).test(model_path=self.model_path.as_posix(),
                                                      algorithm=self.algorithm)

        # Deinit
        self._deinit_environment(environment)
        self.program.log.info(f"END Training {self.algorithm} algorithm.")

    def train(self):
        if self.program.is_wandb_enabled(check_init=False):
            self._init_wandb()
            self.train_run()
            self._deinit_wandb()
        else:
            self.train_run()


def main(
    program: Program = Program()
):
    dataset_path: Path
    for dataset_path in program.args.dataset_paths:
        for algorithm in program.args.algorithms:
            for i in range(program.args.train):
                program.log.info(f"START {i}. Training {algorithm} algorithm.")
                #
                wandb_train = Train(program=program, dataset_path=dataset_path, algorithm=algorithm)
                if not program.args.wandb_sweep:
                    wandb_train.train()
                else:
                    sweep_id = wandb.sweep(sweep=sweep_configuration, project=program.args.wandb_project)
                    wandb.agent(
                        sweep_id,
                        function=wandb_train.train,
                        project=program.args.wandb_project,
                        count=program.args.wandb_sweep_count,
                    )
                #
                program.log.info(f"END {i}. Training {algorithm} algorithm.")


if __name__ == "__main__":
    main()
