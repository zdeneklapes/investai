# -*- coding: utf-8 -*-
# TODO: wandb_test , How to test wandb_train
# TODO: Add to action +1 more action from 30 actions increase to 31 actions, because Agent can als decide for cash
# TODO: next datasets
# TODO: Put into dataset change of price form one index to another index: e.g. 10->15=0.5, 10->5=-0.5
from copy import deepcopy  # noqa

import wandb
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback

from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.portfolio_allocation.test.wandbtest import WandbTest
from run.shared.callback.wandbcallbackextendmemory import WandbCallbackExtendMemory
# from run.shared.callback.tensorboardcallback import TensorboardCallback
from run.shared.tickers import DOW_30_TICKER
from run.shared.algorithmsb3 import ALGORITHM_SB3
from run.shared.hyperparameters.sweep_configuration import sweep_configuration
from run.shared.environmentinitializer import EnvironmentInitializer
from shared.program import Program


class WandbTrain:
    def __init__(self, program: Program, dataset: StockFaDailyDataset, algorithm: str):
        self.dataset: StockFaDailyDataset = dataset
        self.program: Program = program
        self.algorithm: str = algorithm
        self.model_path = self.program.project_structure.models.joinpath(f"{self.algorithm}.zip")

    def _init_hyper_parameters(self) -> dict:
        """Hyperparameters for algorithm"""
        config = {}

        # Because __code__.co_varnames returns all variables inside function and I need only function parameters
        algorithm_init_parameter_count: int = ALGORITHM_SB3[self.algorithm].__init__.__code__.co_argcount
        algorithm_init_parameters = set(
            ALGORITHM_SB3[self.algorithm].__init__.__code__.co_varnames[:algorithm_init_parameter_count]
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

    def _init_wandb(self):
        self.program.log.info("Init wandb")
        wandb.init(
            # Environment variables
            project=self.program.args.wandb_project,
            dir=self.program.args.wandb_dir,
            group=self.program.args.wandb_group,
            job_type=self.program.args.wandb_job_type,
            mode=self.program.args.wandb_mode,
            tags=self.program.args.wandb_tags,
            # Other
            config=(None if self.program.args.wandb_sweep else self._init_hyper_parameters()),
            notes=f"Portfolio allocation with {self.algorithm} algorithm.",
            allow_val_change=False,
            resume=None,
            force=True,  # True: User must be logged in to W&B, False: User can be logged in or not
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    def _init_callbacks(self):
        self.program.log.info("Init callbacks")
        callbacks = CallbackList([ProgressBarCallback(), ])

        if self.program.is_wandb_enabled():
            callbacks.callbacks.append(WandbCallbackExtendMemory(
                verbose=self.program.args.wandb_verbose,
                model_save_path=self.model_path.parent.as_posix() if self.program.args.wandb_model_save else None,
                model_save_freq=self.program.args.wandb_model_save_freq if self.program.args.wandb_model_save else 0,
                gradient_save_freq=self.program.args.wandb_gradient_save_freq
            ))

        return callbacks

    def _deinit_environment(self, env):
        self.program.log.info("Deinit environment")
        env.close()

    def _init_model(self, environment, callbacks):
        model = ALGORITHM_SB3[self.algorithm](
            env=environment,
            **self._init_hyper_parameters(),
        )
        model.learn(
            total_timesteps=self.program.args.total_timesteps,
            tb_log_name=f"{self.algorithm}",
            callback=callbacks
        )
        return model

    def log_artifact(self, name: str, _type: str, path: str):
        self.program.log.info(f"Log artifact {name=}, {_type=}, {path=}")
        # Log dataset
        artifact = wandb.Artifact(name, type=_type)
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def _deinit_wandb(self):
        self.program.log.info("Deinit wandb")
        wandb.finish()

    def train_run(self):
        self.program.log.info(f"START Training {self.algorithm} algorithm.")
        # Initialize
        environment = EnvironmentInitializer(self.program, self.dataset) \
            .initialize_portfolio_allocation(self.dataset.train_dataset)
        callbacks = self._init_callbacks()

        # Model training
        model = self._init_model(environment, callbacks)
        model.save(self.model_path.as_posix())

        # Wandb: Log artifacts
        if self.program.is_wandb_enabled():
            self.log_artifact("dataset", "dataset", self.program.args.dataset_path)
            self.log_artifact("model", "model", self.model_path.as_posix())

        if self.program.args.test:
            WandbTest(program=self.program, dataset=self.dataset).test(model=model)

        # Deinit
        self._deinit_environment(environment)
        self.program.log.info(f"END Training {self.algorithm} algorithm.")

    def train(self):
        if self.program.is_wandb_enabled():
            self._init_wandb()
            self.train_run()
            self._deinit_wandb()
        else:
            self.train_run()


def main():
    from dotenv import load_dotenv

    program = Program()
    load_dotenv(dotenv_path=program.project_structure.root.joinpath(".env").as_posix())

    for algorithm in program.args.algorithms:
        if program.args.train:
            dataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
            dataset.load_dataset(program.args.dataset_path)
            wandb_train = WandbTrain(program=program, dataset=dataset, algorithm=algorithm)
            if program.args.wandb_sweep:
                sweep_id = wandb.sweep(sweep=sweep_configuration, project=program.args.wandb_project)
                wandb.agent(sweep_id, function=wandb_train.train, project=program.args.wandb_project,
                            count=program.args.wandb_sweep_count)
            else:
                wandb_train.train()


if __name__ == '__main__':
    main()
