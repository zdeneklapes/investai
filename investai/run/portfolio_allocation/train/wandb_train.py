# -*- coding: utf-8 -*-
# TODO: wandb sweep
# TODO: all algorithms
# TODO: tests
# TODO: next datasets
import os
from typing import Union

from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
import wandb
from wandb.sdk.wandb_run import Run
from wandb.sdk.lib.disabled import RunDisabled
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from run.shared.tickers import DOW_30_TICKER

from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.portfolio_allocation.envs.portfolioallocationenv import PortfolioAllocationEnv
from run.shared.callbacks import WandbCallbackExtendMemory
from run.shared.callbacks import TensorboardCallback
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
        # TODO: Rewrite parameters from wandb.config when they are defined on cli
        algorithm_parameters = Train.ALGORITHMS[self.algorithm].__init__.__code__.co_varnames
        self.program.args.__dict__.keys()
        return {key: self.program.args.__dict__[key] for key in algorithm_parameters if
                key in self.program.args.__dict__}

    def _init_wandb(self) -> Union[Run, RunDisabled, None]:
        if self.program.args.sweep:
            return wandb.init()
        else:
            run = wandb.init(
                job_type="train",
                dir=self.program.project_structure.models.as_posix(),
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
            tensorboard_log=self.program.project_structure.tensorboard.as_posix(),
            env=env,
            **wandb.config,
        )
        model.learn(
            total_timesteps=self.program.args.total_timesteps,
            tb_log_name=f"{self.algorithm}",
            callback=callbacks
        )
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
    load_dotenv(dotenv_path=program.project_structure.root.as_posix())
    dataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    for algorithm in ["ppo", "a2c", "sac", "td3", "dqn", "ddpg"]:
        t = Train(program=program, dataset=dataset, algorithm=algorithm)
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
        break


if __name__ == '__main__':
    main()
