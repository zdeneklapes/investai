# -*- coding: utf-8 -*-
from agents.stablebaselines3_models import TensorboardCallback
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
import wandb

from examples.portfolio_allocation_fa_dji30.PortfolioAllocationEnv import PortfolioAllocationEnv
from examples.portfolio_allocation_fa_dji30.dataset import StockDataset
from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import model_config.algorithm_parameters as algo_params
from project_configs.program import Program
from model_config.callbacks import WandbCallbackExtendMemory


class Train:
    def __init__(self, stock_dataset: StockDataset, program: Program):
        self.stock_dataset: StockDataset = stock_dataset
        self.program: Program = program
        self.algorithm: str = "ppo"

    def _init_folder(self) -> None:
        self.program.experiment_dir.set_algo(
            f"{self.algorithm}_{self.program.experiment_dir.get_last_algorithm_index(self.algorithm) + 1}"
        )
        self.program.experiment_dir.create_specific_dirs()

    def _init_wandb(self):
        run = wandb.init(
            project="ai-investing",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            dir=self.program.experiment_dir.algo.as_posix(),
        )
        intersect_keys = (
            set(self.program.args.__dict__.keys())
            .intersection(
                set(algo_params.STABLE_BASELINE_PARAMETERS[self.algorithm].keys())
            )
        )
        default_keys = set(algo_params.STABLE_BASELINE_PARAMETERS[self.algorithm].keys()).difference(
            intersect_keys
        )

        cli_config = {key: self.program.args.__dict__[key] for key in intersect_keys}
        default_config: dict = {
            key: algo_params.STABLE_BASELINE_PARAMETERS[self.algorithm][key]
            for key in default_keys
        }
        assert len(set(cli_config.keys()).intersection(set(default_config.keys()))) == 0, \
            "cli_config keys and default_config keys should be unique"
        wandb.config.update(cli_config)
        wandb.config.update(default_config)
        return run

    def _init_environment(self):
        env = PortfolioAllocationEnv(df=self.stock_dataset.train_dataset,
                                     initial_portfolio_value=100_000,
                                     tickers=self.stock_dataset.tickers,
                                     features=self.stock_dataset.get_features(),
                                     save_path=self.program.experiment_dir.algo,
                                     start_from_index=0,
                                     wandb=True)
        env = Monitor(env, wandb.run.dir, allow_early_resets=True)
        # env = Monitor(env, wandb.run.dir)
        env = DummyVecEnv([lambda: env])
        return env

    def _init_callbacks(self):
        callbacks = CallbackList([
            TensorboardCallback(),
            ProgressBarCallback(),
            WandbCallbackExtendMemory(
                verbose=2,
                model_save_path=self.program.experiment_dir.algo.as_posix(),
                model_save_freq=1000,
                gradient_save_freq=1000,
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
        # Output folder
        self._init_folder()

        # Wandb
        wandb_run = self._init_wandb()

        # Environment
        env = self._init_environment()

        # Callbacks
        callbacks = self._init_callbacks()

        # Model
        model = self._init_model(env, callbacks)
        model.save(
            (self.program.experiment_dir.algo / f"model_{self.program.args.total_timesteps}_steps.zip").as_posix()
        )

        # Deinit
        self._deinit_environment(env)
        self._deinit_wandb()
