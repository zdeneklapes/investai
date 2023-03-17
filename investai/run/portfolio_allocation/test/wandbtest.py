# -*- coding: utf-8 -*-
"""TODO docstring"""
from tqdm import trange

from shared.program import Program
from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.shared.algorithmsb3 import ALGORITHM_SB3_TYPE
from run.shared.tickers import DOW_30_TICKER
from run.shared.environmentinitializer import EnvironmentInitializer


class WandbTest:
    def __init__(self, program: Program, dataset: StockFaDailyDataset):
        self.program: Program = program
        self.dataset: StockFaDailyDataset = dataset

    def _deinit_environment(self, env):
        self.program.log.info("Deinit environment")
        env.close()

    def test(self, model: ALGORITHM_SB3_TYPE, deterministic=True) -> None:
        environment = EnvironmentInitializer(self.program, self.dataset) \
            .initialize_portfolio_allocation(self.dataset.test_dataset)
        obs = environment.reset()
        # -2 because we don't want to go till terminal state, because the environment will be reset
        iterable = trange(len(environment.envs[0].env._df.index.unique()) - 2, desc="Test") \
            if self.program.args.project_verbose \
            else range(len(environment.envs[0].env._df.index.unique()) - 2)
        for _ in iterable:
            action, _ = model.predict(obs, deterministic=deterministic)
            # if self.program.is_wandb_enabled():
            #     memory_dict = environment.envs[0].env._memory.df.iloc[-1].to_dict()
            #     del memory_dict['action']
            #     del memory_dict['date']
            #     log_dict = {f"memory/test_{k}": v for k, v in memory_dict.items()}
            #     wandb.log(log_dict)
            environment.step(action)
        # if self.program.is_wandb_enabled():
        #     info = {"test/total_reward": (environment.envs[0].env._memory.df['reward'] + 1.0).cumprod().iloc[-1],
        #             # TODO: Add sharpe ratio
        #             # TODO: reward annualized
        #             "test/start_date": environment.envs[0].env._df['date'].unique()[0],
        #             "test/end_date": environment.envs[0].env._df['date'].unique()[-1], }
        #     wandb_summary(info)


def get_best_model(self): pass


def main():
    from dotenv import load_dotenv

    program = Program()
    load_dotenv(dotenv_path=program.project_structure.root.as_posix())
    dataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)

    # TODO: get best model from wandb
    # get_best_model()


if __name__ == '__main__':
    main()
