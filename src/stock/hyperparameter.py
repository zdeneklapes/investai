# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple

import pandas as pd


class HyperParameter:
    A2C_PARAMS = ("a2c", None, 1000)
    DDPG_PARAMS = ("ddpg", None, 50_000)
    PPO_PARAMS = (
        "ppo",
        {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        },
        30_000,
    )

    TD3_PARAMS = ("sac", {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}, 30_000)

    SAC_PARAMS = (
        "sac",  # name
        {  # params
            "batch_size": 128,
            "buffer_size": 1000000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        },
        30_000,  # time_steps
    )

    @staticmethod
    def get_hyperparameter(name: str) -> Tuple[str, Dict, int]:
        return getattr(HyperParameter, f"{name.upper()}_PARAMS")

    @staticmethod
    def get_hyperparameters() -> List[Tuple[str, Dict, int]]:
        return [
            HyperParameter.get_hyperparameter(name)
            for name in HyperParameter.__dict__.keys()
            if name.endswith("_PARAMS")
        ]

    @staticmethod
    def get_different_timesteps(samples: int, step: int):
        yield from range(step, samples * step, step)

    @staticmethod
    def get_env_kwargs(df: pd.DataFrame) -> Dict[str, int]:
        ratio_list_2 = {  # noqa: F841 # pylint: disable=unused-variable
            "opm": "operating profit margin",
            "roe": "return on equity",
            "roa": "return on assets",
            "roic": "return on invested capital",
            "roce": "return on capital employed",
            "npm": "net profit margin",
            "cur_ratio": "current ratio",
            "quick_ratio": "quick ratio",
            "cash_ratio": "cash ratio",
            "inv_turnover": "inventory turnover",
            "acc_rec_turnover": "accounts receivable turnover",
            "acc_pay_turnover": "accounts payable turnover",
            "debt_ratio": "debt ratio",
            "debt_to_equity": "debt to equity ratio",
            "pe": "price to earnings ratio",
            "pb": "price to book ratio",
            "div_yield": "dividend yield",
        }

        stock_dimension = len(df.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(ratio_list_2.keys()) * stock_dimension  # TODO: Why?
        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

        # TODO: Why these hyperparams?
        return {
            "hmax": 100,
            "initial_amount": 1000000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": ratio_list_2.keys(),
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            # "num_stock_shares": 0,
        }
