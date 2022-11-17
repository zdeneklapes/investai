# -*- coding: utf-8 -*-
import pandas as pd
from typing import Dict


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
    def get_different_timesteps(samples: int, step: int):
        for i in range(step, samples * step, step):
            yield i


def get_env_kwargs(df: pd.DataFrame) -> Dict[str, int]:
    ratio_list = [
        "OPM",
        "NPM",
        "ROA",
        "ROE",
        "cur_ratio",
        "quick_ratio",
        "cash_ratio",
        "inv_turnover",
        "acc_rec_turnover",
        "acc_pay_turnover",
        "debt_ratio",
        "debt_to_equity",
        "PE",
        "PB",
        "Div_yield",
    ]

    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(ratio_list) * stock_dimension  # TODO: Why?
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    return {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": ratio_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        # "num_stock_shares": 0,
    }
