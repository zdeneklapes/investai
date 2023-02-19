# -*- coding: utf-8 -*-

# ##############################################################################
# # Stock trading with fundamentals
# ##############################################################################

# ##############################################################################
# * This notebook is based on the tutorial:
# https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530


# ##############################################################################
# TODO
# ##############################################################################
# TODO: Add argument parser


# ##############################################################################
# INCLUDES
# ##############################################################################

##
import os
import sys
import warnings

from pyfolio import timeseries

import matplotlib

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import (
    TRAINED_MODEL_DIR,
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.plot import backtest_stats, convert_daily_return_to_pyfolio_ts, get_baseline

sys.path.append("./src/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from common.Args import Args, argument_parser

from rl.data.DataTechnicalAnalysis import DataTechnicalAnalysis, DataBase
from rl.envs.StockPortfolioAllocationEnv import StockPortfolioAllocationEnv
from common.utils import now_time
from configuration.dirs import ProjectDir

##
_TRAIN_DATA_START = "2010-01-01"
_TRAIN_DATA_END = "2021-12-31"
_TEST_DATA_START = "2021-01-01"
_TEST_DATA_END = "2021-12-31"


##
# ##############################################################################
# FUNCTIONS
# ##############################################################################
def config():
    ##
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO: zipline problem
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO, WARN are ignored and only ERROR messages will be printed

    ##
    matplotlib.use("Agg")

    ##
    ProjectDir().check_and_create_dirs()


# def run_stable_baseline(args: Args):
#     if args.train:
#         ##
#         data = DataFundamentalAnalysis(_TRAIN_DATA_START, _TRAIN_DATA_END, ticker_list=DOW_30_TICKER)
#         processed_data = data.retrieve_data(args)
#         train_data = data_split(processed_data, _TRAIN_DATA_START, _TRAIN_DATA_END)
#         trade_data = data_split(processed_data, _TEST_DATA_START, TEST_END_DATE)
#         ##
#         agent = Agent(train_data, trade_data)
#         agent.train()
#         agent.save_trained_model()
#         if args.test:
#             agent.test()


# class EnvHyperParams(Enum):
#
class Pipeline:
    def __init__(self, args: Args):
        self.args = args
        self.trained_model = None

    def run_framework(self):
        if self.args.stable_baseline:
            self.run_sb3()

    def run_sb3(self):
        if self.args.train:
            self.train()

        if self.args.test:
            self.test()

    def get_env_kwargs(self, data) -> dict:
        stock_dimension = len(data.tic.unique())
        state_space = stock_dimension
        tech_indicator_list = ["macd", "rsi_30", "cci_30", "dx_30"]
        env_kwargs = {
            "df": data,
            "hmax": 100,
            "initial_amount": 1000000,
            "transaction_cost_pct": 0,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": tech_indicator_list,
            "action_space": stock_dimension,
            "reward_scaling": 1e-1,
        }
        return env_kwargs

    def get_agent_kwargs(self) -> dict:
        A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
        return A2C_PARAMS

    def train(self):
        ##
        # Data, Hyperparams
        data = DataTechnicalAnalysis(_TRAIN_DATA_START, _TEST_DATA_END, ticker_list=DOW_30_TICKER)
        data.load_data(args=self.args)
        train_data = data.get_data_type(
            df=data.dataframe, data_type=DataBase.DataType.TRAIN, date_divider=_TEST_DATA_START
        )
        env_kwargs = self.get_env_kwargs(data=train_data).copy()

        ##
        # Prepare Env, Model agent and policy
        gym_train_env = StockPortfolioAllocationEnv(**env_kwargs)
        env_train, _ = gym_train_env.get_sb_env()
        agent = DRLAgent(env=env_train)
        model_a2c = agent.get_model(model_name="a2c", model_kwargs=self.get_agent_kwargs())

        ##
        # Train
        trained = agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=5000)
        self.trained_model = trained

        ##
        # Save
        self.trained_model.save(os.path.join(TRAINED_MODEL_DIR, "a2c_{}".format(now_time())))

    def test(self):
        ##
        # Data, Hyperparams
        test_data = self.get_data(DataBase.DataType.TEST)  # TODO: fixme # pylint: disable=no-member
        env_kwargs = self.get_env_kwargs(data=test_data).copy()

        ##
        # Get trained model results
        gym_test_env = StockPortfolioAllocationEnv(**env_kwargs)
        test_daily_return, _ = DRLAgent.DRL_prediction(model=self.trained_model, environment=gym_test_env)
        # TODO: save results and actions
        DRL_strat = convert_daily_return_to_pyfolio_ts(test_daily_return)
        perf_stats_all = timeseries.perf_stats(
            returns=DRL_strat, factor_returns=DRL_strat, positions=None, transactions=None, turnover_denom="AGB"
        )
        print(f"==============STATS: Portfolio of ticker: {test_data.tic.unique()}===========")
        print(perf_stats_all)

        ##
        # Compare with the market
        index_compare = "^DJI"
        baseline_df = get_baseline(ticker=index_compare, start=_TEST_DATA_START, end=_TEST_DATA_END)
        stats = backtest_stats(baseline_df, value_col_name="close")
        print(f"==============STATS {index_compare}===========")
        print(stats)


##
# ##############################################################################
# # Main
# ##############################################################################
if __name__ == "__main__" and "__file__" in globals():
    config()
    _args: Args = argument_parser()
    pipeline = Pipeline(_args)
    pipeline.run_framework()
