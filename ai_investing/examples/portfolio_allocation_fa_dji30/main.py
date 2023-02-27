# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np
from finta import TA
from tvDatafeed import TvDatafeed, Interval
from meta.config_tickers import DOW_30_TICKER
from tqdm import tqdm

from utils.project import get_argparse
from project_configs.experiment_dir import ExperimentDir
from project_configs.project_dir import ProjectDir
from project_configs.program import Program
from data.train.company_info import CompanyInfo
from agent.custom_drl_agent import CustomDRLAgent
from examples.portfolio_allocation_fa_dji30.PortfolioAllocationEnv import PortfolioAllocationEnv

from utils.project import reload_module  # noqa # pylint: disable=unused-import


class StockDataset:
    def __init__(self, program: Program):
        TICKERS = deepcopy(DOW_30_TICKER)
        TICKERS.remove("DOW")  # TODO: "DOW" is not DJI30 or what?
        #
        self.program: Program = program
        self.tickers = TICKERS
        self.unique_columns = ["date", "tic"]
        self.base_columns = ["open", "high", "low", "close", "volume"]
        # Technical analysis indicators
        self.ta_indicators = []
        # Fundamental analysis indicators
        self.fa_indicators = [
            "operatingProfitMargin",
            "netProfitMargin",
            "returnOnAssets",
            "returnOnEquity",
            "currentRatio",
            "quickRatio",
            "cashRatio",
            "inventoryTurnover",
            "receivablesTurnover",
            "payablesTurnover",
            "debtRatio",
            "debtEquityRatio",
            "priceEarningsRatio",
            "priceBookValueRatio",
            "dividendYield",
        ]

        # Final dataset for training and testing
        self.dataset = None

    def data_split(self):
        """Split dataset into train and test"""

    def get_features(self):
        """Return features for training and testing"""
        return self.fa_indicators + self.ta_indicators + self.base_columns

    def preprocess(self) -> pd.DataFrame:
        """Return dataset"""
        self.dataset = self.get_stock_dataset()
        return self.dataset

    def get_stock_dataset(self) -> pd.DataFrame:
        df = pd.DataFrame()
        pbar = tqdm(self.tickers)
        for tic in pbar:
            pbar.set_description(f"Processing {tic}")

            # Load tickers data
            raw_data: CompanyInfo = self.load_raw_data(tic)

            # Add features
            feature_data = self.add_fa_features(raw_data)

            # Add ticker to dataset
            df = pd.concat([feature_data, df])

        df.insert(0, "date", df.index)
        df = self.clean_dataset_from_missing_stock_in_some_days(df)
        df = self.make_index_by_date(df)
        df = df.sort_values(by=self.unique_columns)
        assert not self.is_dataset_correct(df), "Dataset is not correct"
        return df

    def is_dataset_correct(self, df: pd.DataFrame) -> bool:
        """Check if all data are correct"""
        return df.isna().any().any()  # Can't be any Nan/np.inf values

    def add_fa_features(self, ticker_raw_data: CompanyInfo) -> pd.DataFrame:
        """
        Add fundamental analysis features to dataset
        Merge tickers information into one pd.Dataframe
        """
        # Prices
        prices = ticker_raw_data.data_detailed[self.base_columns]
        prices.insert(0, "tic", ticker_raw_data.symbol)

        # Fill before or forward
        prices = prices.fillna(method="bfill")
        prices = prices.fillna(method="ffill")

        # Ratios
        ratios = ticker_raw_data.financial_ratios.loc[self.fa_indicators].transpose()

        # Fill 0, where Nan/np.inf
        ratios = ratios.fillna(0)
        ratios = ratios.replace(np.inf, 0)

        merge = pd.merge(prices, ratios,
                         how="outer", left_index=True, right_index=True)
        filled = merge.fillna(method="bfill")
        filled = filled.fillna(method="ffill")
        clean = filled.drop(filled[~filled.index.str.contains(r"\d{4}-\d{2}-\d{2}")].index)  # Remove non-dates
        return clean

        # df = pd.concat([clean, df])

        # df.insert(0, "date", df.index)
        # assert not df.isna().any().any()  # Can't be any Nan/np.inf values
        # return df

    def clean_dataset_from_missing_stock_in_some_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Take only those dates where we have data for all stock in each day
        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        df: pd.DataFrame
        """
        ticker_in_each_date = df.groupby("date").size()
        where_are_all_tickers = ticker_in_each_date.values == ticker_in_each_date.values.max()

        # FIXME: This is not correct, because we can have missing data in the middle of the dataset
        earliest_date = ticker_in_each_date[where_are_all_tickers].index[0]
        df = df[df["date"] > earliest_date]
        return df

    def make_index_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create index for same dates"""
        df.sort_values(by="date", inplace=True)
        df.index = df["date"].factorize()[0]
        assert df.groupby("date").size().unique().size == 1, "Why is it here?"
        return df

    def add_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features to dataset"""
        # Add features
        df["macd"] = TA.MACD(df).SIGNAL
        df["boll_ub"] = TA.BBANDS(df).BB_UPPER
        df["boll_lb"] = TA.BBANDS(df).BB_LOWER
        df["rsi_30"] = TA.RSI(df, period=30)
        df["dx_30"] = TA.ADX(df, period=30)
        # df["close_30_sma"] = TA.SMA(df, period=30) # Unnecessary correlated with boll_lb
        # df["close_60_sma"] = TA.SMA(df, period=60) # Unnecessary correlated with close_30_sma
        df = df.fillna(0)  # Nan will be replaced with 0
        return df

    def add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick features to dataset"""
        # get candlestick features
        candle_size = abs(df["high"] - df["low"])  # noqa
        body_size = abs(df["close"] - df["open"])  # noqa

        # get upper and lower shadow
        upper_shadow = np.where((df["close"] > df["open"]), df["high"] - df["close"], df["high"] - df["open"])
        lower_shadow = np.where((df["close"] > df["open"]), df["open"] - df["low"], df["close"] - df["low"])

        df["candle_size"] = candle_size
        df["body_size"] = body_size / candle_size  # pertentage of body size in candle size
        df["candle_upper_shadow"] = upper_shadow / candle_size  # pertentage of upper shadow in candle size
        df["candle_lower_shadow"] = lower_shadow / candle_size  # pertentage of lower shadow in candle size
        df["candle_direction"] = np.where((df["close"] - df["open"]) > 0, 1, 0)  # 1 - up, 0 - down
        return df

    def feature_correlation_drop(self, df: pd.DataFrame, threshold: float = 0.6,
                                 method: Literal['pearson', 'kendall', 'spearman'] = 'pearson') -> pd.DataFrame:
        """Drop features with correlation higher than threshold"""
        corr = df.corr(method=method)
        triu = pd.DataFrame(np.triu(corr.T).T, corr.columns, corr.columns)
        threshhold = triu[(triu > threshold) & (triu < 1)]
        return threshhold

    # TODO: Create function for downloading data using yfinance and another one for TradingView
    def download(self, ticker,
                 exchange: str,
                 interval: Interval | str,
                 period: str = '',  # FIXME: not used
                 n_bars: int = 10) -> pd.DataFrame:
        """Return raw data"""
        # df = yf.download(tickers=tickers, period=period, interval=interval) # BUG: yfinance bad candlestick data

        tv = TvDatafeed()
        df = tv.get_hist(symbol=ticker, exchange=exchange, interval=interval, n_bars=n_bars)
        df = df.fillna(0)  # Nan will be replaced with 0

        return df

    def save(self) -> None:
        """Save dataset"""
        file_name = self.program.experiment_dir.datasets.joinpath(self.program.args.dataset)
        print(f"Saving dataset to: {file_name}")
        self.dataset.to_csv(file_name, index=True)

    def load_raw_data(self, tic) -> CompanyInfo:
        """Check if folders with ticker exists and load all data from them into CompanyInfo class"""
        data = {"symbol": tic}
        files = deepcopy(CompanyInfo.Names.list())
        files.remove("symbol")
        for f in files:
            tic_file = self.program.project_dir.data.tickers.joinpath(tic).joinpath(f + ".csv")
            if tic_file.exists():
                data[f] = pd.read_csv(tic_file, index_col=0)
            else:
                raise FileExistsError(f"File not exists: {tic_file}")
        return CompanyInfo(**data)

    def load_dataset(self) -> None:
        """Load dataset"""
        file_name = self.program.experiment_dir.datasets.joinpath(self.program.args.dataset)
        print(f"Loading dataset from: {file_name}")
        self.dataset = pd.read_csv(file_name, index_col=0)


class Train:
    def __init__(self, stock_dataset: StockDataset, program: Program, algorithm_name: str = "ppo"):
        self.stock_dataset: StockDataset = stock_dataset
        self.program: Program = program
        self.model = None
        self.env: PortfolioAllocationEnv | None = None
        self.algorithm = algorithm_name

    def train(self) -> None:
        self.env = PortfolioAllocationEnv(df=self.stock_dataset.dataset,
                                          initial_portfolio_value=100_000,
                                          tickers=self.stock_dataset.tickers,
                                          features=self.stock_dataset.get_features(),
                                          save_path=self.program.experiment_dir.try_number,
                                          start_from_index=0)
        env_train, _ = self.env.get_stable_baseline3_environment()
        drl_agent = CustomDRLAgent(env=env_train, program=self.program, algorithm=self.algorithm)

        ALGORITHM_PARAMS = {  # noqa: F841 # pylint: disable=unused-variable
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        }
        # Parameter for algorithm
        algorithm = drl_agent.get_model(
            model_name=self.algorithm,
            model_kwargs=ALGORITHM_PARAMS,
            tensorboard_log=self.program.experiment_dir.tensorboard.as_posix(),
            verbose=0,
            device="cpu",
        )

        # Train
        self.model = drl_agent.train_model(
            model=algorithm, tb_log_name=f"tb_run_{self.algorithm}", checkpoint_freq=10_000,
            total_timesteps=200_000
        )

    def save_model(self) -> None:
        pass

    def load_model(self) -> None:
        pass


class Test:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.results = None

    def test(self) -> None:
        pass

    def save_results(self) -> None:
        pass

    def load_results(self) -> None:
        pass

    def plot_results(self) -> None:
        pass


def initialisation(arg_parse: bool = True) -> Program:
    prj_dir = ProjectDir(root=Path("/Users/zlapik/my-drive-zlapik/0-todo/ai-investing"))
    experiment_dir = ExperimentDir(Path(__file__).parent)
    experiment_dir.create_dirs()
    return Program(
        project_dir=prj_dir,
        experiment_dir=experiment_dir,
        args=get_argparse()[1] if arg_parse else None,
        train_date_start='2019-01-01',
        train_date_end='2020-01-01',
        test_date_start='2020-01-01',
        test_date_end='2021-01-01',
    )


def t1():
    program = initialisation(False)
    d = StockDataset(program)
    df = d.get_stock_dataset()
    return {
        "d": d,
        "df": df,
    }


def main():
    program_init = initialisation()

    if program_init.debug is None:
        if program_init.args.prepare_dataset:  # Dataset is not provided create it
            stock_dataset_init = StockDataset(program_init)
            stock_dataset_init.preprocess()
            stock_dataset_init.save()
        else:
            stock_dataset_init = StockDataset(program_init)
            stock_dataset_init.load_dataset()
        if program_init.args.train:
            #
            train = Train(stock_dataset=stock_dataset_init, program=program_init, algorithm_name="ppo")

            #
            program_init.experiment_dir.add_attributes_for_models(train.algorithm)
            program_init.experiment_dir.create_specific_dirs()

            #
            train.train()
        if program_init.args.test:
            test = Test()
            test.test()


if __name__ == "__main__":
    main()
