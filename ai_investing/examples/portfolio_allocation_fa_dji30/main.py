# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path
from typing import Literal

import plotly.graph_objs as go
import pandas as pd
import numpy as np
from finta import TA
from tvDatafeed import TvDatafeed, Interval
from meta.config_tickers import DOW_30_TICKER
from tqdm import tqdm
from agents.stablebaselines3_models import TensorboardCallback
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback

from utils.project import reload_module, now_time, get_argparse  # noqa # pylint: disable=unused-import
from project_configs.experiment_dir import ExperimentDir
from project_configs.project_dir import ProjectDir
from project_configs.program import Program
from data.train.company_info import CompanyInfo
from examples.portfolio_allocation_fa_dji30.PortfolioAllocationEnv import PortfolioAllocationEnv
from model_config.algorithm import Algorithm, SB3_MODEL_TYPE
from model_config.callbacks import CustomCheckpointCallback
from model_config.plot import get_baseline, get_daily_return, convert_daily_return_to_pyfolio_ts
from extra.math.finance.minimum_variance import minimum_variance


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

    @property
    def train_dataset(self) -> pd.DataFrame:
        """Split dataset into train and test"""
        print("Train dataset from", self.dataset["date"].min(), "to", self.get_split_date())
        return self.dataset[self.dataset["date"] < self.get_split_date()]

    @property
    def test_dataset(self) -> pd.DataFrame:
        """Split dataset into train and test"""
        df = self.dataset[self.dataset["date"] >= self.get_split_date()]
        df.index = df["date"].factorize()[0]
        return df

    def get_split_date(self) -> str:
        """Return split date"""
        coef = 0.8
        dates = self.dataset["date"].unique()
        return dates[int(len(dates) * coef)]

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
        self.algorithm: str = algorithm_name

    def train(self) -> None:
        # Output folder
        self.program.experiment_dir.set_algo(
            f"{self.algorithm}_{self.program.experiment_dir.get_last_algorithm_index(self.algorithm) + 1}"
        )
        self.program.experiment_dir.create_specific_dirs()

        # Environment
        env = PortfolioAllocationEnv(df=self.stock_dataset.train_dataset,
                                     initial_portfolio_value=100_000,
                                     tickers=self.stock_dataset.tickers,
                                     features=self.stock_dataset.get_features(),
                                     save_path=self.program.experiment_dir.algo,
                                     start_from_index=0)
        env_train, _ = env.get_stable_baseline3_environment()

        # Callbacks
        callbacks = CallbackList([
            TensorboardCallback(),
            ProgressBarCallback(),
            CustomCheckpointCallback(memory_name="train_memory.json",
                                     save_freq=1_00,
                                     save_path=self.program.experiment_dir.algo.as_posix(),
                                     name_prefix="model",
                                     save_replay_buffer=True,
                                     save_vecnormalize=True),
        ])

        # Get model with appropriate parameters based on algorithm and framework
        algorithm = Algorithm(
            program=self.program,
            algorithm=self.algorithm,
            env=env_train
        )
        model_sb3 = algorithm.get_model(
            tensorboard_log=self.program.experiment_dir.tensorboard.as_posix(),
            verbose=0,
        )

        # Train and using CheckpointCallback save model
        model_sb3.learn(total_timesteps=1_000, tb_log_name=f"{self.algorithm}", callback=callbacks)


class Test:
    def __init__(self, program: Program, stock_dataset: StockDataset):
        self.program: Program = program
        self.stock_dataset: StockDataset = stock_dataset
        self.env: PortfolioAllocationEnv = PortfolioAllocationEnv(
            df=self.stock_dataset.test_dataset,
            initial_portfolio_value=100_000,
            tickers=self.stock_dataset.tickers,
            features=self.stock_dataset.get_features(),
            save_path=self.program.experiment_dir.algo,
            start_from_index=self.stock_dataset.test_dataset.index[0]
        )

    def test(self, model_path: Path = None) -> None:
        if model_path is None:  # Test all
            for model_path in self.program.experiment_dir.models.iterdir():
                self.program.experiment_dir.set_algo(model_path.name)
                self.test_model(model_path)
        else:  # Test one
            self.test_model(model_path)

    def test_model(self, model_path: Path) -> None:
        # Get files
        model_files = [
            m for m in model_path.iterdir()
            if m.is_file() and m.name.startswith('model') and m.suffix == ".zip"
        ]
        model_files.sort(key=lambda x: int(x.name.split("_")[1]))  # Sort by number of steps

        # Prepare model
        algorithm_name = model_path.name.split("_")[0]
        algorithm = Algorithm(
            program=self.program,
            algorithm=algorithm_name,
            env=self.env
        )
        model = algorithm.get_model(tensorboard_log=self.program.experiment_dir.tensorboard.as_posix(), verbose=0)

        # All checkpoints
        pbar = tqdm(model_files)
        for checkpoint_model_path in pbar:
            pbar.set_description(f"Testing {algorithm_name} model: {checkpoint_model_path.name}")
            self.get_checkpoints_performance(model, checkpoint_model_path)

    def get_checkpoints_performance(self, model: SB3_MODEL_TYPE, checkpoint_model_path: Path) -> None:
        loaded_model = model.load(checkpoint_model_path.as_posix())
        obs = self.env.reset()
        done = False
        while not done:
            action, _states = loaded_model.predict(obs)
            done = self.env.step(action)[2]

        # Save memory
        memory_name = f"model_{checkpoint_model_path.name.split('_')[1]}_steps_memory.json"
        self.env._memory.save(self.program.experiment_dir.algo.joinpath(memory_name))

    def get_memory_stats(self, memory_file: Path) -> None:
        pass

    def plot_stats(self) -> None:
        import pyfolio
        for p in self.program.experiment_dir.models.iterdir():
            memories = [m for m in p.iterdir() if m.name.endswith("memory.json")]
            memories.sort(key=lambda x: int(x.name.split("_")[1]))
            # TODO: Take only best memory

            for memory in memories:
                df = pd.read_json(memory)
                # Get Baseline
                baseline_df = get_baseline(ticker='^DJI', start=df.loc[0, 'date'], end=df.tail(1).iloc[0]['date'])
                baseline_returns = get_daily_return(baseline_df, value_col_name="close")

                # Get DRL
                df_ts = convert_daily_return_to_pyfolio_ts(df, "portfolio_return")

                with pyfolio.plotting.plotting_context(font_scale=1.1):
                    pyfolio.create_full_tear_sheet(returns=df_ts, benchmark_rets=baseline_returns, set_context=False)

                # print("\n\n\n\n")
                # print(memory.name)
                break

    def plot_compare_portfolios(self):
        for p in self.program.experiment_dir.models.iterdir():
            memories = [m for m in p.iterdir() if m.name.endswith("memory.json")]
            memories.sort(key=lambda x: int(x.name.split("_")[1]))
            # TODO: Take only best memory
            for memory in memories:
                #
                df = pd.read_json(memory)

                #
                baseline_df = get_baseline(ticker='^DJI', start=df.loc[0, 'date'], end=df.tail(1).iloc[0]['date'])
                baseline_returns = get_daily_return(baseline_df, value_col_name="close")

                #
                model_cumpod = (1 + df['portfolio_return']).cumprod() - 1
                dji_cumpod = (1 + baseline_returns).cumprod() - 1
                min_var_cumpod = minimum_variance(self.stock_dataset.test_dataset)

                #
                trace0_portfolio = go.Scatter(x=df['date'], y=model_cumpod, mode='lines',
                                              name=f"{p.name}]")
                trace1_portfolio = go.Scatter(x=df['date'], y=dji_cumpod, mode='lines', name='DJIA')
                trace2_portfolio = go.Scatter(x=df['date'], y=min_var_cumpod, mode='lines', name='Min-Variance')

                #
                fig = go.Figure()
                fig.add_trace(trace0_portfolio)
                fig.add_trace(trace1_portfolio)
                fig.add_trace(trace2_portfolio)

                #
                fig.update_layout(
                    legend=dict(
                        x=0,
                        y=1,
                        traceorder="normal",
                        font=dict(
                            family="sans-serif",
                            size=15,
                            color="black"
                        ),
                        bgcolor="White",
                        bordercolor="white",
                        borderwidth=2

                    ),
                )
                # fig.update_layout(legend_orientation="h")
                fig.update_layout(title={
                    # 'text': "Cumulative Return using FinRL",
                    'y': 0.85,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
                # with Transaction cost
                # fig.update_layout(title =  'Quarterly Trade Date')
                fig.update_layout(
                    #    margin=dict(l=20, r=20, t=20, b=20),

                    paper_bgcolor='rgba(1,1,0,0)',
                    plot_bgcolor='rgba(1, 1, 0, 0)',
                    # xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    xaxis={'type': 'date',
                           'tick0': df.loc[0, 'date'],
                           'tickmode': 'linear',
                           'dtick': 86400000.0 * 80}

                )
                fig.update_xaxes(showline=True, linecolor='black', showgrid=True, gridwidth=1,
                                 gridcolor='LightSteelBlue', mirror=True)
                fig.update_yaxes(showline=True, linecolor='black', showgrid=True, gridwidth=1,
                                 gridcolor='LightSteelBlue', mirror=True)
                fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

                fig.show()
                break


def initialisation(arg_parse: bool = True) -> Program:
    prj_dir = ProjectDir(root=Path("/Users/zlapik/my-drive-zlapik/0-todo/ai-investing"))
    experiment_dir = ExperimentDir(Path(__file__).parent)
    experiment_dir.create_dirs()
    return Program(
        project_dir=prj_dir,
        experiment_dir=experiment_dir,
        args=get_argparse()[1] if arg_parse else None,
    )


def t1():
    print("t1")


def stock_dataset(program: Program) -> StockDataset:
    stock_dataset_init = StockDataset(program)

    #
    if program.args is None:
        stock_dataset_init.load_dataset()
        return stock_dataset_init

    if program.args.prepare_dataset:  # Dataset is not provided create it
        stock_dataset_init.preprocess()
        stock_dataset_init.save()
        return stock_dataset_init
    else:
        stock_dataset_init.load_dataset()
        return stock_dataset_init


def train(program: Program, dataset: StockDataset):
    t = Train(stock_dataset=dataset, program=program, algorithm_name="ppo")
    t.train()


def test(program: Program, dataset: StockDataset):
    test = Test(program=program, stock_dataset=dataset)
    # test.test()
    # test.plot_stats()
    test.plot_compare_portfolios()


def main():
    program = initialisation()
    dataset = stock_dataset(program)

    if program.debug is None:
        if program.args.train:
            train(program, dataset)
        if program.args.test:
            test(program, dataset)


if __name__ == "__main__":
    main()
