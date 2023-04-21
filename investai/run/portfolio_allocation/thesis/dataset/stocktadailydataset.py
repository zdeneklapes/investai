# -*- coding: utf-8 -*-
"""
Stock Technical analysis dataset
"""
import inspect
from functools import partial
from copy import deepcopy
from typing import Dict, List, TypedDict
from pprint import pprint  # noqa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa
from extra.math.finance.ticker.ticker import Ticker
from IPython.display import display  # noqa
from run.shared.dataset.candlestickengineer import CandlestickEngineer as CSE
from run.shared.dataset.dataengineer import DataEngineer as DE
from run.shared.tickers import DOW_30_TICKER
from shared.program import Program
from finta import TA

# For Debugging
from shared.reload import reload_module  # noqa
from tqdm import tqdm
from tvDatafeed import Interval, TvDatafeed
from run.shared.memory import Memory


class StockTaDailyDataset(Memory):
    def __init__(self, program: Program, tickers: List[str], dataset_split_coef: float):
        TICKERS = deepcopy(tickers)
        TICKERS.remove("DOW")  # TODO: Fixme: "DOW" is not in DJI30 or what?
        #
        self.program: Program = program
        self.tickers = TICKERS
        self.unique_columns = ["date", "tic"]
        self.base_columns = ["open", "high", "low", "close", "volume", "changePercent"]
        # Technical analysis indicators
        self.ta_indicators = {
            "macd": TA.MACD,
            "boll_ub": TA.BBANDS,
            "boll_lb": TA.BBANDS,
            "rsi_30": partial(TA.RSI, period=30),
            "adx_30": partial(TA.ADX, period=30),
            # "close_30_sma": TA.SMA,
            # "close_60_sma": TA.SMA,
        }
        # Fundamental analysis indicators
        self.fa_indicators = []
        self.TA_functions: Dict[str, callable] = self._get_TA_functions()

        # Final dataset for training and testing
        self.dataset_split_coef = dataset_split_coef
        super().__init__(program, pd.DataFrame())

    def _get_TA_functions(self) -> Dict[str, callable]:
        """Return list of technical analysis functions"""
        TA_items = dict(TA.__dict__)

        # These are uncorrelated indicators
        wanted_ta_indicators = set(['SMA', 'ER', 'MACD', 'PPO', 'EV_MACD', 'VBM', 'ADX', 'UO',
                                    'MI', 'BOP', 'CHAIKIN', 'BASP', 'BASPN', 'QSTICK', 'SQZMI', 'FVE', 'VFI',
                                    'WILLIAMS_FRACTAL'])

        TA_wanted_indicators = {name: func for name, func in TA_items.items() if name in wanted_ta_indicators}
        return TA_wanted_indicators

    @property
    def train_dataset(self) -> pd.DataFrame:
        """Split dataset into train and test"""
        if self.program.args.project_verbose > 0:
            self.program.log.info(
                f"Train dataset from {self.df['date'].iloc[0]} to {DE.get_split_date(self.df, self.dataset_split_coef)}"
                # noqa
            )
        df: pd.DataFrame = self.df[self.df["date"] < DE.get_split_date(self.df, self.dataset_split_coef)]
        return df

    @property
    def test_dataset(self) -> pd.DataFrame:
        """Split dataset into train and test"""
        df: pd.DataFrame = self.df[
            self.df["date"] >= DE.get_split_date(self.df, self.dataset_split_coef)
        ]
        df.index = df["date"].factorize()[0]
        return df

    def get_features(self) -> List[str]:
        """Return features for training and testing"""
        return self.fa_indicators + list(self.ta_indicators.keys()) + self.base_columns

    def preprocess(self) -> pd.DataFrame:
        """Return dataset"""
        self.df = self.get_stock_dataset()
        return self.df

    def get_stock_dataset(self) -> pd.DataFrame:
        df = pd.DataFrame()

        iterable = tqdm(self.tickers, leave=False) if self.program.args.project_verbose > 0 else self.tickers
        for tic in iterable:
            if isinstance(iterable, tqdm): iterable.set_description(f"Processing {tic}")
            # Load tickers raw_data
            raw_data: Ticker = self.load_raw_data(tic)  # Load tickers raw_data

            # Technical analysis features
            price_data = raw_data.data_detailed[self.base_columns]
            ta_feature_data = self.get_uncorrelated_ta_features(price_data)  # Add features

            # Ticker with features
            tic_with_features = pd.concat([price_data, ta_feature_data], axis=1)
            tic_with_features.insert(0, "tic", tic)

            # Merge ticker with dataset
            df = pd.concat([tic_with_features, df], axis=0)

        df.insert(0, "date", df.index)
        df = df.sort_values(by=["tic"])
        df = DE.clean_dataset_from_missing_tickers_by_date(df)
        df = df.sort_values(by=self.unique_columns)
        df.index = df["date"].factorize()[0]
        DE.check_dataset_correctness_assert(df)

        return df

    def add_fa_features(self, ticker_raw_data: Ticker) -> pd.DataFrame:
        """
        Add fundamental analysis features to dataset
        Merge tickers information into one pd.Dataframe
        """
        # Prices
        prices = ticker_raw_data.data_detailed[self.base_columns]
        prices.insert(0, "tic", ticker_raw_data.ticker)

        # Fill before or forward
        prices = prices.fillna(method="bfill")
        prices = prices.fillna(method="ffill")

        # Ratios
        ratios = ticker_raw_data.financial_ratios.loc[self.fa_indicators].transpose()

        # Fill 0, where Nan/np.inf
        ratios = ratios.fillna(0)
        ratios = ratios.replace(np.inf, 0)

        merge = pd.merge(prices, ratios, how="outer", left_index=True, right_index=True)
        filled = merge.fillna(method="bfill")
        filled = filled.fillna(method="ffill")
        clean = filled.drop(filled[~filled.index.str.contains(r"\d{4}-\d{2}-\d{2}")].index)  # Remove non-dates
        return clean

    def add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick features to dataset"""
        candle_size = CSE.candlestick_size(df)
        body_size = CSE.body_size(df)
        upper_shadow = CSE.candlestick_up_shadow(df)
        down_shadow = CSE.candlestick_down_shadow(df)

        # Add features
        df["shadow_size_pct"] = (upper_shadow + down_shadow) / candle_size
        df["body_size_pct"] = body_size / candle_size
        df["candle_up_shadow_pct"] = upper_shadow / candle_size
        df["candle_down_shadow_pct"] = down_shadow / candle_size
        df["candle_direction"] = CSE.candlestick_direction(df)
        return df

    def get_uncorrelated_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get uncorrelated technical analysis features"""
        df = self.get_ta_features(df)
        # df = self.drop_correlated_features(df)
        df.fillna(0, inplace=True)
        return df

    def drop_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop correlated features"""
        correlation_threshold = 0.5
        correlation_upper_matrix = self.get_correlation_matrix(df)
        to_drop = [column
                   for column in correlation_upper_matrix.columns
                   if any(correlation_upper_matrix[column] > correlation_threshold)]
        df_not_correlated = df.drop(df[to_drop], axis=1)
        return df_not_correlated

    def get_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get correlation matrix"""
        # TODO: Check this!
        corr_matrix = df.corr()  # .abs()
        ones_map = np.ones(corr_matrix.shape)
        upper_ones_map = np.triu(ones_map, k=0).astype(np.bool)
        correlation_upper_matrix = corr_matrix.where(upper_ones_map)
        return correlation_upper_matrix

    def get_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get uncorrelated technical analysis features"""
        df_ta = pd.DataFrame()
        if self.program.args.project_figure:
            iterable = tqdm(dict(TA.__dict__).items())
            # iterable = tqdm(dict(VORTEX=TA.VORTEX).items())
        else:
            iterable = (tqdm(self.TA_functions.items(), leave=False) if self.program.args.project_verbose > 0
                        else self.TA_functions.items())

        for name, func in iterable:
            if isinstance(iterable, tqdm): iterable.set_description(f"Calculating indicator: {name}")
            if callable(func):  # TODO: remove this if statement
                none_optional_params = {name: param
                                        for name, param in inspect.signature(func).parameters.items()
                                        if param.default is inspect.Parameter.empty}
                try:
                    if "period" in none_optional_params:
                        ta_indicator = func(df, period=30)
                    else:
                        ta_indicator = func(df)
                except (NotImplementedError, TypeError):
                    self.program.log.info(f"Skipping {name} indicator")
                    continue

                if isinstance(ta_indicator, pd.Series):
                    df_ta[name] = ta_indicator
                elif isinstance(ta_indicator, pd.DataFrame):
                    if "SIGNAL" in ta_indicator.columns:
                        df_ta[f"{name}_SIGNAL"] = ta_indicator["SIGNAL"]
                    else:
                        for col in ta_indicator.columns:
                            df_ta[f"{name}_{col}"] = ta_indicator[col]
        return df_ta

    # TODO: Create function for downloading raw_data using yfinance and another one for TradingView
    def download(
        self, ticker, exchange: str, interval: Interval | str, period: str = "", n_bars: int = 10  # FIXME: not used
    ) -> pd.DataFrame:
        """Return raw raw_data"""
        # df = yf.download(tickers=tickers, period=period, interval=interval) # BUG: yfinance bad candlestick raw_data
        tv = TvDatafeed()
        df = tv.get_hist(symbol=ticker, exchange=exchange, interval=interval, n_bars=n_bars)
        df = df.fillna(0)  # Nan will be replaced with 0
        return df

    def load_raw_data(self, tic) -> Ticker:
        """Check if folders with ticker exists and load all raw_data from them into Ticker class"""
        data = {"ticker": tic}
        files = deepcopy(Ticker.Names.list())
        files.remove("symbol")
        for f in files:
            tic_file = self.program.args.folder_ticker.joinpath(tic).joinpath(f + ".csv")
            if tic_file.exists():
                data[f] = pd.read_csv(tic_file, index_col=0)
            else:
                raise FileExistsError(f"File not exists: {tic_file}")
        return Ticker(**data)


class TestStockTaDailyDataset:
    def __init__(self):
        self.program: Program = Program()
        self.program.args.project_verbose = 1
        self.program.args.debug = 1

    def t1(self) -> Dict:
        """
        Debugging and manual testing in ipython
        :return: dict
        """
        dataset = StockTaDailyDataset(self.program, tickers=DOW_30_TICKER,
                                      dataset_split_coef=self.program.args.dataset_split_coef)

        r = {}
        r['program'] = self.program
        r["dataset"] = dataset
        r["d"] = dataset.get_stock_dataset()
        return r


def t():
    return TestStockTaDailyDataset().t1()


def create_heatmap_of_all_indicators():
    program: Program = Program()
    program.args.project_verbose = 1
    program.args.project_figure = 1

    dataset = StockTaDailyDataset(program, tickers=DOW_30_TICKER,
                                  dataset_split_coef=program.args.dataset_split_coef)
    r_T = TypedDict("r", {"program": Program, "dataset": StockTaDailyDataset, "raw_data": Ticker, "d": pd.DataFrame,
                          "corr_m": pd.DataFrame})
    r: r_T = {}
    r['program'] = program
    r["dataset"] = dataset
    r['raw_data'] = dataset.load_raw_data("AAPL")  # Load tickers raw_data
    price_data = r['raw_data'].data_detailed[r['dataset'].base_columns]

    # all
    # r["d"] = r['dataset'].get_uncorrelated_ta_features(price_data)  # Add features
    # r['corr_m'] = r['dataset'].get_correlation_matrix(r['d'])
    # ax = sns.heatmap(r['corr_m'], cmap="coolwarm", xticklabels=False, yticklabels=False)  # type: Axes
    # plt.savefig(program.args.folder_figure.joinpath("ta_correlation_matrix_all_indicators.png"))

    # uncorrelated
    program.args.project_figure = 0
    r["d"] = r['dataset'].get_uncorrelated_ta_features(price_data)  # Add features
    r['corr_m'] = r['dataset'].get_correlation_matrix(r['d'])
    ax = sns.heatmap(r['corr_m'], cmap="coolwarm", )  # type: Axes
    indicators = pd.Series(['SMA', 'ER', 'MACD_SIGNAL', 'PPO_SIGNAL', 'EV_MACD_SIGNAL', 'VBM',
                            'ADX', 'UO', 'MI', 'BOP', 'CHAIKIN', 'BASP_B', 'BASP_S',
                            'BASPN_B', 'BASPN_S', 'QSTICK', 'SQZMI', 'FVE', 'VFI',
                            'WILLIAMS_FRACTAL_Bear', 'WILLIAMS_FRACTAL_Bull'],
                           dtype='object')
    ax.set_xticks(np.arange(indicators.__len__()), indicators)
    ax.set_yticks(np.arange(indicators.__len__()), indicators)
    plt.tight_layout()
    plt.savefig(program.args.folder_figure.joinpath("ta_correlation_matrix_uncorrelated_indicators.png"))
    return r


def create_heatmap_of_uncorrelated_indicators():
    pass


def main():
    program = Program()
    dataset = StockTaDailyDataset(program, tickers=DOW_30_TICKER, dataset_split_coef=program.args.dataset_split_coef)
    dataset.preprocess()
    dataset.save_csv(program.args.dataset_paths)


if __name__ == "__main__":
    main()
    # t()
    # create_heatmap_of_all_indicators()
