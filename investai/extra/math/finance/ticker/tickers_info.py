# -*- coding: utf-8 -*-
# ######################################################################################################################
# Imports
# ######################################################################################################################
import concurrent.futures
import copy
import dataclasses
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import fundamentalanalysis as fa
import matplotlib

##
import pandas as pd
import tqdm
from dotenv import load_dotenv

##


##
sys.path.append("./ai_investing/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

##
from shared.projectstructure import ProjectStructure
from shared.ticker import Ticker  # Previously CompanyInfo # noqa
from shared.utils import now_time

# ######################################################################################################################
# Global Variables
# ######################################################################################################################
tickers_type = List[Dict[str, pd.DataFrame]]

problem_tickers = []


# ######################################################################################################################
# Functions
# ######################################################################################################################


@dataclasses.dataclass
class Program:
    prj_dir: ProjectStructure
    api_key: Optional[str] = None
    DEBUG: bool = False


def get_dir_path(program: Program) -> Path:
    return program.prj_dir.data.test_tickers if program.DEBUG else program.prj_dir.data.tickers


def config(program: Program):
    ##
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO: zipline problem
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ##
    matplotlib.use("Agg")

    ##
    load_dotenv(program.prj_dir.root.joinpath("env/.env"))
    try:
        os.getenv("FINANCIAL_MODELING_PREP_API")
    except Exception as e:
        raise ValueError(f"ERROR: API keys: {e}") from e


def download(symbols: list, api_key: str, period: str = "quarter") -> tickers_type:
    all_symbols: tickers_type = []
    pbar = tqdm.tqdm(list(symbols))
    for symbol in pbar:
        pbar.set_description("Symbol: %s" % symbol)
        data = {"symbol": symbol}
        problem = False

        key_cb = {
            Ticker.Names.profile.value: fa.profile,
            Ticker.Names.quotes.value: fa.quote,
            Ticker.Names.enterprise_value.value: fa.enterprise,
            Ticker.Names.data_detailed.value: fa.stock_data_detailed,
            Ticker.Names.dividends.value: fa.stock_dividend,
            Ticker.Names.ratings.value: fa.rating,
        }
        key_cb_period = {
            Ticker.Names.balance_sheet.value: fa.balance_sheet_statement,
            Ticker.Names.income.value: fa.income_statement,
            Ticker.Names.cash_flow.value: fa.cash_flow_statement,
            Ticker.Names.key_metrics.value: fa.key_metrics,
            Ticker.Names.financial_ratios.value: fa.financial_ratios,
            Ticker.Names.growth.value: fa.financial_statement_growth,
            Ticker.Names.dcf.value: fa.discounted_cash_flow,
        }

        # TODO: Make it multi-threaded od multi-process
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     result = {k: executor.submit(cb, symbol, api_key) for k, cb in key_cb.items()}
        #     for k in concurrent.futures.as_completed(result):
        #         try:
        #             raw_data[k] = result[k].result()
        #         except Exception:
        #             problem = True
        #
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     result = {k: executor.submit(cb, symbol, api_key, period=period) for k, cb in key_cb_period.items()}
        #     for k in concurrent.futures.as_completed(result):
        #         try:
        #             raw_data[k] = result[k].result()
        #         except Exception:
        #             problem = True

        # Without period argument
        for k, cb in key_cb.items():
            try:
                data[k] = cb(symbol, api_key)
            except Exception:
                problem = True

        # With period argument
        for k, cb in key_cb_period.items():
            try:
                data[k] = cb(symbol, api_key, period=period)
            except Exception:
                problem = True

        if problem:
            problem_tickers.append(symbol)
            print(f"EXCEPTION raised: {symbol}")

        all_symbols.append(data)

    return all_symbols


def download_subprocess(companies_lists: List[List], program: Program) -> tickers_type:
    all_symbols: tickers_type = []

    ##
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = [executor.submit(download, symbols, program.api_key) for symbols in companies_lists]
        for i in concurrent.futures.as_completed(result):
            try:
                all_symbols.extend(i.result())
            except Exception as e:
                print(e)

    ##
    return all_symbols


@dataclasses.dataclass
class SaveData:
    key: str
    data: pd.DataFrame
    filepath: Path

    def _concat_rows(self, old_df1: pd.DataFrame, new_df2: pd.DataFrame) -> pd.DataFrame:
        binary = pd.concat([old_df1, new_df2]).index.duplicated(keep="first")
        return pd.concat([old_df1, new_df2])[~binary]

    def _concat_columns(self, old_df1: pd.DataFrame, new_df2: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([old_df1, new_df2], axis=1)
        df.rename(columns={col: str(col) for col in df.columns})
        binary = df.columns.duplicated(keep="first")
        new_df = df.loc[:, binary]
        return new_df

    def _update_data(self) -> pd.DataFrame:
        old_data = pd.read_csv(self.filepath.as_posix(), index_col=0)

        try:
            df_row_concatenated = self._concat_rows(old_data, self.data)
            new_data = self._concat_columns(df_row_concatenated, self.data)

            if self.key in ["data_detailed", "dividends", "ratings"]:
                new_data = new_data.sort_index(axis=0, ascending=False)
            else:
                new_data = new_data.sort_index(axis=1, ascending=False)

            return new_data
        except Exception as e:
            raise ValueError(f"ERROR: {e} ({self.filepath})") from e

    def update_symbol_data(self):
        # Create new raw_data
        new_data = self._update_data()

        # Save new raw_data
        new_path = copy.deepcopy(self.filepath)

        # Handle old path
        old_path = self.filepath.parent.joinpath("old", self.filepath.with_suffix(f".{now_time()}.csv").name)
        old_path.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.rename(old_path.as_posix())

        # Handle new path
        new_data.to_csv(new_path.as_posix())

    def save_symbol_data(self):
        if self.key in ["data_detailed"]:
            self.data.to_csv(self.filepath.as_posix())
        else:
            self.data.to_csv(self.filepath.as_posix())


def save_downloaded_data(symbols: tickers_type, program: Program):
    directory = get_dir_path(program)

    for symbol_data in symbols:
        for k, v in symbol_data.items():
            if k == "symbol":
                if not directory.joinpath(v).exists():
                    directory.joinpath(v).mkdir(parents=True, exist_ok=True)
            else:
                filepath = directory.joinpath(symbol_data["symbol"], k + ".csv")
                save_data = SaveData(k, v, filepath)
                if not filepath.exists():
                    save_data.save_symbol_data()
                else:
                    save_data.update_symbol_data()


def remove_already_downloaded_tickers(tickers: List[List[str]], program: Program) -> List[List[str]]:
    directory = get_dir_path(program)
    return [ticker for ticker in tickers if not directory.joinpath(ticker).exists()]


# ######################################################################################################################
# Main
# ######################################################################################################################
if __name__ == "__main__":
    program = Program(
        prj_dir=ProjectStructure(__file__),
        DEBUG=False,
    )

    if program.prj_dir.root.name != "ai-investing":
        raise Exception(f"Wrong project directory {program.prj_dir.root}")

    config(program)
    program.api_key = os.getenv("FINANCIAL_MODELING_PREP_API")

    ##
    if not program.DEBUG:
        tickers = fa.available_companies(program.api_key).index.tolist()
        tickers = remove_already_downloaded_tickers(tickers, program)
        stop = tickers.__len__()
        step = 40
        processes = 4
    else:
        tickers = ["AAPL", "MSFT", "TSLA", "AMZN"]
        stop = 4
        step = 4
        processes = 4

    print(f"Total tickers: {stop}")
    ##
    for i in range(0, stop, step):
        _start = i
        _stop = i + step
        _step = step // processes

        ##
        bunches = [(i, i + _step) for i in range(_start, _stop, _step)]
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print(current_time, bunches)
        tickers_bunches = [tickers[b[0] : b[1]] for b in bunches]
        # print(tickers_bunches)
        # continue

        ##
        symbols: tickers_type = download_subprocess(tickers_bunches, program)
        save_downloaded_data(symbols, program)

    print(f"Problem tickers: {problem_tickers}")
