# -*- coding: utf-8 -*-
# ######################################################################################################################
# Imports
# ######################################################################################################################
import os
import sys
import copy
import warnings
import concurrent
from typing import Dict, List, Optional
import dataclasses

##
import matplotlib
from pathlib import Path

##
sys.path.append("./src/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

##

##
from configuration.settings import ProjectDir


@dataclasses.dataclass
class Program:
    prj_dir: ProjectDir
    api_key: Optional[str] = None
    DEBUG: bool = False


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


##
import pandas as pd
import tqdm
from dotenv import load_dotenv
import fundamentalanalysis as fa

##

# ######################################################################################################################
# Global Variables
# ######################################################################################################################
tickers_type = List[Dict[str, pd.DataFrame]]

problem_tickers = []


def download(symbols: list, api_key: str, period: str = "quarter") -> tickers_type:
    all_symbols: tickers_type = []
    pbar = tqdm.tqdm(list(symbols))
    for symbol in pbar:
        pbar.set_description("Symbol: %s" % symbol)
        try:
            data = {
                "symbol": symbol,
                "balance_sheet": fa.balance_sheet_statement(symbol, api_key, period=period),
                "income": fa.income_statement(symbol, api_key, period=period),
                "cash_flow": fa.cash_flow_statement(symbol, api_key, period=period),
                "key_metrics": fa.key_metrics(symbol, api_key, period=period),
                "financial_ratios": fa.financial_ratios(symbol, api_key, period=period),
                "growth": fa.financial_statement_growth(symbol, api_key, period=period),
                "data_detailed": fa.stock_data_detailed(symbol, api_key),
            }
        except Exception as e:
            print(f"ERROR: {symbol} - {e}")
            problem_tickers.append(symbol)
            continue
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


def update_downloaded_data(key: str, data: pd.DataFrame, filepath: Path):
    old_data = pd.read_csv(filepath.as_posix(), index_col=0)

    # Create new data
    if key in ["data_detailed"]:
        binary = data["date"] != old_data["date"]
        new_data = old_data.append(data[binary])
    else:
        for i in data.columns.difference(old_data.columns).values:
            old_data.insert(0, i, data[i])
        new_data = old_data.sort_index(axis=1, ascending=False)

    # Rename previous filename and Save new data
    new_path = copy.deepcopy(filepath)
    filepath.rename(filepath.with_suffix(".csv.old"))
    new_data.to_csv(new_path.as_posix())


def new_downloaded_data(key: str, data: pd.DataFrame, filepath: Path):
    if key in ["data_detailed"]:
        data.to_csv(filepath.as_posix())
    else:
        data.to_csv(filepath.as_posix())


def save_downloaded_data(symbols: tickers_type, program: Program):
    for symbol_data in symbols:
        for k, v in symbol_data.items():
            if k == "symbol":
                if not program.prj_dir.dataset.tickers.joinpath(v).exists():
                    program.prj_dir.dataset.tickers.joinpath(v).mkdir(parents=True, exist_ok=True)
            else:
                filepath = program.prj_dir.dataset.tickers.joinpath(symbol_data["symbol"], k + ".csv")
                if not filepath.exists():
                    new_downloaded_data(k, v, filepath)
                else:
                    update_downloaded_data(k, v, filepath)


# ######################################################################################################################
# Main
# ######################################################################################################################
if __name__ == "__main__":
    program = Program(prj_dir=ProjectDir(root=Path(__file__).parent.parent.parent.parent.parent), DEBUG=True)
    config(program)
    program.api_key = os.getenv("FINANCIAL_MODELING_PREP_API")

    ##
    if not program.DEBUG:
        tickers = fa.available_companies(program.api_key).index.tolist()
        stop = tickers.__len__()
        step = 200
        processes = 4
    else:
        tickers = ["AAPL", "MSFT", "TSLA", "FB"]
        stop = 4
        step = 4
        processes = 4

    ##
    for i in range(0, stop, step):
        _start = i
        _stop = i + step
        _step = step // processes

        ##
        bunches = [(i, i + _step) for i in range(_start, _stop, _step)]
        print(bunches)
        tickers_bunches = [tickers[b[0] : b[1]] for b in bunches]
        # print(tickers_bunches)
        # continue

        ##
        symbols: tickers_type = download_subprocess(tickers_bunches, program)
        save_downloaded_data(symbols, program)

    print(f"Problem tickers: {problem_tickers}")
