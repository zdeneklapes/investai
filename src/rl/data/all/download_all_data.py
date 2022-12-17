# -*- coding: utf-8 -*-
##
import os
import sys
import warnings

##
import matplotlib

##
sys.path.append("./src/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

##
from finrl import config as finrl_config
from finrl.main import check_and_make_directories
from dotenv import load_dotenv
import fundamentalanalysis as fa

##
from common.utils import now_time
from configuration.settings import ProjectDir
from rl.data.financial_modeling_prep import download
import concurrent


def config():
    ##
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO: zipline problem
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ##
    matplotlib.use("Agg")

    ##
    check_and_make_directories(
        [
            finrl_config.DATA_SAVE_DIR,
            finrl_config.TRAINED_MODEL_DIR,
            finrl_config.TENSORBOARD_LOG_DIR,
            finrl_config.RESULTS_DIR,
        ]
    )

    ##
    load_dotenv(prj_dir.root.joinpath("env/.env"))
    try:
        print(os.getenv("EOD_HISTORICAL_DATA_API"))
        print(os.getenv("NASDAQ_HISTORICAL_DATA_API"))
        print(os.getenv("ALPHA_VANTAGE_API"))
        print(os.getenv("FINANCIAL_MODELING_PREP_API"))
    except Exception as e:
        raise ValueError("ERROR: API keys") from e


def get_download_range(_from: int, _plus_costant: int) -> tuple:
    return (_from, _from + _plus_costant - 1)


def download_subprocess(start: int, stop: int, step: int) -> dict:
    all_symbols_info = dict({})  # noqa: C408
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = [
            executor.submit(
                download, companies.index[get_download_range(i, step)[0] : get_download_range(i, step)[1]], api_key
            )
            for i in range(start, stop, step)
        ]

        for i in concurrent.futures.as_completed(result):
            try:
                all_symbols_info.update(i.result())
            except Exception as e:
                print(e)
    return all_symbols_info


def download_start():
    for i in range(start, stop, step):
        ##
        _start = i
        _stop = i + step - 1
        _step = step // processes

        # get_download_range()
        print([get_download_range(i, _step) for i in range(_start, _stop, _step)])

        if DEBUG:
            continue

        ##
        filename = prj_dir.dataset.financial_modeling_prep.joinpath(f"{_start}_{_stop}_companies_{now_time()}.pkl")
        comps = download_subprocess(_start, _stop, _step)
        with open(file=filename, mode="wb") as f:
            import pickle  # nosec

            pickle.dump(comps, f)


if __name__ == "__main__" and __file__ == sys.argv[0]:
    ##
    config()

    ##
    # Global Variables
    prj_dir = ProjectDir()
    api_key = os.getenv("FINANCIAL_MODELING_PREP_API")
    companies = fa.available_companies(api_key)

    ##
    # Configurations for download multiprocessing
    start = 6000
    stop = 10000  # companies.index.size
    step = 400
    processes = 4
    DEBUG = False

    ##
    download_start()
