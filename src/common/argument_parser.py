from typing import Dict, Any
import argparse


def parse_cli_argument() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", dest="profile", required=False)
    parser.add_argument("--backtest", action="store_true", dest="profile", required=False)
    cli_args = vars(parser.parse_args())
    return cli_args
