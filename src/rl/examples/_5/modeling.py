# -*- coding: utf-8 -*-
import pandas as pd

from common.Args import get_argparse


def dataset():
    # TODO: load data for all tickers
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    df = table[0]
    df.to_csv('S&P500-Info.csv')
    df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])

    # TODO: preprocess data

    # TODO: save dataset
    pass


def train():
    # TODO: load dataset

    # TODO: train

    # TODO: save model
    pass


if __name__ == "__main__":
    args_vars, args = get_argparse()
    # print(os.getenv("PYTHON_PATH"))
    if args.dataset:
        dataset()
    if args.train:
        train()
