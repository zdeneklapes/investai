# -*- coding: utf-8 -*-

from common.Args import get_argparse


def dataset():
    # TODO: load data for all tickers

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
