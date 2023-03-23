# -*- coding: utf-8 -*-
"""
Stock fundamental analysis dataset
"""

import pandas as pd
from IPython.display import display  # noqa
from shared.program import Program

# For Debugging
from shared.utils import reload_module  # noqa
from run.shared.memory import Memory


class StockCombinedDailyDataset(Memory):
    def __init__(self, program: Program):
        self.program = program

    def preprocess(self):
        dfs = []
        for dataset_path in self.program.args.dataset_paths:
            dfs.append(pd.read_csv(dataset_path, index_col=0, parse_dates=True))

        cols_to_use = dfs[1].columns.difference(dfs[0].columns).insert(0, ['date', 'tic'])
        df = pd.merge(dfs[0], dfs[1][cols_to_use], on=['date', 'tic'], how='outer')
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df.index = df.date.factorize()[0]
        self.df = df
        return self.df


def main():
    program = Program()
    dataset = StockCombinedDailyDataset(program)
    dataset.preprocess()
    dataset.save_csv(program.args.folder_dataset.joinpath('stockcombineddailydataset.csv').as_posix())


if __name__ == "__main__":
    main()
