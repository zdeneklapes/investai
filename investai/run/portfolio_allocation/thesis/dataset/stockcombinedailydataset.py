# -*- coding: utf-8 -*-
"""
Stock fundamental analysis dataset
"""

import pandas as pd
from IPython.display import display  # noqa
from shared.program import Program

# For Debugging
from shared.reload import reload_module  # noqa
from shared.utils import log_artifact
from run.shared.memory import Memory


class StockCombinedDailyDataset(Memory):
    def __init__(self, program: Program):
        self.program = program

    def preprocess(self):
        dfs = []
        dfs.append(pd.read_csv(self.program.args.dataset_paths[0], index_col=0, parse_dates=True))
        dfs.append(pd.read_csv(self.program.args.dataset_paths[1], index_col=0, parse_dates=True))

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
    file_path = program.args.dataset_paths[2]
    dataset.save_csv(file_path.as_posix())

    # Save to wandb
    if program.is_wandb_enabled():
        log_artifact(program.args, file_path.as_posix(), file_path.name.split('.')[0], "dataset",
                     {"path": file_path.as_posix()})


if __name__ == "__main__":
    main()
