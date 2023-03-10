# -*- coding: utf-8 -*-
from pathlib import Path

from examples.portfolio_allocation_fa_dji30.dataset import StockDataset
from examples.portfolio_allocation_fa_dji30.test import Test
from examples.portfolio_allocation_fa_dji30.train import Train
from utils.project import reload_module, now_time, get_argparse  # noqa # pylint: disable=unused-import
from project_configs.experiment_dir import ExperimentDir
from project_configs.program import Program


def initialisation(arg_parse: bool = True) -> Program:
    prj_dir = ProjectDir(__file__)
    experiment_dir = ExperimentDir(Path(__file__).parent)
    experiment_dir.create_dirs()
    return Program(
        project_dir=prj_dir,
        experiment_dir=experiment_dir,
        args=get_argparse()[1] if arg_parse else None,
    )


def dataset(program: Program) -> StockDataset:
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
    t = Train(stock_dataset=dataset, program=program)
    t.train()


def test(program: Program, dataset: StockDataset):
    test = Test(program=program, stock_dataset=dataset)
    # test.test()
    # test.plot_stats()
    test.plot_compare_portfolios()


def t1():
    program = initialisation()
    _dataset = dataset(program)

    if program.debug:
        import wandb
        wandb.plot.line_chart()
        # return Train(stock_dataset=_dataset, program=program), wandb
    else:
        if program.args.train:
            train(program, _dataset)
        if program.args.test:
            test(program, _dataset)


if __name__ == '__main__':
    from project_configs.project_dir import ProjectDir
    from dotenv import load_dotenv

    project_dir = ProjectDir(__file__)
    load_dotenv(dotenv_path=project_dir.root.joinpath(".env"))
    t1()
