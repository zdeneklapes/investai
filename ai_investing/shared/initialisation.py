# -*- coding: utf-8 -*-
from pathlib import Path

from shared.dir.project_dir import ProjectDir
from shared.dir.experiment_dir import ExperimentDir
from shared.program import Program
from shared.arguments import parse_arguments


def initialisation(arg_parse: bool = True) -> Program:
    prj_dir = ProjectDir(__file__)
    experiment_dir = ExperimentDir(Path(__file__).parent)
    experiment_dir.create_dirs()
    return Program(
        project_dir=prj_dir,
        experiment_dir=experiment_dir,
        args=parse_arguments()[1] if arg_parse else None,
    )
