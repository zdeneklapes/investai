# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from configuration.dirs import ExperimentDir


@pytest.mark.parametrize("a", [pytest.param(2), ], )
def test_experiment_create_dirs(a):
    experiment_dir = ExperimentDir(root=Path(__file__).parent)
    experiment_dir.create_dirs()
    experiment_dir.create_specific_dirs("algo", str(1))
    experiment_dir.delete_out_dir()
