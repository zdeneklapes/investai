# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from configuration.dirs import ExperimentDir


class TestExperimentDir:
    def test_create_dirs(self):
        experiment_dir = ExperimentDir(root=Path(__file__).parent)
        experiment_dir.create_dirs()
        assert experiment_dir.root.exists()

    @pytest.mark.depends(on=["test_create_dirs"])
    def test_add_attributes_for_models(self):
        experiment_dir = ExperimentDir(root=Path(__file__).parent)
        experiment_dir.create_dirs()
        experiment_dir.add_attributes_for_models("algo")
        assert experiment_dir.algo is not None
        assert experiment_dir.tensorboard is not None
        assert experiment_dir.results is not None

    @pytest.mark.depends(on=["test_create_dirs", "test_add_attributes_for_models"])
    def test_create_specific_dirs(self):
        experiment_dir = ExperimentDir(root=Path(__file__).parent)
        experiment_dir.create_dirs()
        experiment_dir.add_attributes_for_models("algo")
        experiment_dir.create_specific_dirs()
        assert experiment_dir.algo.exists()
        assert experiment_dir.tensorboard.exists()
        assert experiment_dir.results.exists()

    @pytest.mark.depends(on=["test_create_dirs"])
    def test_delete_out_dir(self):
        experiment_dir = ExperimentDir(root=Path(__file__).parent)
        experiment_dir.delete_out_dir()
        assert not experiment_dir.out.exists()
