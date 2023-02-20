# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from configuration.dirs import ExperimentDir


class TestExperimentDir:
    def teardown_class(self):
        """Delete out directory after all tests are done."""
        experiment_dir = ExperimentDir(root=Path(__file__).parent)
        experiment_dir.delete_out_dir()

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

    @pytest.mark.depends(on=["test_create_dirs", "test_create_specific_dirs", "test_add_attributes_for_models"])
    @pytest.mark.parametrize("_id, add_attr", [
        pytest.param("1", False),
        pytest.param("2", True),
    ])
    def test__get_next_algo_folder_id_1(self, _id, add_attr):
        """Check when self.algo is None, then return 1."""
        experiment_dir = ExperimentDir(root=Path(__file__).parent)

        if add_attr:
            experiment_dir.add_attributes_for_models("algo")

        folder_id = experiment_dir._get_next_algo_folder_id()
        assert folder_id == _id

    @pytest.mark.depends(on=["test_create_dirs"])
    def test_delete_out_dir(self):
        experiment_dir = ExperimentDir(root=Path(__file__).parent)
        experiment_dir.delete_out_dir()
        assert not experiment_dir.out.exists()
