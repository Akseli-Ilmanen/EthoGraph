import numpy as np
import pytest
import xarray as xr


class TestTrialTree:

    def test_open_returns_trial_tree(self, test_nc_path):
        from ethograph import TrialTree
        dt = TrialTree.open(test_nc_path)
        assert isinstance(dt, TrialTree)

    def test_trials_property_returns_list(self, trial_tree):
        trials = trial_tree.trials
        assert isinstance(trials, list)
        assert len(trials) > 0

    def test_itrial_returns_dataset(self, trial_tree):
        ds = trial_tree.itrial(0)
        assert isinstance(ds, xr.Dataset)

    def test_trial_by_id_returns_dataset(self, trial_tree):
        trial_id = trial_tree.trials[0]
        ds = trial_tree.trial(trial_id)
        assert isinstance(ds, xr.Dataset)

    def test_itrial_out_of_range_raises(self, trial_tree):
        with pytest.raises(IndexError):
            trial_tree.itrial(9999)

    def test_get_label_dt(self, trial_tree):
        label_dt = trial_tree.get_label_dt()
        first_trial = label_dt.trials[0]
        assert "labels" in label_dt.trial(first_trial).data_vars

    def test_get_label_dt_empty(self, trial_tree):
        label_dt = trial_tree.get_label_dt(empty=True)
        first_trial = label_dt.trials[0]
        labels = label_dt.trial(first_trial).labels.values
        assert np.all(labels == 0)

    def test_get_all_trials(self, trial_tree):
        all_trials = trial_tree.get_all_trials()
        assert isinstance(all_trials, dict)
        assert len(all_trials) == len(trial_tree.trials)

    def test_from_datasets_roundtrip(self, first_trial_ds):
        from ethograph import TrialTree
        dt = TrialTree.from_datasets([first_trial_ds])
        assert len(dt.trials) == 1


class TestValidation:

    def test_validate_datatree_returns_list(self, trial_tree):
        from ethograph.utils.validation import validate_datatree
        errors = validate_datatree(trial_tree)
        assert isinstance(errors, list)

    def test_extract_type_vars_has_features(self, type_vars_dict):
        assert "features" in type_vars_dict
        assert len(type_vars_dict["features"]) > 0

    def test_extract_type_vars_has_individuals(self, type_vars_dict):
        assert "individuals" in type_vars_dict

    def test_extract_type_vars_has_cameras(self, type_vars_dict):
        assert "cameras" in type_vars_dict

    def test_extract_type_vars_has_trial_conditions(self, type_vars_dict):
        assert "trial_conditions" in type_vars_dict

    def test_find_temporal_dims(self, first_trial_ds):
        from ethograph.utils.validation import find_temporal_dims
        dims = find_temporal_dims(first_trial_ds)
        assert isinstance(dims, set)
        assert "time" not in dims

    def test_validate_required_attrs(self, first_trial_ds):
        from ethograph.utils.validation import validate_required_attrs
        errors = validate_required_attrs(first_trial_ds)
        assert isinstance(errors, list)

    def test_validate_dataset_with_type_vars(self, first_trial_ds, type_vars_dict):
        from ethograph.utils.validation import validate_dataset
        errors = validate_dataset(first_trial_ds, type_vars_dict)
        assert isinstance(errors, list)


class TestDataUtils:

    def test_get_time_coord_labels(self, first_trial_ds):
        from ethograph.utils.data_utils import get_time_coord
        time = get_time_coord(first_trial_ds.labels)
        assert time is not None
        assert len(time) > 0

    def test_get_time_coord_feature(self, first_trial_ds, type_vars_dict):
        from ethograph.utils.data_utils import get_time_coord
        feature_name = type_vars_dict["features"][0]
        time = get_time_coord(first_trial_ds[feature_name])
        assert time is not None
        assert len(time) > 0

    def test_sel_valid_filters_individual(self, first_trial_ds, type_vars_dict):
        from ethograph.utils.data_utils import sel_valid
        individual = str(type_vars_dict["individuals"][0])
        kwargs = {"individuals": individual}
        data, filt = sel_valid(first_trial_ds.labels, kwargs)
        assert data.ndim == 1
        assert "individuals" in filt

    def test_sel_valid_ignores_nonexistent_dim(self, first_trial_ds):
        from ethograph.utils.data_utils import sel_valid
        kwargs = {"nonexistent_dim": "value"}
        data, filt = sel_valid(first_trial_ds.labels, kwargs)
        assert len(filt) == 0


class TestFirstTrialDataset:

    def test_has_labels_variable(self, first_trial_ds):
        assert "labels" in first_trial_ds.data_vars

    def test_has_fps_attribute(self, first_trial_ds):
        assert "fps" in first_trial_ds.attrs
        assert first_trial_ds.attrs["fps"] > 0

    def test_has_trial_attribute(self, first_trial_ds):
        assert "trial" in first_trial_ds.attrs

    def test_has_cameras_attribute(self, first_trial_ds):
        assert "cameras" in first_trial_ds.attrs

    def test_has_individuals_coord(self, first_trial_ds):
        assert "individuals" in first_trial_ds.coords

    def test_labels_are_integer(self, first_trial_ds):
        from ethograph.utils.validation import is_integer_array
        labels = first_trial_ds.labels.values
        assert is_integer_array(labels)

    def test_has_feature_variables(self, first_trial_ds, type_vars_dict):
        for feat in type_vars_dict["features"]:
            assert feat in first_trial_ds.data_vars
