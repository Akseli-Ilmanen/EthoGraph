import numpy as np
import pytest
import xarray as xr


class TestTrialTree:

    def test_open_and_access_trials(self, test_nc_path):
        from ethograph import TrialTree
        dt = TrialTree.open(test_nc_path)
        assert isinstance(dt, TrialTree)

        trials = dt.trials
        assert isinstance(trials, list)
        assert len(trials) > 0

        ds = dt.itrial(0)
        assert isinstance(ds, xr.Dataset)

        ds_by_id = dt.trial(trials[0])
        assert isinstance(ds_by_id, xr.Dataset)

    def test_itrial_out_of_range_raises(self, trial_tree):
        with pytest.raises(IndexError):
            trial_tree.itrial(99999)

    def test_label_dt(self, trial_tree):
        label_dt = trial_tree.get_label_dt()
        first_trial = label_dt.trials[0]
        trial_ds = label_dt.trial(first_trial)
        assert "onset_s" in trial_ds.data_vars
        assert "offset_s" in trial_ds.data_vars
        assert "labels" in trial_ds.data_vars
        assert "individual" in trial_ds.data_vars

        label_dt_empty = trial_tree.get_label_dt(empty=True)
        empty_ds = label_dt_empty.trial(first_trial)
        assert empty_ds.sizes.get("segment", 0) == 0

    def test_from_datasets_roundtrip(self, first_trial_ds):
        from ethograph import TrialTree
        dt = TrialTree.from_datasets([first_trial_ds])
        assert len(dt.trials) == 1


class TestValidation:

    def test_validate_datatree(self, trial_tree):
        from ethograph.utils.validation import validate_datatree
        errors = validate_datatree(trial_tree)
        assert isinstance(errors, list)

    def test_extract_type_vars(self, type_vars_dict):
        assert "features" in type_vars_dict
        assert len(type_vars_dict["features"]) > 0
        assert "individuals" in type_vars_dict
        assert "cameras" in type_vars_dict
        assert "trial_conditions" in type_vars_dict

    def test_find_temporal_dims_and_validate(self, first_trial_ds, type_vars_dict):
        from ethograph.utils.validation import (
            find_temporal_dims, validate_required_attrs, validate_dataset,
        )
        dims = find_temporal_dims(first_trial_ds)
        assert isinstance(dims, set)
        assert "time" not in dims

        assert isinstance(validate_required_attrs(first_trial_ds), list)
        assert isinstance(validate_dataset(first_trial_ds, type_vars_dict), list)


class TestDataUtils:

    def test_get_time_coord(self, first_trial_ds, type_vars_dict):
        from ethograph.utils.data_utils import get_time_coord

        time_labels = get_time_coord(first_trial_ds.labels)
        assert time_labels is not None
        assert len(time_labels) > 0

        feature_name = type_vars_dict["features"][0]
        time_feat = get_time_coord(first_trial_ds[feature_name])
        assert time_feat is not None
        assert len(time_feat) > 0

    def test_sel_valid(self, first_trial_ds, type_vars_dict):
        from ethograph.utils.data_utils import sel_valid

        individual = str(type_vars_dict["individuals"][0])
        data, filt = sel_valid(first_trial_ds.labels, {"individuals": individual})
        assert data.ndim == 1
        assert "individuals" in filt

        data, filt = sel_valid(first_trial_ds.labels, {"nonexistent_dim": "value"})
        assert len(filt) == 0


class TestFirstTrialDataset:

    def test_dataset_schema(self, first_trial_ds, type_vars_dict):
        from ethograph.utils.validation import is_integer_array

        assert "labels" in first_trial_ds.data_vars
        assert "fps" in first_trial_ds.attrs
        assert first_trial_ds.attrs["fps"] > 0
        assert "trial" in first_trial_ds.attrs
        assert "cameras" in first_trial_ds.attrs
        assert "individuals" in first_trial_ds.coords
        assert is_integer_array(first_trial_ds.labels.values)

        for feat in type_vars_dict["features"]:
            assert feat in first_trial_ds.data_vars
