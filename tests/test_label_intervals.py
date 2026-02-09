"""Tests for ethograph.utils.label_intervals module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ethograph.utils.label_intervals import (
    add_interval,
    delete_interval,
    dense_to_intervals,
    empty_intervals,
    find_interval_at,
    get_interval_bounds,
    intervals_to_dense,
    intervals_to_xr,
    purge_short_intervals,
    stitch_intervals,
    xr_to_intervals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dense():
    """Dense label array: 10 samples, 1 individual, labels at idx 2-4 (id=1) and 7-9 (id=2)."""
    arr = np.array([0, 0, 1, 1, 1, 0, 0, 2, 2, 2], dtype=np.int8)
    time = np.arange(10) * 0.1  # 10 Hz
    individuals = ["bird1"]
    return arr.reshape(-1, 1), time, individuals


@pytest.fixture
def sample_intervals():
    """Matching interval DataFrame for sample_dense."""
    return pd.DataFrame(
        {
            "onset_s": [0.2, 0.7],
            "offset_s": [0.4, 0.9],
            "label_id": [1, 2],
            "individual": ["bird1", "bird1"],
        }
    )


# ---------------------------------------------------------------------------
# empty_intervals
# ---------------------------------------------------------------------------

class TestEmptyIntervals:
    def test_columns(self):
        df = empty_intervals()
        assert list(df.columns) == ["onset_s", "offset_s", "label_id", "individual"]

    def test_empty(self):
        df = empty_intervals()
        assert len(df) == 0


# ---------------------------------------------------------------------------
# dense_to_intervals
# ---------------------------------------------------------------------------

class TestDenseToIntervals:
    def test_basic(self, sample_dense):
        arr, time, inds = sample_dense
        df = dense_to_intervals(arr, time, inds)
        assert len(df) == 2
        assert df.iloc[0]["label_id"] == 1
        assert df.iloc[1]["label_id"] == 2
        np.testing.assert_almost_equal(df.iloc[0]["onset_s"], 0.2)
        np.testing.assert_almost_equal(df.iloc[0]["offset_s"], 0.4)

    def test_all_zeros(self):
        arr = np.zeros((10, 1), dtype=np.int8)
        time = np.arange(10) * 0.1
        df = dense_to_intervals(arr, time, ["ind1"])
        assert len(df) == 0

    def test_1d_input(self):
        arr = np.array([0, 1, 1, 0], dtype=np.int8)
        time = np.arange(4) * 0.5
        df = dense_to_intervals(arr, time, ["ind1"])
        assert len(df) == 1
        assert df.iloc[0]["onset_s"] == 0.5
        assert df.iloc[0]["offset_s"] == 1.0

    def test_multi_individual(self):
        arr = np.array(
            [[1, 0], [1, 2], [0, 2], [0, 0]], dtype=np.int8
        )
        time = np.arange(4) * 0.25
        df = dense_to_intervals(arr, time, ["a", "b"])
        assert len(df) == 2
        assert set(df["individual"]) == {"a", "b"}

    def test_mismatched_columns_raises(self):
        arr = np.zeros((5, 2), dtype=np.int8)
        time = np.arange(5)
        with pytest.raises(ValueError, match="individuals"):
            dense_to_intervals(arr, time, ["only_one"])


# ---------------------------------------------------------------------------
# intervals_to_dense
# ---------------------------------------------------------------------------

class TestIntervalsToDense:
    def test_basic(self, sample_intervals):
        dense = intervals_to_dense(sample_intervals, sample_rate=10.0, duration=0.9, individuals=["bird1"])
        assert dense.shape == (10, 1)
        assert dense[2, 0] == 1
        assert dense[4, 0] == 1
        assert dense[7, 0] == 2
        assert dense[0, 0] == 0
        assert dense[5, 0] == 0

    def test_empty_df(self):
        df = empty_intervals()
        dense = intervals_to_dense(df, 10.0, 1.0, ["ind1"])
        assert dense.shape == (11, 1)
        assert np.all(dense == 0)

    def test_unknown_individual_ignored(self):
        df = pd.DataFrame(
            {"onset_s": [0.0], "offset_s": [0.5], "label_id": [1], "individual": ["unknown"]}
        )
        dense = intervals_to_dense(df, 10.0, 1.0, ["bird1"])
        assert np.all(dense == 0)


# ---------------------------------------------------------------------------
# Round-trip dense -> intervals -> dense
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_round_trip(self, sample_dense):
        arr, time, inds = sample_dense
        df = dense_to_intervals(arr, time, inds)
        sr = 1.0 / np.median(np.diff(time))
        duration = time[-1]
        reconstructed = intervals_to_dense(df, sr, duration, inds, n_samples=len(time))
        np.testing.assert_array_equal(arr.flatten(), reconstructed.flatten())

    def test_round_trip_multi_individual(self):
        arr = np.array(
            [[1, 0], [1, 2], [1, 2], [0, 2], [0, 0]], dtype=np.int8
        )
        time = np.arange(5) * 0.1
        inds = ["a", "b"]
        df = dense_to_intervals(arr, time, inds)
        sr = 10.0
        duration = time[-1]
        reconstructed = intervals_to_dense(df, sr, duration, inds)
        np.testing.assert_array_equal(arr, reconstructed)


# ---------------------------------------------------------------------------
# intervals_to_xr / xr_to_intervals
# ---------------------------------------------------------------------------

class TestXarrayConversion:
    def test_round_trip(self, sample_intervals):
        ds = intervals_to_xr(sample_intervals)
        assert "segment" in ds.dims
        assert "onset_s" in ds.data_vars
        df_back = xr_to_intervals(ds)
        assert len(df_back) == len(sample_intervals)
        np.testing.assert_array_almost_equal(
            df_back["onset_s"].values, sample_intervals["onset_s"].values
        )

    def test_empty(self):
        ds = intervals_to_xr(empty_intervals())
        assert ds.sizes.get("segment", 0) == 0
        df_back = xr_to_intervals(ds)
        assert len(df_back) == 0

    def test_missing_vars(self):
        ds = xr.Dataset()
        df = xr_to_intervals(ds)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# add_interval
# ---------------------------------------------------------------------------

class TestAddInterval:
    def test_add_no_overlap(self):
        df = empty_intervals()
        df = add_interval(df, 1.0, 2.0, 1, "bird1")
        assert len(df) == 1
        df = add_interval(df, 3.0, 4.0, 2, "bird1")
        assert len(df) == 2

    def test_add_full_overlap_replaces(self):
        df = empty_intervals()
        df = add_interval(df, 1.0, 3.0, 1, "bird1")
        df = add_interval(df, 0.5, 3.5, 2, "bird1")
        assert len(df) == 1
        assert df.iloc[0]["label_id"] == 2

    def test_add_partial_overlap_trims(self):
        df = empty_intervals()
        df = add_interval(df, 1.0, 3.0, 1, "bird1")
        df = add_interval(df, 2.0, 4.0, 2, "bird1")
        assert len(df) == 2
        assert df.iloc[0]["offset_s"] == 2.0  # trimmed
        assert df.iloc[1]["onset_s"] == 2.0

    def test_add_splits_existing(self):
        df = empty_intervals()
        df = add_interval(df, 1.0, 5.0, 1, "bird1")
        df = add_interval(df, 2.0, 3.0, 2, "bird1")
        assert len(df) == 3
        labels = sorted(df["label_id"].tolist())
        assert labels == [1, 1, 2]

    def test_add_different_individual_no_conflict(self):
        df = empty_intervals()
        df = add_interval(df, 1.0, 3.0, 1, "bird1")
        df = add_interval(df, 1.0, 3.0, 2, "bird2")
        assert len(df) == 2

    def test_swapped_times(self):
        df = add_interval(empty_intervals(), 3.0, 1.0, 1, "bird1")
        assert df.iloc[0]["onset_s"] == 1.0
        assert df.iloc[0]["offset_s"] == 3.0


# ---------------------------------------------------------------------------
# delete_interval
# ---------------------------------------------------------------------------

class TestDeleteInterval:
    def test_delete(self):
        df = add_interval(empty_intervals(), 1.0, 2.0, 1, "bird1")
        df = add_interval(df, 3.0, 4.0, 2, "bird1")
        df = delete_interval(df, 0)
        assert len(df) == 1
        assert df.iloc[0]["label_id"] == 2


# ---------------------------------------------------------------------------
# find_interval_at
# ---------------------------------------------------------------------------

class TestFindIntervalAt:
    def test_find(self, sample_intervals):
        idx = find_interval_at(sample_intervals, 0.3, "bird1")
        assert idx == 0

    def test_find_second(self, sample_intervals):
        idx = find_interval_at(sample_intervals, 0.8, "bird1")
        assert idx == 1

    def test_miss(self, sample_intervals):
        idx = find_interval_at(sample_intervals, 0.55, "bird1")
        assert idx is None

    def test_wrong_individual(self, sample_intervals):
        idx = find_interval_at(sample_intervals, 0.3, "bird2")
        assert idx is None


# ---------------------------------------------------------------------------
# get_interval_bounds
# ---------------------------------------------------------------------------

class TestGetIntervalBounds:
    def test_bounds(self, sample_intervals):
        onset, offset, lid = get_interval_bounds(sample_intervals, 0)
        assert onset == 0.2
        assert offset == 0.4
        assert lid == 1


# ---------------------------------------------------------------------------
# purge_short_intervals
# ---------------------------------------------------------------------------

class TestPurgeShortIntervals:
    def test_purge(self):
        df = add_interval(empty_intervals(), 0.0, 0.01, 1, "bird1")  # 10ms
        df = add_interval(df, 1.0, 2.0, 2, "bird1")  # 1s
        df = purge_short_intervals(df, 0.05)
        assert len(df) == 1
        assert df.iloc[0]["label_id"] == 2

    def test_per_label(self):
        df = add_interval(empty_intervals(), 0.0, 0.1, 1, "bird1")
        df = add_interval(df, 1.0, 1.08, 2, "bird1")
        df = purge_short_intervals(df, 0.05, {2: 0.1})
        assert len(df) == 1
        assert df.iloc[0]["label_id"] == 1


# ---------------------------------------------------------------------------
# stitch_intervals
# ---------------------------------------------------------------------------

class TestStitchIntervals:
    def test_stitch(self):
        df = add_interval(empty_intervals(), 0.0, 1.0, 1, "bird1")
        df = add_interval(df, 1.05, 2.0, 1, "bird1")
        df = stitch_intervals(df, 0.1, "bird1")
        assert len(df) == 1
        assert df.iloc[0]["offset_s"] == 2.0

    def test_no_stitch_different_labels(self):
        df = add_interval(empty_intervals(), 0.0, 1.0, 1, "bird1")
        df = add_interval(df, 1.05, 2.0, 2, "bird1")
        df = stitch_intervals(df, 0.1, "bird1")
        assert len(df) == 2

    def test_no_stitch_large_gap(self):
        df = add_interval(empty_intervals(), 0.0, 1.0, 1, "bird1")
        df = add_interval(df, 2.0, 3.0, 1, "bird1")
        df = stitch_intervals(df, 0.1, "bird1")
        assert len(df) == 2
