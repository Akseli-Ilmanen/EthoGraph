"""Tests for ethograph.utils.label_intervals module."""

import numpy as np
import pandas as pd
import pytest

from ethograph.features.changepoints import correct_changepoints
from ethograph.labels.intervals import (
    NO_RECIPIENT,
    add_interval,
    add_point,
    delete_interval,
    empty_intervals,
    ensure_confidence,
    ensure_individual_rec,
    find_interval_at,
    get_interval_bounds,
    purge_short_intervals,
    select_subject,
    snap_boundaries,
    stitch_intervals,
)
from ethograph.labels.ml import dense_to_intervals, intervals_to_dense

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
            "labels": [1, 2],
            "individual": ["bird1", "bird1"],
        }
    )


# ---------------------------------------------------------------------------
# empty_intervals
# ---------------------------------------------------------------------------


class TestEmptyIntervals:
    def test_columns(self):
        df = empty_intervals()
        assert list(df.columns) == [
            "onset_s",
            "offset_s",
            "labels",
            "individual",
            "individual_rec",
            "event_type",
            "confidence",
            "labeling_method",
        ]

    def test_empty(self):
        df = empty_intervals()
        assert len(df) == 0


# ---------------------------------------------------------------------------
# dense_to_intervals
# ---------------------------------------------------------------------------


class TestDenseToIntervals:
    def test_basic(self, sample_dense):
        arr, time, inds = sample_dense
        df = dense_to_intervals(arr, inds, time_coord=time)
        assert len(df) == 2
        assert df.iloc[0]["labels"] == 1
        assert df.iloc[1]["labels"] == 2
        np.testing.assert_almost_equal(df.iloc[0]["onset_s"], 0.2)
        np.testing.assert_almost_equal(df.iloc[0]["offset_s"], 0.4)

    def test_all_zeros(self):
        arr = np.zeros((10, 1), dtype=np.int8)
        time = np.arange(10) * 0.1
        df = dense_to_intervals(arr, ["ind1"], time_coord=time)
        assert len(df) == 0

    def test_1d_input(self):
        arr = np.array([0, 1, 1, 0], dtype=np.int8)
        time = np.arange(4) * 0.5
        df = dense_to_intervals(arr, ["ind1"], time_coord=time)
        assert len(df) == 1
        assert df.iloc[0]["onset_s"] == 0.5
        assert df.iloc[0]["offset_s"] == 1.0

    def test_multi_individual(self):
        arr = np.array([[1, 0], [1, 2], [0, 2], [0, 0]], dtype=np.int8)
        time = np.arange(4) * 0.25
        df = dense_to_intervals(arr, ["a", "b"], time_coord=time)
        assert len(df) == 2
        assert set(df["individual"]) == {"a", "b"}

    def test_mismatched_columns_raises(self):
        arr = np.zeros((5, 2), dtype=np.int8)
        with pytest.raises(ValueError, match="individuals"):
            dense_to_intervals(arr, ["only_one"], sample_rate=1.0)


# ---------------------------------------------------------------------------
# intervals_to_dense
# ---------------------------------------------------------------------------


class TestIntervalsToDense:
    def test_basic(self, sample_intervals):
        dense = intervals_to_dense(sample_intervals, sample_rate=10.0, individuals=["bird1"], n_samples=10)
        assert dense.shape == (10, 1)
        assert dense[2, 0] == 1
        assert dense[4, 0] == 1
        assert dense[7, 0] == 2
        assert dense[0, 0] == 0
        assert dense[5, 0] == 0

    def test_empty_df(self):
        df = empty_intervals()
        dense = intervals_to_dense(df, 10.0, ["ind1"], n_samples=11)
        assert dense.shape == (11, 1)
        assert np.all(dense == 0)

    def test_unknown_individual_ignored(self):
        df = pd.DataFrame({"onset_s": [0.0], "offset_s": [0.5], "labels": [1], "individual": ["unknown"]})
        dense = intervals_to_dense(df, 10.0, ["bird1"], n_samples=11)
        assert np.all(dense == 0)


# ---------------------------------------------------------------------------
# Round-trip dense -> intervals -> dense
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip(self, sample_dense):
        arr, time, inds = sample_dense
        df = dense_to_intervals(arr, inds, time_coord=time)
        sr = 1.0 / np.median(np.diff(time))
        reconstructed = intervals_to_dense(df, sr, inds, n_samples=len(time))
        np.testing.assert_array_equal(arr.flatten(), reconstructed.flatten())

    def test_round_trip_multi_individual(self):
        arr = np.array([[1, 0], [1, 2], [1, 2], [0, 2], [0, 0]], dtype=np.int8)
        time = np.arange(5) * 0.1
        inds = ["a", "b"]
        df = dense_to_intervals(arr, inds, time_coord=time)
        sr = 10.0
        reconstructed = intervals_to_dense(df, sr, inds, n_samples=len(time))
        np.testing.assert_array_equal(arr, reconstructed)


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
        assert df.iloc[0]["labels"] == 2

    def test_add_partial_overlap_trims(self):
        df = empty_intervals()
        df = add_interval(df, 1.0, 3.0, 1, "bird1")
        df = add_interval(df, 2.0, 4.0, 2, "bird1")
        assert len(df) == 2
        assert df.iloc[0]["offset_s"] == pytest.approx(2.0, abs=0.002)  # trimmed
        assert df.iloc[1]["onset_s"] == pytest.approx(2.0, abs=0.002)

    def test_add_splits_existing(self):
        df = empty_intervals()
        df = add_interval(df, 1.0, 5.0, 1, "bird1")
        df = add_interval(df, 2.0, 3.0, 2, "bird1")
        assert len(df) == 3
        labels = sorted(df["labels"].tolist())
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
        assert df.iloc[0]["labels"] == 2


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
        assert df.iloc[0]["labels"] == 2

    def test_per_label(self):
        df = add_interval(empty_intervals(), 0.0, 0.1, 1, "bird1")
        df = add_interval(df, 1.0, 1.08, 2, "bird1")
        df = purge_short_intervals(df, 0.05, {2: 0.1})
        assert len(df) == 1
        assert df.iloc[0]["labels"] == 1


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


# ---------------------------------------------------------------------------
# snap_boundaries
# ---------------------------------------------------------------------------


class TestSnapBoundaries:
    def test_snap_to_nearest_cp(self):
        df = add_interval(empty_intervals(), 1.0, 3.0, 1, "bird1")
        cp_times = np.array([0.9, 3.1])
        result = snap_boundaries(df, cp_times, max_expansion_s=0.2, max_shrink_s=0.2)
        assert len(result) == 1
        np.testing.assert_almost_equal(result.iloc[0]["onset_s"], 0.9)
        np.testing.assert_almost_equal(result.iloc[0]["offset_s"], 3.1)

    def test_max_expansion_enforced(self):
        df = add_interval(empty_intervals(), 1.0, 3.0, 1, "bird1")
        cp_times = np.array([0.5, 3.6])  # 0.5s expansion on each side
        result = snap_boundaries(df, cp_times, max_expansion_s=0.3, max_shrink_s=0.3)
        np.testing.assert_almost_equal(result.iloc[0]["onset_s"], 1.0)  # not snapped
        np.testing.assert_almost_equal(result.iloc[0]["offset_s"], 3.0)  # not snapped

    def test_max_shrink_enforced(self):
        df = add_interval(empty_intervals(), 1.0, 3.0, 1, "bird1")
        cp_times = np.array([1.5, 2.5])  # 0.5s shrink on each side
        result = snap_boundaries(df, cp_times, max_expansion_s=1.0, max_shrink_s=0.3)
        np.testing.assert_almost_equal(result.iloc[0]["onset_s"], 1.0)  # not snapped
        np.testing.assert_almost_equal(result.iloc[0]["offset_s"], 3.0)  # not snapped

    def test_empty_df(self):
        result = snap_boundaries(empty_intervals(), np.array([1.0]), 0.5, 0.5)
        assert len(result) == 0

    def test_empty_cp_times(self):
        df = add_interval(empty_intervals(), 1.0, 3.0, 1, "bird1")
        result = snap_boundaries(df, np.array([]), 0.5, 0.5)
        assert len(result) == 1
        np.testing.assert_almost_equal(result.iloc[0]["onset_s"], 1.0)

    def test_invalid_range_fallback(self):
        df = add_interval(empty_intervals(), 1.2, 1.25, 1, "bird1")
        cp_times = np.array([1.3])  # onset shrinks to 1.3, offset expands to 1.3 -> onset >= offset
        result = snap_boundaries(df, cp_times, max_expansion_s=1.0, max_shrink_s=1.0)
        np.testing.assert_almost_equal(result.iloc[0]["onset_s"], 1.2)
        np.testing.assert_almost_equal(result.iloc[0]["offset_s"], 1.25)

    def test_overlap_resolution(self):
        df = add_interval(empty_intervals(), 1.0, 2.0, 1, "bird1")
        df = add_interval(df, 2.1, 3.0, 2, "bird1")
        cp_times = np.array([1.0, 2.2, 3.0])  # label 1 offset snaps to 2.2, overlapping label 2
        result = snap_boundaries(df, cp_times, max_expansion_s=0.3, max_shrink_s=0.3)
        assert result.iloc[0]["offset_s"] <= result.iloc[1]["onset_s"]

    def test_multi_individual(self):
        df = add_interval(empty_intervals(), 1.0, 2.0, 1, "bird1")
        df = add_interval(df, 1.0, 2.0, 2, "bird2")
        cp_times = np.array([0.9, 2.1])
        result = snap_boundaries(df, cp_times, max_expansion_s=0.2, max_shrink_s=0.2)
        assert len(result) == 2
        for _, row in result.iterrows():
            np.testing.assert_almost_equal(row["onset_s"], 0.9)
            np.testing.assert_almost_equal(row["offset_s"], 2.1)


# ---------------------------------------------------------------------------
# correct_changepoints (only stich/purge, snapping done in integrationt test)
# ---------------------------------------------------------------------------


class TestCorrectChangepoints:
    def test_per_label_thresholds(self):
        df = add_interval(empty_intervals(), 1.0, 1.08, 1, "bird1")  # 80ms
        df = add_interval(df, 2.0, 2.08, 2, "bird1")  # 80ms
        result = correct_changepoints(
            df,
            np.array([1.0, 1.08, 2.0, 2.08]),
            min_duration_s=0.05,
            stitch_gap_s=0.1,
            max_expansion_s=0.5,
            max_shrink_s=0.5,
            label_thresholds_s={2: 0.1},  # label 2 needs 100ms
        )
        assert len(result) == 1
        assert result.iloc[0]["labels"] == 1


# ---------------------------------------------------------------------------
# Recipients: (actor, recipient) pairs are independent tracks
# ---------------------------------------------------------------------------


class TestRecipient:
    def test_default_is_no_recipient(self):
        df = add_interval(empty_intervals(), 1.0, 2.0, 1, "bird1")
        assert df.iloc[0]["individual_rec"] == NO_RECIPIENT

    def test_pairs_do_not_overwrite_each_other(self):
        """The same actor labelled towards two recipients keeps both."""
        df = add_interval(empty_intervals(), 1.0, 3.0, 1, "bird1", individual_rec="bird2")
        df = add_interval(df, 1.0, 3.0, 2, "bird1", individual_rec="bird3")
        assert len(df) == 2
        assert sorted(df["individual_rec"]) == ["bird2", "bird3"]

    def test_same_pair_still_resolves_overlap(self):
        df = add_interval(empty_intervals(), 1.0, 3.0, 1, "bird1", individual_rec="bird2")
        df = add_interval(df, 0.5, 3.5, 2, "bird1", individual_rec="bird2")
        assert len(df) == 1
        assert df.iloc[0]["labels"] == 2

    def test_solo_and_dyadic_are_separate(self):
        df = add_interval(empty_intervals(), 1.0, 3.0, 1, "bird1")
        df = add_interval(df, 1.0, 3.0, 2, "bird1", individual_rec="bird2")
        assert len(df) == 2

    def test_find_matches_the_pair_only(self):
        df = add_interval(empty_intervals(), 1.0, 3.0, 1, "bird1", individual_rec="bird2")
        assert find_interval_at(df, 2.0, "bird1", individual_rec="bird2") is not None
        assert find_interval_at(df, 2.0, "bird1", individual_rec=NO_RECIPIENT) is None
        # None on either side is the "any" fallback used for click selection.
        assert find_interval_at(df, 2.0, "bird1") is not None

    def test_select_subject(self):
        df = add_interval(empty_intervals(), 1.0, 2.0, 1, "bird1")
        df = add_interval(df, 3.0, 4.0, 1, "bird1", individual_rec="bird2")
        assert len(select_subject(df, "bird1", NO_RECIPIENT)) == 1
        assert len(select_subject(df, "bird1", "bird2")) == 1
        assert len(select_subject(df, "bird1")) == 2

    def test_stitch_never_merges_across_pairs(self):
        df = add_interval(empty_intervals(), 0.0, 1.0, 1, "bird1", individual_rec="bird2")
        df = add_interval(df, 1.05, 2.0, 1, "bird1", individual_rec="bird3")
        assert len(stitch_intervals(df, max_gap_s=0.5)) == 2

    def test_legacy_frame_without_the_column_reads_as_solo(self):
        legacy = pd.DataFrame(
            {
                "onset_s": [1.0],
                "offset_s": [2.0],
                "labels": [1],
                "individual": ["bird1"],
            }
        )
        assert len(select_subject(ensure_individual_rec(legacy), "bird1", NO_RECIPIENT)) == 1


class TestConfidence:
    """A label's confidence: 1.0 by hand, the model's own score otherwise."""

    def test_hand_placed_labels_are_certain(self):
        df = add_interval(empty_intervals(), 0.0, 1.0, 1, "bird1")
        df = add_point(df, 2.0, 2, "bird1")
        assert list(df["confidence"]) == [1.0, 1.0]

    def test_model_score_is_stored(self):
        df = add_point(empty_intervals(), 2.0, 1, "bird1", confidence=0.37)
        assert df.iloc[0]["confidence"] == pytest.approx(0.37)

    def test_trimmed_remnant_keeps_its_own_confidence(self):
        df = add_point(empty_intervals(), 9.0, 3, "bird1", confidence=0.2)
        df = add_interval(df, 0.0, 2.0, 1, "bird1", confidence=0.5)
        df = add_interval(df, 1.0, 3.0, 2, "bird1")
        by_label = dict(zip(df["labels"], df["confidence"]))
        assert by_label[1] == pytest.approx(0.5)  # trimmed remnant, unchanged
        assert by_label[2] == pytest.approx(1.0)  # the new hand-placed one
        assert by_label[3] == pytest.approx(0.2)  # points are never touched

    def test_stitch_takes_the_weakest_part(self):
        df = add_interval(empty_intervals(), 0.0, 1.0, 1, "bird1", confidence=0.9)
        df = add_interval(df, 1.05, 2.0, 1, "bird1", confidence=0.4)
        stitched = stitch_intervals(df, max_gap_s=0.5)
        assert len(stitched) == 1
        assert stitched.iloc[0]["confidence"] == pytest.approx(0.4)

    def test_legacy_frame_without_the_column_reads_as_certain(self):
        legacy = pd.DataFrame(
            {
                "onset_s": [1.0],
                "offset_s": [2.0],
                "labels": [1],
                "individual": ["bird1"],
            }
        )
        assert ensure_confidence(legacy).iloc[0]["confidence"] == 1.0
