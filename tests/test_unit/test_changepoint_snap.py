"""A click snaps to the changepoints the clicked panel draws — and to nothing else.

The lineplot draws a feature's changepoint masks at the panel's own keypoint /
individual selections (``XarrayLoader.select``); a click snaps to
``XarrayLoader.get_cp_times`` for the same feature and selections. Both read
``changepoint_fired``, so the two sets agree by construction. When they
diverged — the snap set was the OR over *every* keypoint, individual and
target feature — every click still snapped, just to marks the user could not
see, which read as "snapping is irregular and gets worse the further the click
is from the mark I am aiming at".
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ethograph.features.changepoints import (
    changepoint_fired,
    dataset_changepoint_times,
    find_nearest_turning_points_binary,
)
from ethograph.io import schema
from ethograph.io.catalog import XarrayLoader
from ethograph.labels.intervals import (
    HUMAN_CONFIDENCE,
    LABELING_AUTOMATED,
    LABELING_MANUAL,
    snap_boundaries,
)

T = 200
FS = 200.0
TIME = np.arange(T) / FS


def _mask(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    time_axis: int,
    marks: dict[int, tuple[int, ...]],
    target: str = "speed",
    coords: dict | None = None,
) -> xr.DataArray:
    """A mask with *marks* per index along the first non-time axis (time at *time_axis*)."""
    data = np.zeros(shape, dtype=np.int8)
    other_axis = next(i for i in range(len(shape)) if i != time_axis)
    for other_idx, samples in marks.items():
        for m in samples:
            idx = [0] * len(shape)
            idx[time_axis] = m
            idx[other_axis] = other_idx
            data[tuple(idx)] = 1
    da = xr.DataArray(data, dims=dims, coords=coords)
    da.attrs.update(schema.changepoint_attrs(target_feature=target))
    return da


def _moll_like(marks_by_keypoint: dict[int, tuple[int, ...]], target: str = "speed") -> xr.Dataset:
    """Moll2025's layout: ``(keypoint, individual, time)`` masks beside a speed feature."""
    keypoints = ["beak", "head", "tail"]
    coords = {"keypoint": keypoints, "individual": ["bird_0"], "time": TIME}
    speed = xr.DataArray(np.random.default_rng(0).random((3, 1, T)), dims=("keypoint", "individual", "time"))
    mask = _mask(("keypoint", "individual", "time"), (3, 1, T), 2, marks_by_keypoint, target=target)
    return xr.Dataset({"speed": speed, f"{target}_troughs": mask}, coords=coords)


class TestDrawnIsSnapped:
    """The lineplot's marks and the snap candidates are one set."""

    @pytest.mark.parametrize(
        "dims, shape, time_axis",
        [
            (("time",), (T,), 0),
            (("time", "keypoint"), (T, 3), 0),
            (("keypoint", "individual", "time"), (3, 1, T), 2),  # Moll2025's layout
            (("individual", "time", "keypoint"), (2, T, 3), 1),  # time in the middle
        ],
    )
    def test_marks_span_the_trial_whatever_the_axis_order(self, dims, shape, time_axis):
        marks = (17, 88, 150)
        if len(shape) == 1:
            data = np.zeros(T, dtype=np.int8)
            data[list(marks)] = 1
            mask = xr.DataArray(data, dims=dims)
            mask.attrs.update(schema.changepoint_attrs(target_feature="speed"))
        else:
            mask = _mask(dims, shape, time_axis, {0: marks})
        ds = xr.Dataset({"speed_troughs": mask}, coords={"time": TIME})

        fired = changepoint_fired(ds["speed_troughs"])
        assert fired.shape == (T,), "the reading is always on the time axis"
        np.testing.assert_array_equal(np.flatnonzero(fired), marks)
        np.testing.assert_allclose(dataset_changepoint_times(ds), TIME[list(marks)])

    def test_lineplot_and_click_read_one_set(self):
        ds = _moll_like({0: (20, 60), 1: (90,), 2: (130, 170)})
        loader = XarrayLoader(ds)
        selections = {"keypoint": "head", "individual": "bird_0"}

        drawn = loader.select("speed", selections).changepoints
        assert drawn is not None
        drawn_times = np.unique(np.concatenate([TIME[np.flatnonzero(f)] for f in drawn.values()]))
        snapped = loader.get_cp_times("speed", selections)

        np.testing.assert_allclose(snapped, drawn_times)
        np.testing.assert_allclose(snapped, TIME[[90]])


class TestSelectionsPinTheMarks:
    """Only the displayed keypoint's / individual's marks are candidates."""

    def test_another_keypoints_mark_is_not_a_candidate(self):
        ds = _moll_like({0: (20, 60), 1: (90,), 2: (130, 170)})
        loader = XarrayLoader(ds)

        head = loader.get_cp_times("speed", {"keypoint": "head", "individual": "bird_0"})
        beak = loader.get_cp_times("speed", {"keypoint": "beak", "individual": "bird_0"})

        np.testing.assert_allclose(head, TIME[[90]])
        np.testing.assert_allclose(beak, TIME[[20, 60]])

    def test_a_free_dim_is_ored_across(self):
        """ "All" keypoints (the dim absent from the selections) shows every mark."""
        ds = _moll_like({0: (20,), 1: (90,), 2: (170,)})
        every = XarrayLoader(ds).get_cp_times("speed", {"individual": "bird_0"})
        np.testing.assert_allclose(every, TIME[[20, 90, 170]])

    def test_a_dim_the_mask_lacks_is_ignored(self):
        ds = _moll_like({0: (20,), 1: (90,)})
        times = XarrayLoader(ds).get_cp_times("speed", {"keypoint": "head", "space": "x", "unit": "3"})
        np.testing.assert_allclose(times, TIME[[90]])

    def test_coordless_dim_is_pinned_by_index(self):
        mask = _mask(("time", "keypoint"), (T, 3), 0, {0: (10,), 2: (150,)})
        ds = xr.Dataset({"speed_troughs": mask}, coords={"time": TIME})
        np.testing.assert_allclose(dataset_changepoint_times(ds, selections={"keypoint": "2"}), TIME[[150]])
        np.testing.assert_allclose(dataset_changepoint_times(ds, selections={"keypoint": 0}), TIME[[10]])


class TestTargetFeature:
    """Masks belong to a feature; the ones on another curve are not candidates."""

    def test_only_the_displayed_features_masks(self):
        ds = _moll_like({0: (20,)})
        other = _mask(("keypoint", "individual", "time"), (3, 1, T), 2, {0: (150,)}, target="heading")
        ds["heading_troughs"] = other
        loader = XarrayLoader(ds)
        sel = {"keypoint": "beak", "individual": "bird_0"}

        np.testing.assert_allclose(loader.get_cp_times("speed", sel), TIME[[20]])
        np.testing.assert_allclose(loader.get_cp_times("heading", sel), TIME[[150]])

    def test_masks_of_two_features_never_blank_the_set(self):
        """Different target features are unioned when no feature is asked for, not refused."""
        ds = _moll_like({0: (20,)})
        ds["heading_troughs"] = _mask(("keypoint", "individual", "time"), (3, 1, T), 2, {0: (150,)}, target="heading")
        np.testing.assert_allclose(dataset_changepoint_times(ds), TIME[[20, 150]])

    def test_window_and_display_offset(self):
        ds = _moll_like({0: (20, 60, 150)})
        loader = XarrayLoader(ds)
        loader.set_display_offset_provider(lambda: 0.1)
        sel = {"keypoint": "beak", "individual": "bird_0"}

        np.testing.assert_allclose(loader.get_cp_times("speed", sel), TIME[[20, 60, 150]] - 0.1)
        windowed = loader.get_cp_times("speed", sel, t0=0.0, t1=0.3)
        np.testing.assert_allclose(windowed, TIME[[20, 60]] - 0.1)

    def test_no_masks_is_empty(self):
        ds = xr.Dataset({"speed": ("time", np.zeros(T))}, coords={"time": TIME})
        assert XarrayLoader(ds).get_cp_times("speed", {}).size == 0


class TestTurningPointsHonourTheirParameters:
    """``prominence``/``distance`` reach SciPy under their own names."""

    def test_prominence_filters_the_small_peak(self):
        x = np.zeros(300)
        x[100:110] = 1.0  # small plateau: prominence 1
        x[200:210] = 10.0  # large plateau: prominence 10
        big_only = find_nearest_turning_points_binary(x, threshold=0.5, prominence=5.0, distance=1)
        both = find_nearest_turning_points_binary(x, threshold=0.5, prominence=0.5, distance=1)

        assert not big_only[90:120].any(), "a peak below the prominence must contribute no boundary"
        assert big_only[190:220].any()
        assert both[90:120].any() and both[190:220].any()


class TestSnapBoundariesKeepsProvenance:
    """Snapping moves a boundary; it never rewrites who placed the label or how sure they were."""

    def test_confidence_and_method_survive(self):
        df = pd.DataFrame(
            {
                "onset_s": [1.02],
                "offset_s": [2.03],
                "labels": [3],
                "individual": ["bird_0"],
                "individual_rec": [""],
                "event_type": ["state"],
                "confidence": [0.42],
                "labeling_method": [LABELING_AUTOMATED],
            }
        )
        out = snap_boundaries(df, np.array([1.0, 2.0]), max_expansion_s=0.1, max_shrink_s=0.1)

        assert out.loc[0, "onset_s"] == pytest.approx(1.0)
        assert out.loc[0, "offset_s"] == pytest.approx(2.0)
        assert out.loc[0, "confidence"] == pytest.approx(0.42)
        assert out.loc[0, "labeling_method"] == LABELING_AUTOMATED
        assert out.loc[0, "confidence"] != HUMAN_CONFIDENCE
        assert out.loc[0, "labeling_method"] != LABELING_MANUAL

    def test_two_labels_meeting_at_one_changepoint_both_sit_on_it(self):
        """A shared boundary is not an overlap: neither edge is pushed off the changepoint."""
        df = pd.DataFrame(
            {
                "onset_s": [1.0, 2.03],
                "offset_s": [1.97, 3.0],
                "labels": [1, 2],
                "individual": ["bird_0", "bird_0"],
                "individual_rec": ["", ""],
                "event_type": ["state", "state"],
            }
        )
        out = snap_boundaries(df, np.array([1.0, 2.0, 3.0]), max_expansion_s=0.1, max_shrink_s=0.1)

        assert out.loc[0, "offset_s"] == pytest.approx(2.0)
        assert out.loc[1, "onset_s"] == pytest.approx(2.0)

    def test_a_true_overlap_is_clipped_onto_the_next_onset(self):
        df = pd.DataFrame(
            {
                "onset_s": [1.0, 2.1],
                "offset_s": [2.0, 3.0],
                "labels": [1, 2],
                "individual": ["bird_0", "bird_0"],
                "individual_rec": ["", ""],
                "event_type": ["state", "state"],
            }
        )
        # label 1's offset expands to 2.2, past label 2's onset, which may only shrink 0.05 s and so stays at 2.1
        out = snap_boundaries(df, np.array([2.2]), max_expansion_s=0.3, max_shrink_s=0.05)

        assert out.loc[1, "onset_s"] == pytest.approx(2.1)
        assert out.loc[0, "offset_s"] == pytest.approx(2.1)
