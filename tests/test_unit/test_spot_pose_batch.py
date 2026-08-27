"""Merging a clip's keypoints onto its trial's clock.

The one conversion with a sign that nothing checks: ``trial = video +
offset``. Getting it backwards shifts every keypoint by the offset and is
invisible in the result, so the direction is tested against a known event.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ethograph.spot.pose_batch import sample_onto_trial


def _trial(n=200, fs=200.0):
    t = np.arange(n) / fs
    return xr.Dataset({"speed": ("time", np.zeros(n))}, coords={"time": t})


def _pose(n=50, fps=100.0):
    """A keypoint that jumps to 1.0 at video frame 20 (0.20 s on the video clock)."""
    x = np.zeros((n, 2, 1, 1))
    x[20:, 0, 0, 0] = 1.0
    conf = np.ones((n, 1, 1))
    return xr.Dataset(
        {
            "position": (("time", "space", "keypoint", "individual"), x),
            "confidence": (("time", "keypoint", "individual"), conf),
        },
        coords={"time": np.arange(n) / fps, "space": ["x", "y"], "keypoint": ["beak"], "individual": ["crow"]},
    )


class TestSampleOntoTrial:
    def test_the_jump_lands_at_video_time_plus_offset(self):
        ds = sample_onto_trial(_trial(), _pose(), offset=0.3)
        x = ds.position.sel(space="x").values.ravel()
        t = ds.time.values
        first = t[np.argmax(x >= 1.0)]
        assert first == pytest.approx(0.3 + 0.20, abs=0.006)  # trial = video + offset

    def test_outside_the_clip_is_nan_not_extrapolated(self):
        ds = sample_onto_trial(_trial(), _pose(), offset=0.3)
        x = ds.position.sel(space="x").values.ravel()
        assert np.isnan(x[ds.time.values < 0.3]).all()
        assert np.isnan(x[ds.time.values > 0.3 + 0.49]).all()

    def test_dims_and_confidence_come_along(self):
        ds = sample_onto_trial(_trial(), _pose(), offset=0.0, var="pose2d")
        assert ds.pose2d.dims == ("time", "space", "keypoint", "individual")
        assert "pose2d_confidence" in ds
        assert list(ds.keypoint.values) == ["beak"]

    def test_an_existing_pose_is_never_replaced(self):
        ds = _trial().assign(position=("time", np.zeros(200)))
        with pytest.raises(ValueError, match="already has"):
            sample_onto_trial(ds, _pose(), offset=0.0)
