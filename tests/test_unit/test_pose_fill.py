"""Fill backend protocol conformance and the anchor-preservation invariant.

Backend tests use a fake predictor returning known tracks, so no torch is
needed in CI. The assertion that matters for every backend is the same:
**anchor frames come back exactly as they were labelled.**
"""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui import pose_fill
from ethograph.gui.pose_fill import (
    CoTrackerBackend,
    FillBackend,
    OpticalFlowBackend,
    SplineBackend,
    _GapBackend,
    available_backends,
    build_backend,
    resolve_device,
)

N_FRAMES = 21
N_KEYPOINTS = 2


def _anchors() -> dict[int, np.ndarray]:
    return {
        0: np.array([[0.0, 0.0], [10.0, 10.0]]),
        10: np.array([[10.0, 5.0], [20.0, 15.0]]),
        20: np.array([[20.0, 0.0], [30.0, 10.0]]),
    }


def _partial_anchors() -> dict[int, np.ndarray]:
    """Beak labelled on some frames, tail on others — the normal case."""
    return {
        0: np.array([[0.0, 0.0], [10.0, 10.0]]),
        7: np.array([[7.0, 3.0], [np.nan, np.nan]]),
        14: np.array([[np.nan, np.nan], [24.0, 12.0]]),
        20: np.array([[20.0, 0.0], [30.0, 10.0]]),
    }


class _HoldBackend(_GapBackend):
    """Exercises the shared gap machinery without torch or OpenCV.

    ``_track`` holds every query point still, so blending, scaling and endpoint
    seeding are observable in isolation from any tracker.
    """

    name = "hold"

    def _track(self, clip, points, query_frame):
        positions = np.repeat(np.asarray(points)[None], len(clip), axis=0)
        return positions.astype(np.float64), np.ones((len(clip), len(points)))


class _ScaledFrames:
    """Frame source decoded at half resolution (``scale`` back to source px)."""

    scale = 2.0

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class _FakePredictor:
    """Stands in for CoTrackerPredictor: holds every query point still."""

    def __call__(self, video, queries=None, backward_tracking=False):
        import torch

        n_frames = video.shape[1]
        points = queries[0, :, 1:]
        tracks = points[None, None].repeat(1, n_frames, 1, 1)
        visibility = torch.ones((1, n_frames, points.shape[0]))
        return tracks, visibility


def _frames(n: int = N_FRAMES, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(n, size, size, 3), dtype=np.uint8)


# ----------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------


def test_backends_satisfy_the_protocol():
    for backend in (SplineBackend(), OpticalFlowBackend(), CoTrackerBackend(object())):
        assert isinstance(backend, FillBackend)
        assert isinstance(backend.name, str)
        assert isinstance(backend.requires_video, bool)


def test_spline_needs_no_video():
    assert SplineBackend().requires_video is False
    assert OpticalFlowBackend().requires_video is True


def test_available_backends_always_offers_spline():
    infos = {info.key: info for info in available_backends()}
    assert infos["spline"].available is True
    for key in ("flow", "cotracker"):
        assert infos[key].available or infos[key].hint


def test_build_backend_rejects_unknown_key():
    with pytest.raises(ValueError):
        build_backend("nope")


# ----------------------------------------------------------------------
# CoTracker checkpoint resolution
# ----------------------------------------------------------------------


def test_explicit_missing_checkpoint_raises_instead_of_downloading(tmp_path):
    """A path the user named must never be silently replaced by a download."""
    # torch too: cotracker.predictor imports it, and a half-installed env
    # (cotracker present, torch absent) is a real state to skip cleanly on.
    pytest.importorskip("cotracker")
    pytest.importorskip("torch")
    with pytest.raises(FileNotFoundError):
        pose_fill.load_cotracker_predictor(checkpoint=tmp_path / "nope.pth")


def test_absent_checkpoint_triggers_a_download(tmp_path, monkeypatch):
    """Regression: passing checkpoint=None through builds an *unloaded* network
    that returns confident nonsense — the weights must be fetched instead."""
    pytest.importorskip("cotracker")
    pytest.importorskip("torch")
    monkeypatch.setattr(pose_fill, "cotracker_checkpoint_dir", lambda: tmp_path / "absent")
    calls = []
    monkeypatch.setattr(pose_fill, "download_cotracker_checkpoint", lambda progress=None: calls.append(1))
    with pytest.raises(Exception):  # noqa: B017 - the stub returns no usable path
        pose_fill.load_cotracker_predictor()
    assert calls == [1]


def test_find_checkpoint_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(pose_fill, "cotracker_checkpoint_dir", lambda: tmp_path / "absent")
    assert pose_fill.find_cotracker_checkpoint() is None


def test_find_checkpoint_prefers_the_offline_model(tmp_path, monkeypatch):
    (tmp_path / "scaled_online.pth").touch()
    (tmp_path / "scaled_offline.pth").touch()
    monkeypatch.setattr(pose_fill, "cotracker_checkpoint_dir", lambda: tmp_path)
    assert pose_fill.find_cotracker_checkpoint().name == "scaled_offline.pth"


def test_find_checkpoint_honours_an_explicit_path(tmp_path):
    weights = tmp_path / "custom.pth"
    weights.touch()
    assert pose_fill.find_cotracker_checkpoint(weights) == weights
    assert pose_fill.find_cotracker_checkpoint(tmp_path / "nope.pth") is None


def test_missing_weights_do_not_block_the_backend(tmp_path, monkeypatch):
    """Weights download on first use, so absent weights are a note, not a block."""
    pytest.importorskip("torch")
    pytest.importorskip("cotracker")
    monkeypatch.setattr(pose_fill, "cotracker_checkpoint_dir", lambda: tmp_path / "absent")
    info = {i.key: i for i in available_backends()}["cotracker"]
    assert info.available is True
    assert "download" in info.hint


def test_install_hint_pins_a_commit():
    """An unpinned branch is the moving target we avoid torch.hub for."""
    assert pose_fill.COTRACKER_COMMIT in pose_fill.COTRACKER_INSTALL_HINT
    assert len(pose_fill.COTRACKER_COMMIT) == 40


def test_uninstalled_cotracker_reports_the_single_install_command(monkeypatch):
    monkeypatch.setattr(pose_fill, "_module_available", lambda name: name not in {"torch", "cotracker"})
    info = {i.key: i for i in available_backends()}["cotracker"]
    assert info.available is False
    assert info.hint == pose_fill.COTRACKER_INSTALL_HINT


def test_resolve_device_returns_a_usable_device():
    device = resolve_device()
    assert device in {"cpu", "cuda", "mps"}


def test_resolve_device_falls_back_when_preference_is_unavailable():
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present — nothing to fall back from")
    assert resolve_device("cuda") == resolve_device()


# ----------------------------------------------------------------------
# Spline
# ----------------------------------------------------------------------


def test_spline_preserves_anchors_exactly():
    anchors = _anchors()
    filled, confidence = SplineBackend().fill(anchors, N_FRAMES, None)

    assert filled.shape == (N_FRAMES, N_KEYPOINTS, 2)
    assert confidence.shape == (N_FRAMES, N_KEYPOINTS)
    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)
        np.testing.assert_allclose(confidence[frame], 1.0)


def test_spline_fills_every_frame():
    filled, _ = SplineBackend().fill(_anchors(), N_FRAMES, None)
    assert not np.any(np.isnan(filled))


def test_spline_interpolates_between_anchors():
    filled, _ = SplineBackend().fill(_anchors(), N_FRAMES, None)
    # x moves monotonically 0 -> 20, so the midpoint sits strictly between.
    assert 0.0 < filled[5, 0, 0] < 10.0


def test_spline_confidence_decays_away_from_anchors():
    _, confidence = SplineBackend().fill(_anchors(), N_FRAMES, None)
    assert confidence[5, 0] < confidence[1, 0] < 1.0
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))


def test_spline_handles_partial_anchors():
    anchors = _partial_anchors()
    filled, confidence = SplineBackend().fill(anchors, N_FRAMES, None)

    for frame, points in anchors.items():
        labelled = ~np.isnan(points[:, 0])
        np.testing.assert_allclose(filled[frame][labelled], points[labelled])
        np.testing.assert_allclose(confidence[frame][labelled], 1.0)
    assert not np.any(np.isnan(filled))


def test_spline_holds_endpoints_outside_the_anchored_span():
    anchors = {5: np.array([[5.0, 5.0], [1.0, 1.0]]), 15: np.array([[15.0, 5.0], [2.0, 1.0]])}
    filled, _ = SplineBackend().fill(anchors, N_FRAMES, None)
    np.testing.assert_allclose(filled[0, 0], filled[5, 0])
    np.testing.assert_allclose(filled[20, 0], filled[15, 0])


def test_spline_with_a_single_anchor_is_constant():
    anchors = {4: np.array([[3.0, 4.0], [5.0, 6.0]])}
    filled, _ = SplineBackend().fill(anchors, N_FRAMES, None)
    np.testing.assert_allclose(filled[0], anchors[4])
    np.testing.assert_allclose(filled[20], anchors[4])


def test_unlabelled_keypoint_stays_nan():
    anchors = {0: np.array([[1.0, 1.0], [np.nan, np.nan]])}
    filled, confidence = SplineBackend().fill(anchors, N_FRAMES, None)
    assert np.all(np.isnan(filled[:, 1]))
    assert np.all(confidence[:, 1] == 0.0)


def test_fill_without_anchors_raises():
    with pytest.raises(ValueError):
        SplineBackend().fill({}, N_FRAMES, None)


def test_progress_cancellation_stops_early():
    calls = []

    def progress(fraction):
        calls.append(fraction)
        return False

    SplineBackend().fill(_anchors(), N_FRAMES, None, progress)
    assert len(calls) == 1


# ----------------------------------------------------------------------
# Gap backends
# ----------------------------------------------------------------------


def test_gap_backend_requires_frames():
    with pytest.raises(ValueError):
        OpticalFlowBackend().fill(_anchors(), N_FRAMES, None)


def test_gap_backend_preserves_anchors():
    anchors = _anchors()
    filled, confidence = _HoldBackend().fill(anchors, N_FRAMES, _frames())

    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)
        np.testing.assert_allclose(confidence[frame], 1.0)


def test_gap_backend_blends_linearly_across_the_gap():
    anchors = {0: np.array([[0.0, 0.0]]), 10: np.array([[10.0, 0.0]])}
    filled, _ = _HoldBackend().fill(anchors, 11, _frames(11))
    np.testing.assert_allclose(filled[5, 0], [5.0, 0.0])
    np.testing.assert_allclose(filled[2, 0], [2.0, 0.0])


def test_gap_backend_seeds_missing_endpoints_from_the_spline():
    """A keypoint unlabelled on a gap endpoint is still filled, not left NaN."""
    anchors = _partial_anchors()
    filled, confidence = _HoldBackend().fill(anchors, N_FRAMES, _frames())

    for frame, points in anchors.items():
        labelled = ~np.isnan(points[:, 0])
        np.testing.assert_allclose(filled[frame][labelled], points[labelled])
        np.testing.assert_allclose(confidence[frame][labelled], 1.0)
    assert not np.any(np.isnan(filled))


def test_gap_backend_honours_frame_source_scale():
    """A downscaled frame source must not shrink the returned coordinates."""
    anchors = _anchors()
    filled, _ = _HoldBackend().fill(anchors, N_FRAMES, _ScaledFrames(_frames()))

    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)
    # Midpoint of a held track between (0, 0) and (10, 5), in source pixels.
    np.testing.assert_allclose(filled[5, 0], [5.0, 2.5])


def test_gap_backend_covers_frames_outside_the_anchored_span():
    anchors = {5: np.array([[5.0, 5.0]]), 15: np.array([[15.0, 5.0]])}
    filled, _ = _HoldBackend().fill(anchors, N_FRAMES, _frames())
    assert not np.any(np.isnan(filled))
    np.testing.assert_allclose(filled[0, 0], filled[5, 0])


def test_gap_backend_cancellation_stops_early():
    anchors = _anchors()
    filled, _ = _HoldBackend().fill(anchors, N_FRAMES, _frames(), lambda _f: False)
    # Anchors still hold even when the user cancels mid-fill.
    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)


def test_cotracker_preserves_anchors_with_a_fake_predictor():
    pytest.importorskip("torch")
    anchors = _anchors()
    backend = CoTrackerBackend(_FakePredictor(), device="cpu")
    filled, confidence = backend.fill(anchors, N_FRAMES, _frames())

    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)
        np.testing.assert_allclose(confidence[frame], 1.0)


def test_cotracker_handles_partial_anchors():
    pytest.importorskip("torch")
    anchors = _partial_anchors()
    backend = CoTrackerBackend(_FakePredictor(), device="cpu")
    filled, confidence = backend.fill(anchors, N_FRAMES, _frames())

    for frame, points in anchors.items():
        labelled = ~np.isnan(points[:, 0])
        np.testing.assert_allclose(filled[frame][labelled], points[labelled])
        np.testing.assert_allclose(confidence[frame][labelled], 1.0)
    assert not np.any(np.isnan(filled))


def test_cotracker_blends_a_held_track_across_the_gap():
    """A predictor that never moves gives the plain left/right crossfade."""
    pytest.importorskip("torch")
    anchors = {0: np.array([[0.0, 0.0]]), 10: np.array([[10.0, 0.0]])}
    backend = CoTrackerBackend(_FakePredictor(), device="cpu")
    filled, _ = backend.fill(anchors, 11, _frames(11))
    np.testing.assert_allclose(filled[5, 0], [5.0, 0.0])


def test_optical_flow_preserves_anchors():
    pytest.importorskip("cv2")
    anchors = _anchors()
    filled, confidence = OpticalFlowBackend().fill(anchors, N_FRAMES, _frames())

    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)
        np.testing.assert_allclose(confidence[frame], 1.0)


def test_cotracker_uses_the_resolved_device_by_default():
    pytest.importorskip("torch")
    assert CoTrackerBackend(_FakePredictor())._device == resolve_device()
