"""Test-time refinement: window selection, the delta, and the fit loop.

The tracker itself is faked — a two-level miniature whose predictions depend on
the query features, which is all the refinement actually touches. That exercises
level mapping, target construction, cancellation and persistence without weights
or a GPU, and keeps the real assertion in view: **anchors survive the fit.**
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cotracker")

import torch  # noqa: E402

from ethograph.gui import pose_fill  # noqa: E402
from ethograph.gui.pose_fill import POSEPAL_BACKEND, available_backends, build_backend  # noqa: E402
from ethograph.gui.pose_refine import (  # noqa: E402
    PosePALBackend,
    QueryFeatureRefinement,
    RefinementConfig,
    training_windows,
)

N_FRAMES = 24
N_POINTS = 2
CHANNELS = 4
SUPPORT = 9
LEVELS = 2


def _anchors() -> dict[int, np.ndarray]:
    return {
        0: np.array([[4.0, 4.0], [8.0, 8.0]]),
        8: np.array([[6.0, 5.0], [10.0, 9.0]]),
        16: np.array([[8.0, 6.0], [12.0, 10.0]]),
    }


class _Frames:
    scale = 1.0

    def __init__(self, n: int = N_FRAMES, size: int = 32):
        rng = np.random.default_rng(0)
        self._data = rng.integers(0, 255, size=(n, size, size, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]


class _FakeModel(torch.nn.Module):
    """A tracker in miniature: predictions offset by the query features.

    Only two things matter for the code under test — ``get_track_feat`` is the
    single place features enter, and the predictions must depend on them, or a
    fit would have no gradient to follow.
    """

    def __init__(self):
        super().__init__()
        self.fnet = torch.nn.Conv2d(3, CHANNELS, 3, padding=1)
        self.calls = 0

    def get_track_feat(self, fmaps, queried_frames, queried_coords, support_radius=0):
        self.calls += 1
        # A real feature depends on *when* and *where* it was sampled, which is
        # what the refinement averages over the labelled frames. A constant would
        # hide a query built from the wrong frame.
        sampled = queried_frames.float() + queried_coords.sum(-1)
        support = sampled[:, None, :, None].expand(-1, SUPPORT, -1, CHANNELS).clone()
        return support[:, SUPPORT // 2][:, None], support

    def forward(self, video, queries, iters=4, is_train=False):
        frames = video.shape[1]
        fmaps = self.fnet(video[0])
        coords = queries[:, :, 1:][:, None].repeat(1, frames, 1, 1)
        for level in range(LEVELS):
            # Distinct spatial shapes per level: that is how the refinement
            # identifies which delta belongs to which call.
            level_maps = fmaps[:, :, :: 2**level, :: 2**level][None]
            _, support = self.get_track_feat(level_maps, queries[:, :, 0], queries[:, :, 1:])
            coords = coords + support.mean(dim=(1, 3))[:, None, :, None]
        predictions = [[coords] * 2]
        visibility = [[torch.ones(coords.shape[:-1])] * 2]
        train_data = (predictions, visibility, visibility, torch.ones(coords.shape[:-1]))
        return coords, visibility[0][-1], visibility[0][-1], train_data if is_train else None


class _FakePredictor(torch.nn.Module):
    interp_shape = (32, 32)

    def __init__(self):
        super().__init__()
        self.model = _FakeModel()

    def forward(self, video, queries=None, backward_tracking=False):
        frames = video.shape[1]
        points = queries[0, :, 1:]
        return points[None, None].repeat(1, frames, 1, 1), torch.ones((1, frames, points.shape[0]))


def _refinement(**config) -> QueryFeatureRefinement:
    settings = {"steps": 6, "steps_per_window": 2, "window_frames": 12, **config}
    return QueryFeatureRefinement(_FakePredictor(), N_POINTS, device="cpu", config=RefinementConfig(**settings))


# ----------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------


def test_windows_cover_every_labelled_frame():
    windows = training_windows([0, 8, 16], 24, 12)
    assert windows
    for frame in (0, 8, 16):
        assert any(start <= frame < stop for start, stop in windows)


def test_windows_skip_anchors_with_no_reachable_neighbour():
    """One labelled frame in a window means one supervised frame — the query."""
    assert training_windows([0, 500], 1000, 12) == []


def test_windows_are_clamped_to_the_video():
    for start, stop in training_windows([0, 5, 22], 24, 12):
        assert 0 <= start < stop <= 24


def test_windows_are_deduplicated():
    windows = training_windows([20, 21, 22], 24, 12)
    assert len(windows) == len(set(windows))


# ----------------------------------------------------------------------
# The learned features
# ----------------------------------------------------------------------


def test_unfitted_refinement_is_a_no_op():
    refinement = _refinement()
    assert refinement.fitted is False
    with refinement.applied():
        pass  # nothing patched, nothing to restore


def test_fit_moves_the_features_and_preserves_the_model():
    refinement = _refinement()
    model = refinement._model

    assert refinement.fit(_anchors(), N_FRAMES, _Frames(), "sig") is True

    assert refinement.fitted
    assert refinement._features.shape == (LEVELS, SUPPORT, N_POINTS, CHANNELS)
    assert not torch.allclose(refinement._features.detach(), refinement._initial)
    # The patches are context managers, not installations: nothing shadows the
    # model's own methods once a fit is over.
    assert "get_track_feat" not in model.__dict__
    assert isinstance(model.fnet, torch.nn.Conv2d)


def test_features_start_at_the_mean_over_the_labelled_frames():
    """PosePAL's ``get_kp_feats``: the template is the average of the labels.

    Interp shape equals the frame size here, so model pixels are source pixels
    and the fake's feature is ``frame_within_window + x + y``. The windows for
    labels at 0/8/16 are (0, 12), (8, 20), (12, 24), and each ``(point, frame)``
    pair is counted once, from the first window holding it: frames 0 and 8 from
    window (0, 12), frame 16 from window (8, 20).
    """
    refinement = _refinement(steps=0)
    anchors = _anchors()
    assert refinement.fit(anchors, N_FRAMES, _Frames(), "sig") is True

    for point in range(N_POINTS):
        expected = np.mean([frame - start + anchors[frame][point].sum() for frame, start in ((0, 0), (8, 0), (16, 8))])
        assert refinement._features[0, :, point].detach().numpy() == pytest.approx(expected)
    assert bool(refinement._learned.all())


def test_features_are_scoped_to_the_rows_being_tracked():
    """A gap tracks only the seeded rows, so query ``i`` is not point ``i``."""
    refinement = _refinement()
    refinement.fit(_anchors(), N_FRAMES, _Frames(), "sig")

    with refinement.applied():
        refinement.set_rows(np.array([N_POINTS - 1]))
        support = _track_feat(refinement, 1)

    torch.testing.assert_close(support[0, :, 0], refinement._features[0, :, N_POINTS - 1].detach())


def test_a_point_labelled_nowhere_keeps_the_models_own_feature():
    """Nothing was learned for it, so there is nothing to substitute."""
    anchors = {frame: points.copy() for frame, points in _anchors().items()}
    for points in anchors.values():
        points[1] = np.nan

    refinement = _refinement()
    refinement.fit(anchors, N_FRAMES, _Frames(), "sig")

    assert refinement._learned.tolist() == [True, False]
    with refinement.applied():
        support = _track_feat(refinement, N_POINTS)
    # `_track_feat` samples at frame 0, coordinate 0, where the fake's own
    # feature is exactly zero.
    assert torch.all(support[:, :, 1] == 0)
    assert torch.any(support[:, :, 0] != 0)


def test_fit_records_what_it_was_fitted_on():
    refinement = _refinement()
    refinement.fit(_anchors(), N_FRAMES, _Frames(), "sig")
    assert refinement.n_anchor_frames == len(_anchors())
    assert refinement.matches("sig")
    assert not refinement.matches("other")


def _track_feat(refinement: QueryFeatureRefinement, n: int):
    """Sample the top pyramid level — the resolution the fit registered."""
    size = refinement._predictor.interp_shape[0]
    fmaps = torch.zeros((1, 1, CHANNELS, size, size))
    return refinement._model.get_track_feat(fmaps, torch.zeros((1, n)), torch.zeros((1, n, 2)))[1]


def test_applied_changes_what_the_model_returns():
    refinement = _refinement()
    refinement.fit(_anchors(), N_FRAMES, _Frames(), "sig")

    plain = _track_feat(refinement, N_POINTS)
    with refinement.applied():
        refined = _track_feat(refinement, N_POINTS)
    assert not torch.allclose(plain, refined)


def test_the_support_grid_is_never_given_a_learned_feature():
    """Trailing rows belong to the predictor's own context points."""
    refinement = _refinement()
    refinement.fit(_anchors(), N_FRAMES, _Frames(), "sig")
    with refinement.applied():
        support = _track_feat(refinement, N_POINTS + 5)
    assert torch.all(support[:, :, N_POINTS:] == 0)
    assert torch.any(support[:, :, :N_POINTS] != 0)


def test_fit_needs_two_labelled_frames_within_a_window():
    with pytest.raises(ValueError):
        _refinement().fit({0: _anchors()[0]}, N_FRAMES, _Frames(), "sig")


def test_cancelling_a_fit_leaves_the_previous_one_intact():
    refinement = _refinement()
    refinement.fit(_anchors(), N_FRAMES, _Frames(), "first")
    before = refinement._features.detach().clone()

    assert refinement.fit(_anchors(), N_FRAMES, _Frames(), "second", lambda _f: False) is False

    assert refinement.signature == "first"
    torch.testing.assert_close(refinement._features, before)


def test_cancelling_the_first_fit_leaves_nothing_fitted():
    refinement = _refinement()
    assert refinement.fit(_anchors(), N_FRAMES, _Frames(), "sig", lambda _f: False) is False
    assert refinement.fitted is False


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_saved_refinement_round_trips(tmp_path):
    refinement = _refinement()
    refinement.fit(_anchors(), N_FRAMES, _Frames(), "sig")
    path = tmp_path / "video.mp4.posepal.pt"
    refinement.save(path)

    restored = _refinement()
    assert restored.load(path, "sig") is True
    torch.testing.assert_close(restored._features, refinement._features)
    torch.testing.assert_close(restored._learned, refinement._learned)
    assert restored.n_anchor_frames == refinement.n_anchor_frames


def test_a_sidecar_written_before_the_features_were_absolute_is_a_miss(tmp_path):
    """It holds a residual, which means nothing against a learned feature."""
    path = tmp_path / "video.mp4.posepal.pt"
    torch.save(
        {"delta": torch.zeros((LEVELS, SUPPORT, N_POINTS, CHANNELS)), "signature": "sig", "n_points": N_POINTS},
        str(path),
    )
    assert _refinement().load(path, "sig") is False


def test_a_fit_made_from_other_labels_is_refused(tmp_path):
    refinement = _refinement()
    refinement.fit(_anchors(), N_FRAMES, _Frames(), "sig")
    path = tmp_path / "video.mp4.posepal.pt"
    refinement.save(path)

    assert _refinement().load(path, "labels-have-changed") is False


def test_saving_without_a_fit_raises(tmp_path):
    with pytest.raises(ValueError):
        _refinement().save(tmp_path / "nothing.pt")


# ----------------------------------------------------------------------
# The backend
# ----------------------------------------------------------------------


def _backend(refinement: QueryFeatureRefinement) -> PosePALBackend:
    backend = PosePALBackend(refinement._predictor, refinement, device="cpu")
    backend.signature = "sig"
    return backend


def test_refined_backend_preserves_anchors():
    refinement = _refinement()
    backend = _backend(refinement)
    anchors = _anchors()

    filled, confidence = backend.fill(anchors, N_FRAMES, _Frames())

    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)
        np.testing.assert_allclose(confidence[frame], 1.0)


def test_filling_fits_once_and_then_reuses_the_fit():
    refinement = _refinement()
    backend = _backend(refinement)
    stages: list[str] = []
    backend.on_stage = stages.append

    backend.fill(_anchors(), N_FRAMES, _Frames())
    assert stages[0].startswith("Learning")
    features = refinement._features.detach().clone()

    stages.clear()
    backend.fill(_anchors(), N_FRAMES, _Frames())
    assert not any(stage.startswith("Learning") for stage in stages)
    torch.testing.assert_close(refinement._features, features)


def test_new_labels_make_the_fill_refit():
    refinement = _refinement()
    backend = _backend(refinement)
    backend.fill(_anchors(), N_FRAMES, _Frames())

    backend.signature = "the user labelled another frame"
    stages: list[str] = []
    backend.on_stage = stages.append
    backend.fill(_anchors(), N_FRAMES, _Frames())
    assert any(stage.startswith("Learning") for stage in stages)


def test_cancelling_the_fit_still_returns_a_usable_fill():
    """Same contract as cancelling any other fill: the anchors survive."""
    backend = _backend(_refinement())
    anchors = _anchors()

    filled, _ = backend.fill(anchors, N_FRAMES, _Frames(), lambda _f: False)

    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)
    # The labelled span is filled; past the last label there is nothing to fill.
    assert not np.any(np.isnan(filled[: max(anchors) + 1]))


def test_refined_backend_needs_video():
    with pytest.raises(ValueError):
        _backend(_refinement()).fill(_anchors(), N_FRAMES, None)


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------


def test_refinement_is_offered_but_only_with_a_gpu(monkeypatch):
    monkeypatch.setattr(pose_fill, "_module_available", lambda name: True)
    monkeypatch.setattr(pose_fill, "resolve_device", lambda preferred=None: "cpu")
    info = {i.key: i for i in available_backends()}[POSEPAL_BACKEND]
    assert info.available is False
    assert "GPU" in info.hint

    monkeypatch.setattr(pose_fill, "resolve_device", lambda preferred=None: "cuda")
    info = {i.key: i for i in available_backends()}[POSEPAL_BACKEND]
    assert info.available is True


def test_building_the_refined_backend_needs_a_point_count():
    with pytest.raises(ValueError):
        build_backend(POSEPAL_BACKEND)
