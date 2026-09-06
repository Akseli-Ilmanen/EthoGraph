"""Video-feature ranking dialog: what it offers, what it runs over, what it pastes.

The ranking maths lives in ``ethograph/video_features/select.py``; every test
here but :class:`TestRealRanking` monkeypatches :func:`rank_features` away, so
nothing depends on S3D data or on that module's numerics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from qtpy.QtWidgets import QApplication

import ethograph as eto
from ethograph.gui import dialog_video_feature_rank as dvfr
from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_video_feature_rank import VideoFeatureRankDialog
from ethograph.io.catalog import XarrayLoader
from ethograph.io.schema import KIND, VIDEO_FEATURE
from ethograph.labels.intervals import LABELING_AUTOMATED, LABELING_CURATED

N_FRAMES = 300
N_DIMS = 64
FS = 30.0

MAPPINGS = {
    1: {"name": "groom", "event_type": "state", "branch": 0},
    2: {"name": "rear", "event_type": "state", "branch": 0},
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Meta:
    def __init__(self, app_state):
        self.app_state = app_state


def _dataset(declare_kinds: bool) -> xr.Dataset:
    """A session with a wide video-feature bank beside a narrow kinematic one."""
    rng = np.random.default_rng(0)
    time = np.arange(N_FRAMES) / FS
    s3d = rng.standard_normal((N_FRAMES, N_DIMS))
    s3d[100:200, 7] += 6.0  # dimension 7 answers to whatever happens mid-trial
    ds = xr.Dataset(
        {
            "s3d": (("time", "s3d_dim"), s3d),
            "speed": (("time", "keypoints"), rng.standard_normal((N_FRAMES, 2))),
        },
        coords={
            "time": time,
            "s3d_dim": [str(i) for i in range(N_DIMS)],
            "keypoints": ["head", "tail"],
        },
    )
    if declare_kinds:
        ds["s3d"].attrs[KIND] = VIDEO_FEATURE
        ds["speed"].attrs[KIND] = "kinematic_feature"
    return ds


def _labels(trials: list[int], *, method: str = LABELING_CURATED, background_only: bool = False) -> pd.DataFrame:
    rows = []
    for trial in trials:
        if background_only:
            continue
        rows += [
            {
                "trial": trial,
                "labels": 1,
                "onset_s": 100 / FS,
                "offset_s": 149 / FS,
                "individual": "mouse",
                "individual_rec": "",
                "event_type": "state",
                "labeling_method": method,
                "confidence": 1.0,
            },
            {
                "trial": trial,
                "labels": 2,
                "onset_s": 200 / FS,
                "offset_s": 249 / FS,
                "individual": "mouse",
                "individual_rec": "",
                "event_type": "state",
                "labeling_method": method,
                "confidence": 1.0,
            },
        ]
    columns = [
        "trial",
        "labels",
        "onset_s",
        "offset_s",
        "individual",
        "individual_rec",
        "event_type",
        "labeling_method",
        "confidence",
    ]
    return pd.DataFrame(rows, columns=columns)


def _app_state(tmp_path, ds: xr.Dataset) -> ObservableAppState:
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state.ds = ds
    state.dt = eto.TrialTree.from_datasets([ds.assign_attrs(trial=trial) for trial in (1, 2)], validate=False)
    state.data_loader = XarrayLoader(ds)
    state.trials = [1, 2]
    state._label_mappings = dict(MAPPINGS)
    state._all_labels_df = _labels([1, 2])
    return state


class _FakeRanking:
    """Stand-in for ``FeatureRanking`` — only what the dialog reads."""

    def __init__(self, n_features: int = N_DIMS, n_classes: int = 2):
        rng = np.random.default_rng(1)
        self.scores = rng.random(n_features)
        self.per_class = rng.random((n_features, n_classes))
        self.class_ids = np.arange(1, n_classes + 1, dtype=np.int64)
        self.n_trials = 2
        self.n_features = n_features
        self.saved_to: str | None = None

    def top(self, k: int) -> np.ndarray:
        return np.argsort(-self.scores, kind="stable")[:k]

    def save(self, path):
        self.saved_to = str(path)
        return path


@pytest.fixture()
def fake_rank(monkeypatch):
    """``rank_features`` replaced by a recorder returning a fixed ranking."""
    calls: list[list[tuple[np.ndarray, np.ndarray]]] = []
    ranking = _FakeRanking()

    def _rank(trials, **kwargs):
        collected = list(trials)
        calls.append(collected)
        if not collected:
            raise ValueError("rank_features needs at least one trial, got none.")
        return ranking

    monkeypatch.setattr(dvfr, "rank_features", _rank)
    return calls, ranking


@pytest.fixture()
def dialog(qapp, tmp_path):
    state = _app_state(tmp_path, _dataset(declare_kinds=True))
    dlg = VideoFeatureRankDialog(_Meta(state))
    yield dlg
    dlg.close()


def _features(dlg) -> list[str]:
    return [dlg.feature_combo.itemText(i) for i in range(dlg.feature_combo.count())]


class TestFeatureChoices:
    def test_only_video_features_when_kinds_are_declared(self, dialog):
        assert _features(dialog) == ["s3d"]

    def test_falls_back_to_wide_dims_when_nothing_declares_a_kind(self, qapp, tmp_path):
        state = _app_state(tmp_path, _dataset(declare_kinds=False))
        dlg = VideoFeatureRankDialog(_Meta(state))
        # "speed" has 2 keypoints, well under WIDE_DIM; "s3d" has 64.
        assert _features(dlg) == ["s3d"]
        dlg.close()

    def test_no_candidates_disables_run_and_says_why(self, qapp, tmp_path):
        ds = _dataset(declare_kinds=False)[["speed"]]
        state = _app_state(tmp_path, ds)
        dlg = VideoFeatureRankDialog(_Meta(state))
        assert _features(dlg) == []
        assert not dlg.run_btn.isEnabled()
        assert "no video features" in dlg.status_label.text()
        dlg.close()


class TestTrialsScope:
    """The dialog has no trial filter of its own: the trials table decides."""

    def test_note_reports_the_trials_table(self, dialog):
        assert dialog.trials_note.text().startswith("Runs over the 2 trial(s) the trials table")

    def test_note_follows_the_filter(self, dialog):
        dialog.app_state.trials = [1]
        assert dialog.trials_note.text().startswith("Runs over the 1 trial(s)")
        dialog.app_state.trials = [1, 2, 3]
        assert dialog.trials_note.text().startswith("Runs over the 3 trial(s)")

    def test_run_covers_exactly_the_visible_trials(self, dialog, fake_rank):
        calls, _ = fake_rank
        dialog.app_state.trials = [2]
        dialog._run()
        assert len(calls[-1]) == 1


class TestRun:
    def test_run_fills_the_heatmap_and_the_paste_string(self, dialog, fake_rank):
        _, ranking = fake_rank
        dialog._run()
        assert dialog._ranking is ranking
        assert dialog.image_item.image is not None
        assert dialog.image_item.image.shape == (dialog.topk_spin.value(), 2)
        assert dialog.copy_btn.isEnabled()
        assert dialog.save_btn.isEnabled()

    def test_yaml_matches_the_ranking_top_k(self, dialog, fake_rank):
        _, ranking = fake_rank
        dialog._run()
        expected = [int(i) for i in ranking.top(dialog.topk_spin.value())]
        assert dialog.yaml_text() == f"s3d: {{s3d_dim: [{', '.join(str(i) for i in expected)}]}}"
        assert dialog.yaml_edit.text() == dialog.yaml_text()

    def test_automated_labels_are_not_ranked_against(self, dialog, fake_rank):
        calls, _ = fake_rank
        dialog.app_state._all_labels_df = _labels([1, 2], method=LABELING_AUTOMATED)
        dialog._run()
        assert calls == []  # never reached rank_features: nothing curated to use
        assert dialog.status_label.text().startswith(dvfr.NOTHING_TO_RANK)

    def test_dense_labels_carry_both_classes(self, dialog, fake_rank):
        calls, _ = fake_rank
        dialog._run()
        _, labels = calls[-1][0]
        assert set(np.unique(labels)) == {0, 1, 2}


class TestTopK:
    def test_spin_rerenders_without_recomputing(self, dialog, fake_rank):
        calls, ranking = fake_rank
        dialog._run()
        assert len(calls) == 1

        dialog.topk_spin.setValue(5)
        assert dialog._ranking is ranking  # same object: no second computation
        assert len(calls) == 1
        assert dialog.image_item.image.shape == (5, 2)
        assert dialog.yaml_text() == "s3d: {s3d_dim: [" + ", ".join(str(int(i)) for i in ranking.top(5)) + "]}"


class TestNothingToRank:
    def test_all_background_session_reports_instead_of_raising(self, dialog, fake_rank):
        calls, _ = fake_rank
        dialog.app_state._all_labels_df = _labels([1, 2], background_only=True)
        dialog._run()
        assert dialog.status_label.text().startswith(dvfr.NOTHING_TO_RANK)
        assert dialog._ranking is None
        assert not dialog.copy_btn.isEnabled()
        assert calls == []

    def test_ranking_error_is_reported_not_raised(self, dialog, monkeypatch):
        def _boom(trials, **kwargs):
            list(trials)
            raise ValueError("No trial has two classes to contrast")

        monkeypatch.setattr(dvfr, "rank_features", _boom)
        dialog._run()
        assert dialog.status_label.text().startswith(dvfr.NOTHING_TO_RANK)
        assert dialog._ranking is None


class TestSave:
    def test_save_goes_through_the_browse_helper(self, dialog, fake_rank, monkeypatch, tmp_path):
        _, ranking = fake_rank
        dialog._run()
        target = tmp_path / "ranking.npz"
        seen: dict[str, object] = {}

        def _browse(parent, app_state, caption, default_name, file_filter=None, preferred_dir=None):
            seen["default_name"] = default_name
            return str(target)

        monkeypatch.setattr(dvfr, "browse_save_file", _browse)
        dialog._save()
        assert seen["default_name"] == "s3d_cohens_d.npz"
        assert ranking.saved_to == str(target)

    def test_cancelled_save_writes_nothing(self, dialog, fake_rank, monkeypatch):
        _, ranking = fake_rank
        dialog._run()
        monkeypatch.setattr(dvfr, "browse_save_file", lambda *a, **k: "")
        dialog._save()
        assert ranking.saved_to is None


class TestRealRanking:
    """The one test that uses the real ranking module, when it is present."""

    def test_the_dialog_ranks_the_planted_dimension_first(self, qapp, tmp_path):
        pytest.importorskip("ethograph.video_features.select")
        state = _app_state(tmp_path, _dataset(declare_kinds=True))
        dlg = VideoFeatureRankDialog(_Meta(state))
        dlg._run()
        assert dlg._ranking is not None
        assert dlg.top_indices()[0] == 7  # the dimension planted on class 1
        dlg.close()
