"""The segmentation pipeline over a pynapple session.

pynapple objects have nowhere to put per-variable attrs — a ``Tsd`` has none
at all — so a session declares its schema in a ``.ethograph/schema.yaml``
sidecar instead. These tests pin that the pipeline reads it, and that
everything the xarray backend gets from attrs the pynapple backend gets from
the sidecar: feature kinds, normalise flags, the ablation axis, video
feature ranking, and changepoints for boundary snapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ethograph.io import schema
from ethograph.labels.intervals import LABELING_MANUAL
from ethograph.labels.tsv_store import save_labels_tsv

nap = pytest.importorskip("pynapple")
torch = pytest.importorskip("torch")

FS = 50.0
N_TRIALS = 3
DURATION = 6.0
CP_TIMES = np.array([1.0, 3.5, 7.0, 13.0])  # absolute; trials start at 0, 6, 12


def _write_session(folder: Path, *, sidecar: bool = True, changepoints: bool = True) -> Path:
    """A pynapple folder session: features, labels, alignment and (optionally) a schema."""
    from ethograph.io.nwb_alignment import alignment_from_trials_ep

    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(0, N_TRIALS * DURATION, 1 / FS)
    rng = np.random.default_rng(0)
    nap.Tsd(t=t, d=np.abs(np.sin(t * 3))).save(str(folder / "speed.npz"))
    nap.Tsd(t=t, d=np.arctan2(np.sin(t), np.cos(t))).save(str(folder / "heading_angle.npz"))
    nap.TsdFrame(t=t, d=rng.normal(size=(t.size, 3)), columns=["d0", "d1", "d2"]).save(str(folder / "s3d.npz"))

    if changepoints:
        group = nap.TsGroup({0: nap.Ts(t=CP_TIMES)})
        group.set_info(source_label=["speed"], **schema.changepoint_metadata(1, target_feature="speed"))
        group.save(str(folder / "speed_troughs.npz"))

    if sidecar:
        schema.write_sidecar(
            folder,
            {
                "speed": {schema.KIND: schema.KINEMATIC_FEATURE},
                "heading_angle": {schema.KIND: schema.KINEMATIC_FEATURE, schema.NORMALISE: False},
                "s3d": {schema.KIND: schema.VIDEO_FEATURE},
            },
        )

    starts = np.arange(N_TRIALS) * DURATION
    ep = nap.IntervalSet(start=starts, end=starts + DURATION - 1 / FS)
    ep.set_info(trial=np.arange(1, N_TRIALS + 1))
    alignment_from_trials_ep(ep, folder / ".ethograph" / "alignment.nwb")

    rows = [
        {
            "trial": trial,
            "individual": "A",
            "individual_rec": "",
            "labels": 3,
            "onset_s": on,
            "offset_s": off,
            "event_type": "state",
            "confidence": 1.0,
            "labeling_method": LABELING_MANUAL,
            "changepoint_corrected": 0,
            "prediction_source": "",
            "n_samples": int(DURATION * FS),
        }
        for trial in range(1, N_TRIALS + 1)
        for on, off in ((1.0, 2.0), (3.5, 4.5))
    ]
    save_labels_tsv(folder.parent / f"{folder.name}_labels.tsv", pd.DataFrame(rows))
    return folder


def _write_config(root: Path, session: Path, **train) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    config = {
        "sessions": [{"source": str(session), "labels_path": str(session.parent / f"{session.name}_labels.tsv")}],
        "features": {
            "name": "nap",
            "individuals": ["A"],
            "columns": {"speed": {}, "heading_angle": {}, "s3d": {"s3d_columns": ["d0", "d1", "d2"]}},
            "labels": {"mapping": "mapping.txt", "branch": 0},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
        "train": {"epochs": 1, "eval_every": 1, "device": "cpu", "run_name": "nap", **train},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path):
    import ethograph as eto

    session = _write_session(tmp_path / "sessions" / "sess")
    return eto.segment.Project(_write_config(tmp_path / "project", session))


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------


def test_materialise_train_infer(project):
    from ethograph.labels.tsv_store import load_labels_tsv
    from ethograph.segment.materialise import read_index, read_layout

    data_dir = project.materialise()
    index, layout = read_index(data_dir), read_layout(data_dir)
    assert len(index) == N_TRIALS
    assert list(index["n_labelled"]) == [2] * N_TRIALS
    assert layout.names == [
        "speed",
        "heading_angle",
        "s3d|s3d_columns=d0",
        "s3d|s3d_columns=d1",
        "s3d|s3d_columns=d2",
    ]
    assert layout.fs == pytest.approx(FS)

    project.train()
    written = project.inference()
    df = load_labels_tsv(written[0])
    assert written[0].name == "sess_labels.tsv"
    assert set(df["labeling_method"]) <= {"automated"}


# ---------------------------------------------------------------------------
# What the sidecar buys
# ---------------------------------------------------------------------------


def test_the_sidecar_supplies_kinds_and_normalise(project):
    from ethograph.segment.materialise import read_layout

    layout = read_layout(project.materialise())
    assert layout.kinds == [
        schema.KINEMATIC_FEATURE,
        schema.KINEMATIC_FEATURE,
        schema.VIDEO_FEATURE,
        schema.VIDEO_FEATURE,
        schema.VIDEO_FEATURE,
    ]
    # heading_angle is an angle: declared normalise=0, so never z-scored.
    assert layout.normalise == [True, False, True, True, True]


def test_without_a_sidecar_nothing_is_declared(tmp_path: Path):
    import ethograph as eto
    from ethograph.segment.materialise import read_layout

    session = _write_session(tmp_path / "sessions" / "sess", sidecar=False)
    project = eto.segment.Project(_write_config(tmp_path / "project", session))
    layout = read_layout(project.materialise())
    assert layout.kinds == [None] * 5
    assert all(layout.normalise)


def test_ablation_drops_the_video_columns(project):
    from ethograph.segment.inference import load_run

    project.materialise()
    result = project.update("train.drop_kinds=[video_feature]", "train.run_name=able").train()
    run = load_run(result.run_dir)
    assert run.keep is not None
    assert int(run.keep.sum()) == 2  # speed + heading_angle


def test_ranking_reads_the_video_columns(project):
    project.materialise()
    ranking, names = project.rank_video_features()
    assert names == ["s3d|s3d_columns=d0", "s3d|s3d_columns=d1", "s3d|s3d_columns=d2"]
    assert ranking.n_features == 3


# ---------------------------------------------------------------------------
# Changepoints
# ---------------------------------------------------------------------------


def test_changepoint_times_come_back_on_the_trial_clock(project):
    """A TsGroup's units *are* the event times; they need shifting per trial."""
    from ethograph.segment.sessions import changepoint_times, open_session

    session = open_session(project.config.sessions[0], project.config)
    np.testing.assert_allclose(changepoint_times(session, 1, {}), [1.0, 3.5])
    np.testing.assert_allclose(changepoint_times(session, 2, {}), [1.0])  # absolute 7.0, trial starts at 6.0
    np.testing.assert_allclose(changepoint_times(session, 3, {}), [1.0])  # absolute 13.0, trial starts at 12.0


def test_no_changepoints_when_the_session_has_none(tmp_path: Path):
    import ethograph as eto
    from ethograph.segment.sessions import changepoint_times, open_session

    session_dir = _write_session(tmp_path / "sessions" / "sess", changepoints=False)
    project = eto.segment.Project(_write_config(tmp_path / "project", session_dir))
    session = open_session(project.config.sessions[0], project.config)
    assert changepoint_times(session, 1, {}).size == 0


# ---------------------------------------------------------------------------
# Failing loudly
# ---------------------------------------------------------------------------


def test_declares_schema_reports_the_sidecar(project, tmp_path: Path):
    import ethograph as eto
    from ethograph.segment.sessions import open_session

    assert open_session(project.config.sessions[0], project.config).declares_schema()

    bare = _write_session(tmp_path / "bare" / "sess", sidecar=False)
    other = eto.segment.Project(_write_config(tmp_path / "bare_project", bare))
    assert not open_session(other.config.sessions[0], other.config).declares_schema()


def test_materialise_warns_when_nothing_is_declared(tmp_path: Path, caplog):
    import ethograph as eto

    session = _write_session(tmp_path / "sessions" / "sess", sidecar=False)
    project = eto.segment.Project(_write_config(tmp_path / "project", session))
    with caplog.at_level("WARNING"):
        project.materialise()
    assert "declares no `kind`" in caplog.text


def test_ablation_without_kinds_is_refused(tmp_path: Path):
    import ethograph as eto

    session = _write_session(tmp_path / "sessions" / "sess", sidecar=False)
    project = eto.segment.Project(_write_config(tmp_path / "project", session))
    project.materialise()
    project.update("train.drop_kinds=[video_feature]")
    with pytest.raises(ValueError, match="no column .* declares a kind"):
        project.train()
