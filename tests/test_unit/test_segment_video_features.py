"""S3D video features: folder discovery, the sidecar loop, and the merge.

S3D itself is never run here — ``extract_s3d`` is replaced by a stub, so these
tests need no weights, no GPU and no video decoding. What they pin is the
plumbing around it: which files are found, which are skipped, and — the one
piece of real arithmetic — that a sidecar on the *video's* clock lands on the
trial clock the right way round.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

import ethograph as eto
from ethograph.segment.config import VideoFeaturesConfig, load_config
from ethograph.segment.video_features import (
    S3D_TIME_DIM,
    _with_s3d,
    extract_videos,
    iter_video_files,
    sidecar_path,
)

FEATURE_DIM = "s3d_dims"


def _touch_video(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"not really a video")
    return path


def _fake_features(n: int, dims: int = 4, fps: float = 100.0, value: float | None = None) -> xr.DataArray:
    """A sidecar-shaped DataArray on the video clock; column 0 counts frames."""
    data = np.zeros((n, dims), dtype=np.float32)
    data[:, 0] = np.arange(n) if value is None else value
    return xr.DataArray(
        data,
        dims=(S3D_TIME_DIM, FEATURE_DIM),
        coords={S3D_TIME_DIM: np.arange(n) / fps, FEATURE_DIM: np.arange(dims)},
        name="s3d",
        attrs={"time_basis": "video", "video_fps": fps},
    )


# ---------------------------------------------------------------------------
# Finding videos
# ---------------------------------------------------------------------------


def test_iter_video_files_walks_a_folder_recursively(tmp_path: Path):
    a = _touch_video(tmp_path / "vids", "b.mp4")
    b = _touch_video(tmp_path / "vids", "a.avi")
    c = _touch_video(tmp_path / "vids" / "sub", "c.mov")
    _touch_video(tmp_path / "vids", "notes.txt")
    assert list(iter_video_files([tmp_path / "vids"])) == sorted([a.resolve(), b.resolve(), c.resolve()])


def test_iter_video_files_accepts_files_and_globs_and_dedupes(tmp_path: Path):
    a = _touch_video(tmp_path, "a.mp4")
    b = _touch_video(tmp_path, "b.mp4")
    found = list(iter_video_files([a, str(tmp_path / "*.mp4"), a]))
    assert found == [a.resolve(), b.resolve()]


def test_iter_video_files_refuses_non_videos_and_empty_folders(tmp_path: Path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    with pytest.raises(ValueError, match="is not a video"):
        list(iter_video_files([notes]))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No video files under"):
        list(iter_video_files([tmp_path / "empty"]))
    with pytest.raises(FileNotFoundError):
        list(iter_video_files([tmp_path / "nope.mp4"]))


# ---------------------------------------------------------------------------
# The extraction loop
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_s3d(monkeypatch):
    """Replace S3D extraction + probing; record the config each call saw."""
    calls: list[tuple[Path, object]] = []

    def fake_extract(video, cfg, device=None, progress=None):
        calls.append((Path(video), cfg))
        return _fake_features(50)

    class _Probe:
        fps = 200.0

    monkeypatch.setattr("ethograph.video_features.extract_s3d", fake_extract, raising=False)
    monkeypatch.setattr("ethograph.video_features.frames.probe_video", lambda path: _Probe(), raising=False)
    return calls


def test_extract_videos_writes_one_sidecar_per_video(tmp_path: Path, stub_s3d):
    videos = tmp_path / "videos"
    _touch_video(videos, "trial1.mp4")
    _touch_video(videos, "trial2.mp4")
    out = tmp_path / "features"

    written = extract_videos([videos], out, VideoFeaturesConfig(stack_s=0.2))
    assert [p.name for p in written] == ["trial1_s3d.nc", "trial2_s3d.nc"]
    assert xr.load_dataarray(out / "trial1_s3d.nc").dims == (S3D_TIME_DIM, FEATURE_DIM)
    assert {cfg.stack_s for _, cfg in stub_s3d} == {0.2}


def test_extract_videos_skips_existing_unless_overwrite(tmp_path: Path, stub_s3d):
    videos = tmp_path / "videos"
    _touch_video(videos, "trial1.mp4")
    out = tmp_path / "features"

    assert len(extract_videos([videos], out, overwrite=False)) == 1
    assert extract_videos([videos], out, overwrite=False) == []
    assert len(stub_s3d) == 1
    assert len(extract_videos([videos], out, overwrite=True)) == 1
    assert len(stub_s3d) == 2


def test_sidecar_path_is_stem_based(tmp_path: Path):
    assert sidecar_path(Path("/a/b/cam1_trial7.mp4"), tmp_path).name == "cam1_trial7_s3d.nc"


def test_too_short_a_window_names_the_shortest_that_works(tmp_path: Path, stub_s3d, monkeypatch):
    """0.1 s is 3 frames at 30 fps — refused, with the shortest that works named."""

    class _Slow:
        fps = 30.0

    monkeypatch.setattr("ethograph.video_features.frames.probe_video", lambda path: _Slow(), raising=False)
    _touch_video(tmp_path / "videos", "slow.mp4")
    with pytest.raises(ValueError, match="S3D needs at least 13"):
        extract_videos([tmp_path / "videos"], tmp_path / "out", VideoFeaturesConfig(stack_s=0.1))
    assert stub_s3d == []


def test_the_default_window_survives_a_slow_camera(tmp_path: Path, stub_s3d, monkeypatch):
    """0.5 s is 15 frames at 30 fps — which is why it is the default."""

    class _Slow:
        fps = 30.0

    monkeypatch.setattr("ethograph.video_features.frames.probe_video", lambda path: _Slow(), raising=False)
    _touch_video(tmp_path / "videos", "slow.mp4")
    assert len(extract_videos([tmp_path / "videos"], tmp_path / "out")) == 1


# ---------------------------------------------------------------------------
# Merging onto the trial clock
# ---------------------------------------------------------------------------


class _Alignment:
    """Alignment stub: sample 0 of the video sits at this trial time."""

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def stream_offset_for_trial(self, trial, stream, device=None) -> float:
        return self.offset


def _trial_ds(fps: float = 100.0, n: int = 30) -> xr.Dataset:
    time = np.arange(n) / fps
    return xr.Dataset(
        {"speed": (("time", "individual"), np.ones((n, 1)))},
        coords={"time": time, "individual": ["A"]},
        attrs={"trial": 1},
    )


def test_merge_samples_video_clock_onto_trial_clock(tmp_path: Path):
    """A video starting 0.10 s into the trial must be read 0.10 s *earlier*.

    Column 0 of the sidecar is the video's frame index, so the merged value at
    trial time t must be ``(t - offset) * fps`` — the same direction as
    ``VideoSync.frame_to_time`` (trial = video + offset).
    """
    sidecar = tmp_path / "v_s3d.nc"
    _fake_features(60, fps=100.0).to_netcdf(sidecar)

    merged = _with_s3d(_trial_ds(), sidecar, _Alignment(0.10), trial=1)
    frame_index = merged["s3d"].isel({FEATURE_DIM: 0}).values
    # trial t=0.20 s → video time 0.10 s → video frame 10
    assert frame_index[20] == pytest.approx(10.0)
    assert frame_index[10] == pytest.approx(0.0)
    assert merged["s3d"].dims == ("time", FEATURE_DIM)
    np.testing.assert_allclose(merged["time"].values, _trial_ds()["time"].values)


def test_merge_with_zero_offset_is_the_identity_mapping(tmp_path: Path):
    sidecar = tmp_path / "v_s3d.nc"
    _fake_features(60, fps=100.0).to_netcdf(sidecar)
    merged = _with_s3d(_trial_ds(), sidecar, _Alignment(0.0), trial=1)
    np.testing.assert_allclose(merged["s3d"].isel({FEATURE_DIM: 0}).values, np.arange(30), atol=1e-6)


def test_merged_feature_is_selectable_by_the_pipeline(tmp_path: Path):
    """After merging, `s3d` is an ordinary feature the column layout can pin."""
    from ethograph.features.columns import extract_features
    from ethograph.io.catalog import XarrayLoader

    sidecar = tmp_path / "v_s3d.nc"
    _fake_features(60, dims=3, fps=100.0).to_netcdf(sidecar)
    ds = _with_s3d(_trial_ds(), sidecar, _Alignment(0.0), trial=1)

    time, data = extract_features(XarrayLoader(ds), {"s3d": {FEATURE_DIM: ["0", "1"]}})
    assert data.shape == (30, 2)
    assert len(time) == 30


def test_merge_video_features_writes_a_sibling_not_the_source(tmp_path: Path, monkeypatch):
    """A session file is never overwritten unless --in-place says so."""
    from ethograph.segment import video_features as vf

    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    source = session_dir / "s1.nc"
    eto.from_datasets([_trial_ds()]).save(str(source))
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _fake_features(60, fps=100.0).to_netcdf(features_dir / "v_s3d.nc")

    config = load_config(_write_config(tmp_path, source))
    session = _FakeSession(source, eto.open(str(source)), _Alignment(0.0))
    monkeypatch.setattr(vf, "session_videos", lambda s, c: {1: Path("v.mp4")})

    written = vf.merge_video_features(session, config, features_dir=features_dir)
    assert written == session_dir / "s1_s3d.nc"
    assert "s3d" in eto.open(str(written)).trial(1).data_vars
    assert "s3d" not in eto.open(str(source)).trial(1).data_vars


class _FakeSession:
    """Enough of a Session for merge_video_features."""

    def __init__(self, source: Path, dt, alignment) -> None:
        self.source = source
        self.id = "fake"
        self.result = type("R", (), {"dt": dt, "nwb_alignment": alignment})()


def _write_config(root: Path, source: Path) -> Path:
    (root / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    labels_path = source.with_name(f"{source.stem}_labels.tsv")
    data = {
        "sessions": [{"source": str(source), "labels_path": str(labels_path)}],
        "features": {"columns": {"speed": {}}, "labels": {"mapping": "mapping.txt"}},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_video_features_config_reaches_s3d(tmp_path: Path):
    path = _write_config(tmp_path, tmp_path / "s1.nc")
    overrides = ["video_features.stack_s=0.3", "video_features.analysis_fps=25"]
    cfg = load_config(path, overrides)
    s3d = cfg.video_features.s3d_config()
    assert (s3d.stack_s, s3d.analysis_fps) == (0.3, 25)
    assert cfg.video_features_dir == tmp_path / "video_features"


def test_video_features_config_is_only_the_choices_that_matter(tmp_path: Path):
    """Performance knobs are not project settings — a typo must not pass silently."""
    path = _write_config(tmp_path, tmp_path / "s1.nc")
    for gone in ("mode", "precision", "device", "batch", "truncate_at"):
        with pytest.raises(ValueError, match="unknown key"):
            load_config(path, [f"video_features.{gone}=x"])
    assert load_config(path).video_features.stack_s == 0.5


def test_changepoint_times_read_the_current_marker(tmp_path: Path, monkeypatch):
    """Post-processing must find changepoints on a *migrated* file, not a legacy one."""
    from ethograph.io import schema
    from ethograph.segment.sessions import changepoint_times

    n, fps = 30, 100.0
    time = np.arange(n) / fps
    speed = xr.DataArray(np.ones(n), dims="time", coords={"time": time})
    cp = xr.zeros_like(speed, dtype=np.int8)
    cp[[5, 11]] = 1
    cp.attrs = schema.changepoint_attrs(target_feature="speed")
    ds = xr.Dataset({"speed": speed, "speed_troughs": cp})

    session = _FakeSession(tmp_path / "s.nc", None, _Alignment(0.0))
    monkeypatch.setattr(type(session), "trial_dataset", lambda self, trial: ds, raising=False)
    found = changepoint_times(session, 1, {})
    np.testing.assert_allclose(found, [0.05, 0.11])

    # A file still carrying only the legacy marker has no changepoints now.
    legacy = ds.copy(deep=True)
    legacy["speed_troughs"].attrs = {"type": "changepoints", "target_feature": "speed"}
    monkeypatch.setattr(type(session), "trial_dataset", lambda self, trial: legacy, raising=False)
    assert changepoint_times(session, 1, {}).size == 0
    # ...until it is migrated.
    schema.migrate_legacy_attrs(legacy)
    assert changepoint_times(session, 1, {}).size == 2


def test_extract_videos_helper_uses_the_project_default(tmp_path: Path, stub_s3d):
    """The scripted entry point and a project must agree on the window."""
    import ethograph as eto

    _touch_video(tmp_path / "videos", "a.mp4")
    eto.segment.extract_videos([tmp_path / "videos"], tmp_path / "out")
    assert [cfg.stack_s for _, cfg in stub_s3d] == [VideoFeaturesConfig().stack_s]


# ---------------------------------------------------------------------------
# Narrowing which videos are extracted
# ---------------------------------------------------------------------------


class TestIncludeFilter:
    """Two cameras seeing the same thing is an hour of S3D wasted on one of them."""

    def _two_cameras(self, tmp_path: Path) -> Path:
        root = tmp_path / "videos"
        _touch_video(root / "cam-1", "trial003.mp4")
        _touch_video(root / "cam-2", "trial003.mp4")
        _touch_video(root, "trial004_cam-1.mp4")
        _touch_video(root, "trial004_cam-2.mp4")
        return root

    def test_matches_a_camera_folder_or_a_camera_in_the_name(self, tmp_path: Path):
        """The path is searched whole, so either layout works."""
        root = self._two_cameras(tmp_path)
        kept = [str(p) for p in iter_video_files([root], ["cam-1"])]
        assert len(kept) == 2
        assert all("cam-1" in p for p in kept)

    def test_a_plain_substring_and_a_real_regex_both_work(self, tmp_path: Path):
        root = self._two_cameras(tmp_path)
        assert len(list(iter_video_files([root], [r"cam-[12]"]))) == 4
        assert len(list(iter_video_files([root], ["trial003"]))) == 2

    def test_several_patterns_are_a_union(self, tmp_path: Path):
        root = self._two_cameras(tmp_path)
        kept = [str(p) for p in iter_video_files([root], ["cam-1", "trial004_cam-2"])]
        assert len(kept) == 3

    def test_none_keeps_everything(self, tmp_path: Path):
        root = self._two_cameras(tmp_path)
        assert len(list(iter_video_files([root], None))) == 4

    def test_a_filter_that_keeps_nothing_is_an_error(self, tmp_path: Path):
        """Silently extracting zero videos would look like success."""
        root = self._two_cameras(tmp_path)
        with pytest.raises(FileNotFoundError, match="matched none of the 4 videos"):
            list(iter_video_files([root], ["cam-9"]))

    def test_an_empty_list_is_refused(self, tmp_path: Path):
        root = self._two_cameras(tmp_path)
        with pytest.raises(ValueError, match="pass None"):
            list(iter_video_files([root], []))

    def test_a_malformed_pattern_names_itself(self, tmp_path: Path):
        root = self._two_cameras(tmp_path)
        with pytest.raises(ValueError, match=r"not a valid regular expression"):
            list(iter_video_files([root], ["cam-[1"]))

    def test_extract_videos_only_runs_s3d_on_the_kept_videos(self, tmp_path: Path, stub_s3d):
        root = self._two_cameras(tmp_path)
        written = extract_videos([root], tmp_path / "out", include=["cam-1"])
        assert len(written) == 2
        assert all("cam-1" in str(video) for video, _ in stub_s3d)

    def test_the_public_helper_passes_it_through(self, tmp_path: Path, stub_s3d):
        import ethograph as eto

        root = self._two_cameras(tmp_path)
        eto.segment.extract_videos([root], tmp_path / "out", include=["cam-2"])
        assert len(stub_s3d) == 2
        assert all("cam-2" in str(video) for video, _ in stub_s3d)
