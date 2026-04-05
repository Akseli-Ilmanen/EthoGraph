"""Test from_continuous with a BirdPark-like dataset split into 3 x 20s trials.

Verifies:
- TrialTree.from_continuous slices correctly
- NWB alignment file works with session-wide video/audio
- session_io returns correct offsets per trial
- Time coordinates are trial-relative (start at 0)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ethograph as eto
from ethograph.io.trialtree import TrialTree
from ethograph.utils.nwb import build_nwb_from_trial_table

_VIDEO_NAME = "BP_2021-05-25_08-12-51_655154_0380000.mp4"
_AUDIO_NAME = "BP_2021-05-25_08-12-51_655154_0380000.wav"
_FPS = 47.68


def _make_birdpark_ds() -> xr.Dataset:
    """Synthesize a 60s BirdPark-like dataset (vibration at ~100Hz, 2 individuals)."""
    sr = 100.0
    duration = 60.0
    n = int(sr * duration)
    time = np.arange(n) / sr
    ds = xr.Dataset(
        {
            "vibration": xr.DataArray(
                np.random.default_rng(42).standard_normal((n, 2)),
                dims=["time", "individuals"],
                coords={
                    "time": time,
                    "individuals": ["male", "female"],
                },
                attrs={"type": "features"},
            ),
        },
        attrs={"fps": _FPS},
    )
    return ds


def _make_epochs(n_trials: int = 3, chunk: float = 20.0) -> pd.DataFrame:
    return pd.DataFrame({
        "trial": list(range(1, n_trials + 1)),
        "start_time": [i * chunk for i in range(n_trials)],
        "stop_time": [(i + 1) * chunk for i in range(n_trials)],
    })


def _make_alignment_nwb(epochs: pd.DataFrame, output_dir: Path) -> Path:
    """Create an alignment NWB for session-wide video + audio."""
    trial_table = epochs.copy()
    trial_table["video_cam-1"] = _VIDEO_NAME
    trial_table["audio_mic-1"] = _AUDIO_NAME

    nwb_path = output_dir / ".ethograph" / "alignment.nwb"
    build_nwb_from_trial_table(trial_table, stream_rates={"video": _FPS, "pose": _FPS}, output_path=nwb_path)
    return nwb_path


# -- Tests ------------------------------------------------------------------


def test_continuous_trial_slicing():
    """from_continuous slices data and shifts time to 0 per trial."""
    ds = _make_birdpark_ds()
    epochs = _make_epochs()
    dt = TrialTree.from_continuous(ds, epochs)

    assert dt._is_continuous
    assert dt.trials == [1, 2, 3]

    ds1 = dt.trial(1)
    ds2 = dt.trial(2)
    ds3 = dt.trial(3)

    for i, trial_ds in enumerate([ds1, ds2, ds3], 1):
        t0 = float(trial_ds.time.values[0])
        assert abs(t0) < 0.02, f"Trial {i} time should start near 0, got {t0}"
        assert trial_ds.attrs["trial"] == i

    for trial_ds in [ds1, ds2, ds3]:
        duration = float(trial_ds.time.values[-1])
        assert 19.0 < duration < 20.5, f"Expected ~20s, got {duration:.1f}s"

    # Data should differ between trials (seeded RNG → distinct slices)
    assert not np.allclose(ds1["vibration"].values[:10], ds2["vibration"].values[:10])


def test_nwb_alignment_with_continuous(tmp_path):
    """NWB alignment provides correct video/audio offsets per trial."""
    ds = _make_birdpark_ds()
    epochs = _make_epochs()
    dt = TrialTree.from_continuous(ds, epochs)

    nwb_path = _make_alignment_nwb(epochs, tmp_path)
    dt.nwb_path = str(nwb_path)

    sio = dt.session_io
    assert sio.cameras == ["cam-1"]
    assert sio.mics == ["mic-1"]

    # Trial timing (session-absolute)
    assert dt.start_time(1) == 0.0
    assert dt.start_time(2) == 20.0
    assert dt.start_time(3) == 40.0
    assert dt.stop_time(1) == 20.0

    # Media files — all trials reference the same session-wide files
    assert dt.get_media(1, "video", "cam-1") == _VIDEO_NAME
    assert dt.get_media(2, "audio", "mic-1") == _AUDIO_NAME
    assert dt.get_media(3, "video", "cam-1") == _VIDEO_NAME

    # Video offset: session-wide video starts at t=0 absolute,
    # so for trial 2 (starts at 20s), trial-relative offset = 0 - 20 = -20s.
    offset_t1 = dt.stream_offset_for_trial(1, "video", "cam-1")
    offset_t2 = dt.stream_offset_for_trial(2, "video", "cam-1")
    offset_t3 = dt.stream_offset_for_trial(3, "video", "cam-1")
    assert abs(offset_t1 - 0.0) < 0.1, f"Trial 1 offset should be ~0, got {offset_t1}"
    assert abs(offset_t2 - (-20.0)) < 0.1, f"Trial 2 offset should be ~-20, got {offset_t2}"
    assert abs(offset_t3 - (-40.0)) < 0.1, f"Trial 3 offset should be ~-40, got {offset_t3}"

    # FPS from NWB
    detected_fps = dt.get_stream_rate("video", "cam-1")
    assert detected_fps is not None
    assert abs(detected_fps - _FPS) < 0.01


def test_materialise_preserves_data():
    """Materialising a continuous tree yields identical data."""
    ds = _make_birdpark_ds()
    epochs = _make_epochs()
    dt = TrialTree.from_continuous(ds, epochs)

    mat = dt.materialise()

    assert not mat._is_continuous
    assert mat.trials == dt.trials

    for trial_id in dt.trials:
        cont_ds = dt.trial(trial_id)
        mat_ds = mat.trial(trial_id)
        np.testing.assert_allclose(
            cont_ds["vibration"].values,
            mat_ds["vibration"].values,
        )


def test_itrial_and_trial_items():
    """itrial and trial_items work correctly on continuous trees."""
    ds = _make_birdpark_ds()
    epochs = _make_epochs()
    dt = TrialTree.from_continuous(ds, epochs)

    ds_i = dt.itrial(0)
    ds_t = dt.trial(1)
    np.testing.assert_allclose(ds_i.time.values, ds_t.time.values)

    items = list(dt.trial_items())
    assert len(items) == 3
    assert items[0][0] == 1
    assert items[2][0] == 3


def test_pynapple_epochs():
    """from_continuous accepts a pynapple IntervalSet."""
    import pynapple as nap

    ds = _make_birdpark_ds()
    ep = nap.IntervalSet(start=[0.01, 20.01, 40.01], end=[19.99, 39.99, 59.99])
    dt = TrialTree.from_continuous(ds, ep)

    assert dt.trials == [1, 2, 3]
    ds2 = dt.trial(2)
    assert abs(float(ds2.time.values[0])) < 0.02


def test_map_trials_materialises():
    """map_trials on a continuous tree returns a standard per-node tree."""
    ds = _make_birdpark_ds()
    epochs = _make_epochs()
    dt = TrialTree.from_continuous(ds, epochs)

    result = dt.map_trials(lambda d: d)
    assert not result._is_continuous
    assert result.trials == [1, 2, 3]


def test_update_trial_raises():
    """update_trial raises on continuous trees."""
    ds = _make_birdpark_ds()
    epochs = _make_epochs()
    dt = TrialTree.from_continuous(ds, epochs)

    with pytest.raises(TypeError, match="continuous"):
        dt.update_trial(1, lambda d: d)


def test_get_all_trials():
    """get_all_trials returns all sliced datasets."""
    ds = _make_birdpark_ds()
    epochs = _make_epochs()
    dt = TrialTree.from_continuous(ds, epochs)

    all_t = dt.get_all_trials()
    assert sorted(all_t.keys()) == [1, 2, 3]
    for tid, trial_ds in all_t.items():
        assert trial_ds.attrs["trial"] == tid


def test_continuous_xarray_and_nwb_together(tmp_path):
    """End-to-end: continuous xarray + NWB alignment, verify trial data + offsets match."""
    ds = _make_birdpark_ds()
    epochs = _make_epochs()
    dt = TrialTree.from_continuous(ds, epochs)

    nwb_path = _make_alignment_nwb(epochs, tmp_path)
    dt.nwb_path = str(nwb_path)

    for trial_id in [1, 2, 3]:
        trial_ds = dt.trial(trial_id)
        # xarray: time starts at 0, duration ~20s
        t_start = float(trial_ds.time.values[0])
        t_end = float(trial_ds.time.values[-1])
        assert abs(t_start) < 0.02
        assert 19.0 < t_end < 20.5

        # NWB: session-absolute timing
        nwb_start = dt.start_time(trial_id)
        nwb_stop = dt.stop_time(trial_id)
        assert abs(nwb_stop - nwb_start - 20.0) < 0.01

        # Video offset bridges the two
        v_offset = dt.stream_offset_for_trial(trial_id, "video", "cam-1")
        expected_offset = 0.0 - nwb_start
        assert abs(v_offset - expected_offset) < 0.1

    # Verify the data is consistent: trial 2 xarray data should match
    # the original dataset sliced at 20-40s
    ds2 = dt.trial(2)
    orig_slice = ds.sel(time=slice(20.0, 40.0))
    np.testing.assert_allclose(
        ds2["vibration"].values,
        orig_slice["vibration"].values,
    )
