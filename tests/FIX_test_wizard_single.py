"""Tests for wizard_single data-loader functions using template dataset files.

Each test group skips if the corresponding template dataset has not been
downloaded to ``~/.ethograph/example_data/``.
"""

import shutil
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path

from ethograph.datasets import dataset_dir, is_dataset_downloaded
from ethograph.io.data_loader import (
    wizard_single_from_audio,
    wizard_single_from_ds,
    wizard_single_from_npy_file,
    wizard_single_from_pose,
    wizard_single_from_ephys,
    wizard_single_from_video,
)
from ethograph.gui.wizard_single import get_video_fps
from ethograph.io.trialtree import TrialTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_not_downloaded(key: str):
    if not is_dataset_downloaded(key):
        pytest.skip(f"{key} not downloaded")


def _assert_valid_trialtree(dt):
    """Basic structural checks that every wizard output must satisfy."""
    assert isinstance(dt, TrialTree)
    assert len(dt.trials) >= 1
    ds = dt.itrial(0)
    assert ds is not None
    assert ds.attrs.get("trial") is not None


def _safe_to_netcdf(dt, path):
    """Write TrialTree to netcdf, skipping if netCDF4 has binary compat issues."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            dt.to_netcdf(str(path))
    except RuntimeWarning:
        pytest.skip("netCDF4/numpy binary incompatibility in this env")


# ===================================================================
# get_video_fps — read-only, uses real template videos
# ===================================================================

class TestGetVideoFps:

    def test_moll_video(self):
        _skip_if_not_downloaded("moll2025")
        d = dataset_dir("moll2025")
        mp4 = next(d.glob("*.mp4"))
        fps = get_video_fps(str(mp4))
        assert fps is not None
        assert isinstance(fps, int)
        assert 1 <= fps <= 240

    def test_birdpark_video(self):
        _skip_if_not_downloaded("birdpark")
        d = dataset_dir("birdpark")
        mp4 = next(d.glob("*.mp4"))
        fps = get_video_fps(str(mp4))
        assert fps is not None
        assert 1 <= fps <= 240

    def test_lockbox_video(self):
        _skip_if_not_downloaded("lockbox")
        d = dataset_dir("lockbox")
        mp4 = next(d.glob("*.mp4"))
        fps = get_video_fps(str(mp4))
        assert fps is not None
        assert 1 <= fps <= 240

    def test_philodoptera_video(self):
        _skip_if_not_downloaded("philodoptera")
        d = dataset_dir("philodoptera")
        mp4 = d / "philodoptera.mp4"
        fps = get_video_fps(str(mp4))
        assert fps is not None
        assert 1 <= fps <= 240

    def test_nonexistent_file_returns_none(self):
        assert get_video_fps("/nonexistent/video.mp4") is None

    def test_invalid_file_returns_none(self, tmp_path):
        bad = tmp_path / "not_a_video.mp4"
        bad.write_bytes(b"not a video")
        assert get_video_fps(str(bad)) is None


# ===================================================================
# wizard_single_from_audio — Canary .wav and BirdPark .wav
# ===================================================================

class TestWizardFromAudio:

    def test_canary_audio(self, tmp_path):
        _skip_if_not_downloaded("canary")
        src = dataset_dir("canary") / "100_marron1_May_24_2016_62101389.wav"
        wav = tmp_path / src.name
        shutil.copy2(src, wav)

        from ethograph.utils.audio import get_audio_sr
        sr = get_audio_sr(str(wav))
        assert sr is not None

        dt = wizard_single_from_audio(
            video_path=None,
            fps=30,
            audio_path=str(wav),
            audio_sr=sr,
        )
        _assert_valid_trialtree(dt)

    def test_birdpark_audio(self, tmp_path):
        _skip_if_not_downloaded("birdpark")
        src = dataset_dir("birdpark") / "BP_2021-05-25_08-12-51_655154_0380000.wav"
        wav = tmp_path / src.name
        shutil.copy2(src, wav)

        from ethograph.utils.audio import get_audio_sr
        sr = get_audio_sr(str(wav))

        dt = wizard_single_from_audio(
            video_path=None,
            fps=30,
            audio_path=str(wav),
            audio_sr=sr,
        )
        _assert_valid_trialtree(dt)

    def test_philodoptera_audio(self, tmp_path):
        _skip_if_not_downloaded("philodoptera")
        src = dataset_dir("philodoptera") / "philodoptera.wav"
        wav = tmp_path / src.name
        shutil.copy2(src, wav)

        from ethograph.utils.audio import get_audio_sr
        sr = get_audio_sr(str(wav))

        dt = wizard_single_from_audio(
            video_path=None,
            fps=30,
            audio_path=str(wav),
            audio_sr=sr,
        )
        _assert_valid_trialtree(dt)

    def test_roundtrip_to_netcdf(self, tmp_path):
        _skip_if_not_downloaded("canary")
        src = dataset_dir("canary") / "100_marron1_May_24_2016_62101389.wav"
        wav = tmp_path / src.name
        shutil.copy2(src, wav)

        from ethograph.utils.audio import get_audio_sr
        sr = get_audio_sr(str(wav))

        dt = wizard_single_from_audio(
            video_path=None, fps=30,
            audio_path=str(wav), audio_sr=sr,
        )
        nc_out = tmp_path / "audio_trial.nc"
        _safe_to_netcdf(dt, nc_out)
        assert nc_out.exists()
        assert nc_out.stat().st_size > 0

    def test_custom_individuals(self, tmp_path):
        _skip_if_not_downloaded("canary")
        src = dataset_dir("canary") / "100_marron1_May_24_2016_62101389.wav"
        wav = tmp_path / src.name
        shutil.copy2(src, wav)

        from ethograph.utils.audio import get_audio_sr
        sr = get_audio_sr(str(wav))

        dt = wizard_single_from_audio(
            video_path=None, fps=30,
            audio_path=str(wav), audio_sr=sr,
            individuals=["bird_A", "bird_B"],
        )
        _assert_valid_trialtree(dt)
        ds = dt.itrial(0)
        assert "bird_A" in ds.coords["individuals"].values
        assert "bird_B" in ds.coords["individuals"].values


# ===================================================================
# wizard_single_from_pose — Moll DLC .csv, Philodoptera .csv
# ===================================================================

class TestWizardFromPose:

    def test_moll_dlc_pose(self, tmp_path):
        _skip_if_not_downloaded("moll2025")
        d = dataset_dir("moll2025")
        csv_src = next(d.glob("*DLC.csv"))
        csv_dst = tmp_path / csv_src.name
        shutil.copy2(csv_src, csv_dst)

        dt = wizard_single_from_pose(
            video_path=None,
            fps=30,
            pose_path=str(csv_dst),
            source_software="DeepLabCut",
        )
        _assert_valid_trialtree(dt)
        ds = dt.itrial(0)
        assert "speed" in ds or "speed" in [v for v in ds.data_vars]

    def test_pose_has_kinematics(self, tmp_path):
        _skip_if_not_downloaded("moll2025")
        d = dataset_dir("moll2025")
        csv_src = next(d.glob("*DLC.csv"))
        csv_dst = tmp_path / csv_src.name
        shutil.copy2(csv_src, csv_dst)

        dt = wizard_single_from_pose(
            video_path=None, fps=30,
            pose_path=str(csv_dst),
            source_software="DeepLabCut",
        )
        ds = dt.itrial(0)
        assert "velocity" in ds.data_vars
        assert "speed" in ds.data_vars
        assert "acceleration" in ds.data_vars

    def test_roundtrip_to_netcdf(self, tmp_path):
        _skip_if_not_downloaded("moll2025")
        d = dataset_dir("moll2025")
        csv_src = next(d.glob("*DLC.csv"))
        csv_dst = tmp_path / csv_src.name
        shutil.copy2(csv_src, csv_dst)

        dt = wizard_single_from_pose(
            video_path=None, fps=30,
            pose_path=str(csv_dst),
            source_software="DeepLabCut",
        )
        nc_out = tmp_path / "pose_trial.nc"
        _safe_to_netcdf(dt, nc_out)
        assert nc_out.exists()
        assert nc_out.stat().st_size > 0


# ===================================================================
# wizard_single_from_npy_file — synthetic .npy data
# ===================================================================

class TestWizardFromNpy:

    def test_2d_array(self, tmp_path):
        data = np.random.randn(500, 4).astype(np.float32)
        npy = tmp_path / "features.npy"
        np.save(str(npy), data)

        dt = wizard_single_from_npy_file(
            video_path=None, fps=30,
            npy_path=str(npy), data_sr=30,
        )
        _assert_valid_trialtree(dt)
        ds = dt.itrial(0)
        assert "data" in ds.data_vars
        assert ds["data"].shape[0] == 500
        assert ds["data"].shape[1] == 4

    def test_1d_array_reshaped(self, tmp_path):
        data = np.random.randn(200).astype(np.float32)
        npy = tmp_path / "signal.npy"
        np.save(str(npy), data)

        dt = wizard_single_from_npy_file(
            video_path=None, fps=30,
            npy_path=str(npy), data_sr=100,
        )
        _assert_valid_trialtree(dt)
        ds = dt.itrial(0)
        assert ds["data"].shape[0] == 200

    def test_transposed_array(self, tmp_path):
        """If n_samples < n_variables, array should be auto-transposed."""
        data = np.random.randn(3, 1000).astype(np.float32)
        npy = tmp_path / "wide.npy"
        np.save(str(npy), data)

        dt = wizard_single_from_npy_file(
            video_path=None, fps=30,
            npy_path=str(npy), data_sr=50,
        )
        ds = dt.itrial(0)
        assert ds["data"].shape[0] == 1000
        assert ds["data"].shape[1] == 3

    def test_custom_individuals(self, tmp_path):
        data = np.random.randn(100, 2).astype(np.float32)
        npy = tmp_path / "feat.npy"
        np.save(str(npy), data)

        dt = wizard_single_from_npy_file(
            video_path=None, fps=30,
            npy_path=str(npy), data_sr=30,
            individuals=["mouse_1"],
        )
        ds = dt.itrial(0)
        assert "mouse_1" in ds.coords["individuals"].values

    def test_time_coords_match_sr(self, tmp_path):
        data = np.random.randn(300, 2).astype(np.float32)
        npy = tmp_path / "timed.npy"
        np.save(str(npy), data)

        dt = wizard_single_from_npy_file(
            video_path=None, fps=30,
            npy_path=str(npy), data_sr=100,
        )
        ds = dt.itrial(0)
        time = ds.coords["time"].values
        assert len(time) == 300
        assert np.isclose(time[1] - time[0], 1.0 / 100, atol=1e-9)

    def test_roundtrip_to_netcdf(self, tmp_path):
        data = np.random.randn(50, 5).astype(np.float32)
        npy = tmp_path / "rt.npy"
        np.save(str(npy), data)

        dt = wizard_single_from_npy_file(
            video_path=None, fps=30,
            npy_path=str(npy), data_sr=30,
        )
        nc_out = tmp_path / "npy_trial.nc"
        _safe_to_netcdf(dt, nc_out)
        assert nc_out.exists()



# ===================================================================
# wizard_single_from_ephys — minimal (no real ephys files in templates)
# ===================================================================

class TestWizardFromEphys:

    def test_basic_no_files(self, tmp_path):
        dt = wizard_single_from_ephys(video_path=None, fps=30)
        _assert_valid_trialtree(dt)

    def test_custom_individuals(self, tmp_path):
        dt = wizard_single_from_ephys(
            video_path=None, fps=30,
            individuals=["rat_1", "rat_2"],
        )
        ds = dt.itrial(0)
        assert "rat_1" in ds.coords["individuals"].values
        assert "rat_2" in ds.coords["individuals"].values

    def test_with_audio(self, tmp_path):
        _skip_if_not_downloaded("canary")
        src = dataset_dir("canary") / "100_marron1_May_24_2016_62101389.wav"
        wav = tmp_path / src.name
        shutil.copy2(src, wav)

        dt = wizard_single_from_ephys(
            video_path=None, fps=30,
            audio_path=str(wav),
        )
        _assert_valid_trialtree(dt)


# ===================================================================
# wizard_single_from_video — video file only (motion energy)
# ===================================================================

class TestWizardFromVideo:

    def test_moll_video(self, tmp_path):
        _skip_if_not_downloaded("moll2025")
        d = dataset_dir("moll2025")
        mp4 = next(d.glob("*.mp4"))
        dt = wizard_single_from_video(video_path=str(mp4))
        _assert_valid_trialtree(dt)
        ds = dt.itrial(0)
        assert "video_motion" in ds.data_vars
        assert len(ds["video_motion"]) > 0
        assert ds["video_motion"].dtype == np.float32
        assert ds.attrs["fps"] == get_video_fps(str(mp4))

    def test_invalid_file_raises(self, tmp_path):
        bad = tmp_path / "not_a_video.mp4"
        bad.write_bytes(b"not a video")
        with pytest.raises((ValueError, RuntimeError)):
            wizard_single_from_video(video_path=str(bad))
