"""Integration tests for wizard_single_from_pose and wizard_single_from_audio.

Pattern per test class:
  1. Run the wizard function to generate a .nc and alignment.nwb.
  2. Load the generated .nc in the GUI via the gui fixture.
  3. Assert the GUI is ready and the expected data (pose, audio) is accessible.
"""

from pathlib import Path

import pytest
import xarray as xr
from qtpy.QtWidgets import QApplication

from ethograph.datasets import dataset_dir, is_dataset_downloaded
from ethograph.io.data_loader import wizard_single_from_audio, wizard_single_from_pose

MOLL_DIR = dataset_dir("moll2025")
BIRDPARK_DIR = dataset_dir("birdpark")

_MOLL_VIDEO = MOLL_DIR / "2024-12-17_115_Crow1-cam-1.mp4"
_MOLL_POSE = MOLL_DIR / "2024-12-17_115_Crow1-cam-1DLC.csv"

_BP_VIDEO = BIRDPARK_DIR / "BP_2021-05-25_08-12-51_655154_0380000.mp4"
_BP_AUDIO = BIRDPARK_DIR / "BP_2021-05-25_08-12-51_655154_0380000.wav"


def _probe_fps(video_path: Path) -> int:
    from ethograph.gui.wizard_single import get_video_fps

    fps = get_video_fps(str(video_path))
    assert fps is not None, f"Could not probe FPS from {video_path}"
    return fps


def _load_in_gui(meta, nc_path, video_folder=None, pose_folder=None, audio_folder=None, nwb_path=None):
    io = meta.io_widget
    io._clear_all_line_edits()
    io.nc_file_path_edit.setText(str(nc_path))
    meta.app_state.nc_file_path = str(nc_path)
    if video_folder:
        io.video_folder_edit.setText(str(video_folder))
        meta.app_state.video_folder = str(video_folder)
    if pose_folder:
        io.pose_folder_edit.setText(str(pose_folder))
        meta.app_state.pose_folder = str(pose_folder)
    if audio_folder:
        io.audio_folder_edit.setText(str(audio_folder))
        meta.app_state.audio_folder = str(audio_folder)
    if nwb_path and nwb_path.exists():
        io.nwb_file_path_edit.setText(str(nwb_path))
        meta.app_state.nwb_file_path = str(nwb_path)
    meta.data_widget.on_load_clicked()
    QApplication.processEvents()


# ===========================================================================
# Moll2025 — wizard_single_from_pose (DLC CSV + video)
# ===========================================================================


@pytest.mark.skipif(
    not is_dataset_downloaded("moll2025") or not _MOLL_VIDEO.exists() or not _MOLL_POSE.exists(),
    reason="moll2025 assets not downloaded",
)
class TestWizardSingleFromPose:
    @pytest.fixture
    def nc_path(self, tmp_path):
        fps = _probe_fps(_MOLL_VIDEO)
        nc = tmp_path / "session.nc"
        ds = wizard_single_from_pose(
            video_path=str(_MOLL_VIDEO),
            fps=fps,
            pose_path=str(_MOLL_POSE),
            source_software="DeepLabCut",
            output_nc_path=str(nc),
        )
        ds.to_netcdf(str(nc))
        return nc

    def test_nc_has_pose_variables(self, nc_path):
        ds = xr.open_dataset(str(nc_path))
        assert "position" in ds.data_vars
        assert "velocity" in ds.data_vars
        assert "speed" in ds.data_vars

    def test_alignment_nwb_created_next_to_nc(self, nc_path):
        nwb = nc_path.parent / ".ethograph" / "alignment.nwb"
        assert nwb.exists()

    def test_gui_loads_nc(self, nc_path, gui, qtbot):
        _, meta = gui
        nwb = nc_path.parent / ".ethograph" / "alignment.nwb"
        _load_in_gui(meta, nc_path, video_folder=MOLL_DIR, pose_folder=MOLL_DIR, nwb_path=nwb)
        assert meta.app_state.ready, "GUI failed to load wizard-generated .nc"

    def test_gui_has_pose_features(self, nc_path, gui, qtbot):
        _, meta = gui
        nwb = nc_path.parent / ".ethograph" / "alignment.nwb"
        _load_in_gui(meta, nc_path, video_folder=MOLL_DIR, pose_folder=MOLL_DIR, nwb_path=nwb)
        assert meta.app_state.ready
        combo = meta.data_widget.combos.get("features")
        assert combo is not None
        items = [combo.itemText(i) for i in range(combo.count())]
        assert any(f in items for f in ("position", "speed", "velocity")), f"No pose features in combo: {items}"

    def test_gui_has_trials(self, nc_path, gui, qtbot):
        _, meta = gui
        nwb = nc_path.parent / ".ethograph" / "alignment.nwb"
        _load_in_gui(meta, nc_path, video_folder=MOLL_DIR, pose_folder=MOLL_DIR, nwb_path=nwb)
        assert meta.app_state.ready
        assert len(meta.app_state.trials) >= 1


# ===========================================================================
# BirdPark — wizard_single_from_audio (video + WAV)
# ===========================================================================


@pytest.mark.skipif(
    not is_dataset_downloaded("birdpark") or not _BP_VIDEO.exists() or not _BP_AUDIO.exists(),
    reason="birdpark assets not downloaded",
)
class TestWizardSingleFromAudio:
    @pytest.fixture
    def nc_path(self, tmp_path):
        fps = _probe_fps(_BP_VIDEO)
        nc = tmp_path / "session.nc"
        ds = wizard_single_from_audio(
            video_path=str(_BP_VIDEO),
            fps=fps,
            audio_path=str(_BP_AUDIO),
            output_nc_path=str(nc),
        )
        ds.to_netcdf(str(nc))
        return nc

    def test_alignment_nwb_created_next_to_nc(self, nc_path):
        nwb = nc_path.parent / ".ethograph" / "alignment.nwb"
        assert nwb.exists()

    def test_alignment_has_video_and_audio(self, nc_path):
        from ethograph.io.nwb_alignment import make_nwb_alignment

        sio = make_nwb_alignment(nc_path.parent / ".ethograph" / "alignment.nwb")
        assert sio.get_media(1, "video", "cam-1") == _BP_VIDEO.name
        assert sio.get_media(1, "audio", "mic-1") == _BP_AUDIO.name

    def test_gui_loads_nc(self, nc_path, gui, qtbot):
        _, meta = gui
        nwb = nc_path.parent / ".ethograph" / "alignment.nwb"
        _load_in_gui(meta, nc_path, video_folder=BIRDPARK_DIR, audio_folder=BIRDPARK_DIR, nwb_path=nwb)
        assert meta.app_state.ready, "GUI failed to load wizard-generated .nc"

    def test_gui_detects_audio(self, nc_path, gui, qtbot):
        _, meta = gui
        nwb = nc_path.parent / ".ethograph" / "alignment.nwb"
        _load_in_gui(meta, nc_path, video_folder=BIRDPARK_DIR, audio_folder=BIRDPARK_DIR, nwb_path=nwb)
        assert meta.app_state.ready
        assert meta.app_state.has_audio, "has_audio should be True when audio file is present"
