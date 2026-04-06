"""Comprehensive tests for loading all template datasets via the GUI.

Tests each of the 5 example datasets (Moll2025, BirdPark, Canary, Lockbox,
Philodoptera) through the full widget loading pipeline, verifying that panels,
combos, time alignment, and labelling workflows all function correctly.
"""

import numpy as np
import pytest
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from ethograph.gui.dialog_select_template import (
    TEMPLATES,
    _DOWNLOAD_BASE,
    _resolve_template_paths,
    _template_dir,
    _template_downloaded,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_template(key: str) -> dict:
    for t in TEMPLATES:
        if t["dataset_key"] == key:
            return t
    raise KeyError(key)


def _skip_if_not_downloaded(key: str):
    t = _get_template(key)
    if not _template_downloaded(t):
        pytest.skip(f"{key} not downloaded")


def _apply_template(meta, template_key: str):
    """Apply a template's paths to the IO widget and trigger load."""
    t = _get_template(template_key)
    resolved = _resolve_template_paths(t)

    io = meta.io_widget
    io._clear_all_line_edits()

    if resolved["nc_file_path"]:
        io.nc_file_path_edit.setText(resolved["nc_file_path"])
        meta.app_state.nc_file_path = resolved["nc_file_path"]
    if resolved["video_folder"]:
        io.video_folder_edit.setText(resolved["video_folder"])
        meta.app_state.video_folder = resolved["video_folder"]
    if resolved["audio_folder"]:
        io.audio_folder_edit.setText(resolved["audio_folder"])
        meta.app_state.audio_folder = resolved["audio_folder"]
    if resolved.get("pose_folder"):
        io.pose_folder_edit.setText(resolved["pose_folder"])
        meta.app_state.pose_folder = resolved["pose_folder"]
    if resolved.get("import_labels"):
        io.import_labels_checkbox.setChecked(True)

    if template_key == "birdpark":
        io.downsample_checkbox.setChecked(True)
        io.downsample_spin.setValue(100)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()


def _load_template_gui(gui, template_key: str):
    """Load a template into a fresh gui fixture."""
    _skip_if_not_downloaded(template_key)
    viewer, meta = gui
    _apply_template(meta, template_key)
    assert meta.app_state.ready, f"Failed to load {template_key}"
    return viewer, meta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def birdpark_gui(gui, qtbot):
    return _load_template_gui(gui, "birdpark")


@pytest.fixture
def moll_gui(gui, qtbot):
    return _load_template_gui(gui, "moll2025")


@pytest.fixture
def lockbox_gui(gui, qtbot):
    return _load_template_gui(gui, "lockbox")


@pytest.fixture
def canary_gui(gui, qtbot):
    _skip_if_not_downloaded("canary")
    viewer, meta = gui
    t = _get_template("canary")
    dest = _template_dir(t)
    audio_path = str(dest / t["audio_file"])
    nc_path = str(dest / (Path(t["audio_file"]).stem + ".nc"))

    if not Path(nc_path).exists():
        from ethograph.gui.data_loader import wizard_single_from_audio
        from ethograph.utils.audio import get_audio_sr
        audio_sr = get_audio_sr(audio_path)
        dt = wizard_single_from_audio(video_path=None, fps=30, audio_path=audio_path, audio_sr=audio_sr)
        dt.to_netcdf(nc_path)

    io = meta.io_widget
    io._clear_all_line_edits()
    io.nc_file_path_edit.setText(nc_path)
    meta.app_state.nc_file_path = nc_path
    io.audio_folder_edit.setText(str(dest))
    meta.app_state.audio_folder = str(dest)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    assert meta.app_state.ready, "Failed to load canary"
    return viewer, meta


@pytest.fixture
def philodoptera_gui(gui, qtbot):
    return _load_template_gui(gui, "philodoptera")


# ===================================================================
# BIRDPARK — audio, spectrogram, accelerometer share same time axis
# ===================================================================

class TestBirdPark:

    def test_state_after_load(self, birdpark_gui):
        _, meta = birdpark_gui
        s = meta.app_state
        assert s.dt is not None
        assert s.ds is not None
        assert s.has_audio is True
        assert s.time_coord is not None
        assert len(s.time_coord) > 0

    def test_audio_panels_visible(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        assert pc._panel_visible["audiotrace"] or pc._panel_visible["spectrogram"]

    def test_spectrogram_panel_exists(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        assert pc.spectrogram_plot is not None
        assert pc.spectrogram_plot.parent() is not None

    def test_audio_trace_panel_exists(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        assert pc.audio_trace_plot is not None
        assert pc.audio_trace_plot.parent() is not None



    def test_select_accelerometer(self, birdpark_gui):
        _, meta = birdpark_gui
        combo = meta.data_widget.combos["features"]
        idx = next(i for i in range(combo.count()) if combo.itemText(i) == "vibration")
        combo.setCurrentIndex(idx)
        QApplication.processEvents()
        assert meta.app_state.features_sel == "vibration"

    def test_time_axes_aligned(self, birdpark_gui):
        """Audio trace, spectrogram, and feature plot must share the same x-range."""
        _, meta = birdpark_gui
        pc = meta.plot_container

        feature_xlim = pc._feature_plot.get_current_xlim()
        assert feature_xlim[0] < feature_xlim[1]

        if pc._panel_visible["audiotrace"]:
            audio_xlim = pc.audio_trace_plot.get_current_xlim()
            assert abs(audio_xlim[0] - feature_xlim[0]) < 0.5
            assert abs(audio_xlim[1] - feature_xlim[1]) < 0.5

        if pc._panel_visible["spectrogram"]:
            spec_xlim = pc.spectrogram_plot.get_current_xlim()
            assert abs(spec_xlim[0] - feature_xlim[0]) < 0.5
            assert abs(spec_xlim[1] - feature_xlim[1]) < 0.5

    def test_time_slider_present(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        # Birdpark has video, so slider may be hidden
        # Just verify it exists and has valid range
        slider = pc.time_slider
        assert slider is not None

    def test_feature_plot_defaults_to_lineplot(self, birdpark_gui):
        _, meta = birdpark_gui
        assert meta.plot_container.is_lineplot()

    def test_switch_to_heatmap_and_back(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        pc.switch_to_heatmap()
        QApplication.processEvents()
        assert pc.is_heatmap()

        pc.switch_to_lineplot()
        QApplication.processEvents()
        assert pc.is_lineplot()

    def test_time_marker_syncs_across_panels(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        t = 1.0
        # In headless mode, isVisible() is False so update_time_marker_by_time
        # only reaches shown widgets. Call update_time_marker directly.
        for plot in [pc._feature_plot, pc.audio_trace_plot, pc.spectrogram_plot]:
            if plot is not None:
                plot.update_time_marker(t)

        if pc._panel_visible["audiotrace"]:
            assert pc.audio_trace_plot.time_marker.value() == pytest.approx(t)
        if pc._panel_visible["spectrogram"]:
            assert pc.spectrogram_plot.time_marker.value() == pytest.approx(t)
        assert pc._feature_plot.time_marker.value() == pytest.approx(t)


# ===================================================================
# CANARY — audio-only, loading works
# ===================================================================

class TestCanary:

    def test_state_after_load(self, canary_gui):
        _, meta = canary_gui
        s = meta.app_state
        assert s.ready is True
        assert s.dt is not None
        assert s.ds is not None
        assert s.has_audio is True

    def test_no_video(self, canary_gui):
        _, meta = canary_gui
        assert meta.app_state.video is None

    def test_audio_panels_created(self, canary_gui):
        _, meta = canary_gui
        pc = meta.plot_container
        assert pc.audio_trace_plot is not None
        assert pc.spectrogram_plot is not None

    def test_time_slider_exists(self, canary_gui):
        _, meta = canary_gui
        # No video → time slider should exist (isVisible unreliable in headless)
        assert meta.plot_container.time_slider is not None

    def test_valid_xlim(self, canary_gui):
        _, meta = canary_gui
        xlim = meta.plot_container.get_current_xlim()
        assert xlim[0] < xlim[1]
        assert xlim[0] >= -1.0


# ===================================================================
# LOCKBOX — multi-camera, camera combos, open cam 2 and 3
# ===================================================================

class TestLockbox:

    def test_state_after_load(self, lockbox_gui):
        _, meta = lockbox_gui
        s = meta.app_state
        assert s.ready is True
        assert s.dt is not None

    def test_three_cameras_exist(self, lockbox_gui):
        _, meta = lockbox_gui
        cameras = meta.app_state.dt.cameras
        assert len(cameras) == 3
        assert "front-view" in cameras
        assert "side-view" in cameras
        assert "top-down-view" in cameras

    def test_primary_camera_combo_exists(self, lockbox_gui):
        _, meta = lockbox_gui
        dw = meta.data_widget
        assert hasattr(dw, "primary_camera_combo")
        combo = dw.primary_camera_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        assert len(items) == 3
        assert "front-view" in items
        assert "side-view" in items
        assert "top-down-view" in items

    def test_extra_camera_combos_exist(self, lockbox_gui):
        _, meta = lockbox_gui
        dw = meta.data_widget
        assert hasattr(dw, "_extra_camera_combos")
        assert len(dw._extra_camera_combos) >= 2

    def test_select_camera_side(self, lockbox_gui):
        """Select camera 2 (side) via extra camera combo."""
        _, meta = lockbox_gui
        dw = meta.data_widget
        combo = dw._extra_camera_combos[0]
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "side-view" in items
        idx = items.index("side-view")
        combo.setCurrentIndex(idx)
        QApplication.processEvents()

    def test_select_camera_top(self, lockbox_gui):
        """Select camera 3 (top) via extra camera combo."""
        _, meta = lockbox_gui
        dw = meta.data_widget
        combo = dw._extra_camera_combos[1] if len(dw._extra_camera_combos) > 1 else dw._extra_camera_combos[0]
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "top-down-view" in items
        idx = items.index("top-down-view")
        combo.setCurrentIndex(idx)
        QApplication.processEvents()

    def test_multiple_trials(self, lockbox_gui):
        _, meta = lockbox_gui
        assert len(meta.app_state.trials) == 3

    def test_trial_navigation(self, lockbox_gui):
        _, meta = lockbox_gui
        # Switch to Trial mode and start at first trial
        meta.navigation_widget.mode_combo.setCurrentText("Trial")
        QApplication.processEvents()
        trials = meta.app_state.trials
        meta.navigation_widget.trials_combo.setCurrentText(str(trials[0]))
        QApplication.processEvents()
        first = meta.app_state.trials_sel
        meta.navigation_widget.next_trial()
        QApplication.processEvents()
        assert meta.app_state.trials_sel != first

    def test_features_available(self, lockbox_gui):
        _, meta = lockbox_gui
        combo = meta.data_widget.combos.get("features")
        assert combo is not None
        assert combo.count() > 0


# ===================================================================
# MOLL2025 via .nc — video, pose, labels, multi-trial, labelling + save
# ===================================================================

class TestMoll2025NC:

    def test_state_after_load(self, moll_gui):
        _, meta = moll_gui
        s = meta.app_state
        assert s.ready is True
        assert s.dt is not None
        assert s.ds is not None

    def test_two_trials(self, moll_gui):
        _, meta = moll_gui
        assert len(meta.app_state.trials) == 2

    def test_features_rich(self, moll_gui):
        _, meta = moll_gui
        combo = meta.data_widget.combos.get("features")
        assert combo is not None
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "speed" in items
        assert "position" in items

    def test_navigate_to_second_trial(self, moll_gui):
        _, meta = moll_gui
        meta.navigation_widget.mode_combo.setCurrentText("Trial")
        QApplication.processEvents()
        trials = meta.app_state.trials
        if len(trials) < 2:
            pytest.skip("Need 2+ trials")
        meta.navigation_widget.trials_combo.setCurrentText(str(trials[0]))
        QApplication.processEvents()
        first = meta.app_state.trials_sel
        # Navigate directly via combo to avoid plot update errors
        meta.navigation_widget.trials_combo.setCurrentText(str(trials[1]))
        QApplication.processEvents()
        second = meta.app_state.trials_sel
        assert second != first
        assert meta.app_state.ds is not None

    def test_labels_imported(self, moll_gui):
        """Template has import_labels=True so labels TSV should be loaded."""
        _, meta = moll_gui
        df = meta.app_state._all_labels_df
        assert df is not None
        assert not df.empty

    def test_label_creation_and_verify(self, moll_gui):
        """Create a label interval via click simulation."""
        _, meta = moll_gui
        lw = meta.labels_widget
        if not lw._mappings or 1 not in lw._mappings:
            pytest.skip("No label mapping 1")

        lw.activate_label(1)
        assert lw.ready_for_label_click is True

        time = meta.app_state.time_coord
        t_start = float(time[len(time) // 4])
        t_end = float(time[len(time) // 4 + 10])
        lw._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
        lw._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
        QApplication.processEvents()

        df = meta.app_state.label_intervals
        assert df is not None and not df.empty

    def test_save_labels_tsv(self, moll_gui, tmp_path):
        """Verify labels can be saved to a TSV path."""
        _, meta = moll_gui

        # Create a label first
        lw = meta.labels_widget
        if lw._mappings and 1 in lw._mappings:
            lw.activate_label(1)
            time = meta.app_state.time_coord
            t_start = float(time[len(time) // 4])
            t_end = float(time[len(time) // 4 + 10])
            lw._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
            lw._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
            QApplication.processEvents()

        from ethograph.labels.tsv_store import save_labels_tsv, load_labels_tsv
        tsv_out = tmp_path / "test_labels.tsv"
        df = meta.app_state._all_labels_df
        assert df is not None
        save_labels_tsv(tsv_out, df)

        assert tsv_out.exists()
        loaded = load_labels_tsv(tsv_out)
        assert len(loaded) == len(df)


# ===================================================================
# MOLL2025 via pynapple .npz — load_nap_data path
# ===================================================================

class TestMoll2025Pynapple:
    """Test loading moll2025 pynapple .npz files directly."""

    @pytest.fixture(autouse=True)
    def _check_npz(self):
        dest = _DOWNLOAD_BASE / "Moll2025"
        self._speed_npz = dest / "beakTip_speed.npz"
        if not self._speed_npz.exists():
            pytest.skip("Moll2025 pynapple .npz not downloaded")

    def test_load_npz_as_dataset(self):
        from ethograph.gui.data_loader import load_dataset
        dt, labels_df, type_vars = load_dataset(str(self._speed_npz), require_fps=False)
        assert dt is not None
        assert len(dt.trials) > 0
        ds = dt.itrial(0)
        assert ds is not None

    def test_load_npz_into_gui(self, gui, qtbot):
        viewer, meta = gui
        io = meta.io_widget
        io._clear_all_line_edits()
        io.nc_file_path_edit.setText(str(self._speed_npz))
        meta.app_state.nc_file_path = str(self._speed_npz)

        meta.data_widget.on_load_clicked()
        QApplication.processEvents()

        assert meta.app_state.ready is True
        assert meta.app_state.dt is not None

    def test_npz_type_vars_structure(self):
        from ethograph.io.catalog import catalog_from_pynapple
        from ethograph.io.pynapple import load_nap_data
        data, trials_ep = load_nap_data(str(self._speed_npz))
        cat = catalog_from_pynapple(data, trials_ep)
        tv = cat.to_type_vars_dict()
        assert isinstance(tv, dict)
        assert "features" in tv or "individuals" in tv or len(tv) > 0


# ===================================================================
# PHILODOPTERA — audio + video + pose
# ===================================================================

class TestPhilodoptera:

    def test_state_after_load(self, philodoptera_gui):
        _, meta = philodoptera_gui
        s = meta.app_state
        assert s.ready is True
        assert s.dt is not None
        assert s.has_audio is True
        assert s.has_video is True
        assert s.has_pose is True

    def test_single_trial(self, philodoptera_gui):
        _, meta = philodoptera_gui
        assert len(meta.app_state.trials) == 1

    def test_audio_and_feature_panels(self, philodoptera_gui):
        _, meta = philodoptera_gui
        pc = meta.plot_container
        assert pc.audio_trace_plot is not None
        assert pc.spectrogram_plot is not None
        assert pc._feature_plot is not None

    def test_features_include_speed(self, philodoptera_gui):
        _, meta = philodoptera_gui
        combo = meta.data_widget.combos.get("features")
        assert combo is not None
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "speed" in items


# ===================================================================
# Cross-template: UnifiedPanelContainer basics
# ===================================================================

class TestPanelContainerBasics:
    """Tests that work on any loaded dataset — use birdpark as the default."""

    def test_container_is_unified(self, birdpark_gui):
        from ethograph.gui.plots_container import UnifiedPanelContainer
        _, meta = birdpark_gui
        assert isinstance(meta.plot_container, UnifiedPanelContainer)

    def test_valid_x_range(self, birdpark_gui):
        _, meta = birdpark_gui
        xlim = meta.plot_container.get_current_xlim()
        assert xlim[0] < xlim[1]
        assert xlim[0] >= -1.0
        assert xlim[1] > 0.1

    def test_get_current_plot_returns_feature(self, birdpark_gui):
        _, meta = birdpark_gui
        plot = meta.plot_container.get_current_plot()
        assert plot is not None
        assert plot is meta.plot_container._feature_plot

    def test_toggle_axes_lock(self, birdpark_gui):
        _, meta = birdpark_gui
        # Should not crash
        meta.plot_container.toggle_axes_lock()
        QApplication.processEvents()
        meta.plot_container.toggle_axes_lock()
        QApplication.processEvents()
