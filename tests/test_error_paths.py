"""Tests for error handling: corrupt files, missing data, invalid state transitions.

Verifies the GUI handles bad input gracefully without crashing.
"""

import pytest
import numpy as np
from pathlib import Path
from qtpy.QtWidgets import QApplication
from ethograph.gui.dialog_select_template import _DOWNLOAD_BASE

_BIRDPARK_NC = _DOWNLOAD_BASE / "BirdPark" / "copExpBP08_trim.nc"


class TestLoadNonexistentFile:

    def test_load_missing_file_stays_not_ready(self, gui):
        """Loading a file that doesn't exist should not crash, state stays not-ready."""
        _, meta = gui
        meta.io_widget.nc_file_path_edit.setText("/nonexistent/path/fake.nc")
        meta.app_state.nc_file_path = "/nonexistent/path/fake.nc"
        meta.io_widget.downsample_checkbox.setChecked(False)

        meta.data_widget.on_load_clicked()
        QApplication.processEvents()

        assert meta.app_state.ready is False
        assert meta.app_state.dt is None

    def test_load_empty_path_stays_not_ready(self, gui):
        """Empty path should not crash."""
        _, meta = gui
        meta.io_widget.nc_file_path_edit.setText("")
        meta.app_state.nc_file_path = ""

        meta.data_widget.on_load_clicked()
        QApplication.processEvents()

        assert meta.app_state.ready is False


class TestLoadCorruptFile:

    def test_load_non_nc_file(self, gui, tmp_path):
        """Loading a text file renamed to .nc should fail gracefully."""
        fake_nc = tmp_path / "corrupt.nc"
        fake_nc.write_text("this is not a netcdf file")

        _, meta = gui
        meta.io_widget.nc_file_path_edit.setText(str(fake_nc))
        meta.app_state.nc_file_path = str(fake_nc)
        meta.io_widget.downsample_checkbox.setChecked(False)

        meta.data_widget.on_load_clicked()
        QApplication.processEvents()

        assert meta.app_state.ready is False

    def test_load_empty_file(self, gui, tmp_path):
        """Loading an empty file should fail gracefully."""
        empty_nc = tmp_path / "empty.nc"
        empty_nc.write_bytes(b"")

        _, meta = gui
        meta.io_widget.nc_file_path_edit.setText(str(empty_nc))
        meta.app_state.nc_file_path = str(empty_nc)
        meta.io_widget.downsample_checkbox.setChecked(False)

        meta.data_widget.on_load_clicked()
        QApplication.processEvents()

        assert meta.app_state.ready is False


class TestInvalidStateTransitions:

    def test_next_trial_before_load(self, gui):
        """Calling next_trial before data is loaded should not crash."""
        _, meta = gui
        # Should be a no-op or handle gracefully
        meta.navigation_widget.next_trial()
        QApplication.processEvents()
        assert meta.app_state.ready is False

    def test_prev_trial_before_load(self, gui):
        _, meta = gui
        meta.navigation_widget.prev_trial()
        QApplication.processEvents()
        assert meta.app_state.ready is False

    def test_update_main_plot_before_load(self, gui):
        """Calling update_main_plot before data loaded should not crash."""
        _, meta = gui
        meta.data_widget.update_main_plot()
        QApplication.processEvents()
        assert meta.app_state.ready is False

    def test_label_click_before_load(self, gui):
        """Clicking a label button before data loaded should not crash."""
        _, meta = gui
        try:
            meta.labels_widget.activate_label(1)
        except Exception:
            pass  # acceptable to raise, just shouldn't crash the app
        QApplication.processEvents()


class TestDoubleLoad:

    def test_load_twice_does_not_crash(self, loaded_gui):
        """Loading data a second time should not corrupt state."""
        _, meta = loaded_gui
        assert meta.app_state.ready is True

        # Reload same file
        meta.io_widget.nc_file_path_edit.setText(str(_BIRDPARK_NC))
        meta.app_state.nc_file_path = str(_BIRDPARK_NC)

        meta.data_widget.on_load_clicked()
        QApplication.processEvents()

        assert meta.app_state.ready is True
        assert meta.app_state.dt is not None
        assert len(meta.app_state.trials) > 0
