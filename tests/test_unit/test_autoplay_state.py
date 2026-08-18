"""The "Auto-play on navigate" checkbox is a global, persisted preference."""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication, QWidget

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.widgets_navigation import NavigationWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_autoplay_checkbox_is_backed_by_global_state(qapp, tmp_path):
    """The checkbox reads ``autoplay_on_navigate`` at build and writes it back."""
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")  # never touch the real settings
    state.autoplay_on_navigate = True
    widget = NavigationWidget(QWidget(), state)
    try:
        assert widget.autoplay_checkbox.isChecked()
        widget.autoplay_checkbox.setChecked(False)
        assert state.autoplay_on_navigate is False
    finally:
        widget.close()
