"""Test DataPanel toggle buttons — Main / Pose / Audio panels."""

import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture
def data_panel(app_state, qtbot):
    from ethograph.gui.widgets_data import DataPanel
    panel = DataPanel(app_state)
    qtbot.addWidget(panel)
    panel.show()
    QApplication.processEvents()
    return panel


class TestDataPanelToggles:

    def test_initial_state_main_visible(self, data_panel):
        assert data_panel.main_panel.isVisible()
        assert not data_panel.pose_panel.isVisible()
        assert not data_panel.audio_panel.isVisible()
        assert data_panel.main_toggle.isChecked()
        assert not data_panel.pose_toggle.isChecked()
        assert not data_panel.audio_toggle.isChecked()

    def test_click_pose_shows_pose(self, data_panel, qtbot):
        qtbot.mouseClick(data_panel.pose_toggle, 1)  # Qt.LeftButton = 1
        QApplication.processEvents()

        assert not data_panel.main_panel.isVisible()
        assert data_panel.pose_panel.isVisible()
        assert not data_panel.audio_panel.isVisible()
        assert not data_panel.main_toggle.isChecked()
        assert data_panel.pose_toggle.isChecked()
        assert not data_panel.audio_toggle.isChecked()

    def test_click_audio_shows_audio(self, data_panel, qtbot):
        qtbot.mouseClick(data_panel.audio_toggle, 1)
        QApplication.processEvents()

        assert not data_panel.main_panel.isVisible()
        assert not data_panel.pose_panel.isVisible()
        assert data_panel.audio_panel.isVisible()

    def test_click_active_toggle_stays_active(self, data_panel, qtbot):
        """Clicking the already-active toggle should not hide everything."""
        qtbot.mouseClick(data_panel.main_toggle, 1)
        QApplication.processEvents()

        assert data_panel.main_panel.isVisible()
        assert data_panel.main_toggle.isChecked()

    def test_switch_main_to_pose_to_audio(self, data_panel, qtbot):
        qtbot.mouseClick(data_panel.pose_toggle, 1)
        QApplication.processEvents()
        assert data_panel.pose_panel.isVisible()

        qtbot.mouseClick(data_panel.audio_toggle, 1)
        QApplication.processEvents()
        assert data_panel.audio_panel.isVisible()
        assert not data_panel.pose_panel.isVisible()

        qtbot.mouseClick(data_panel.main_toggle, 1)
        QApplication.processEvents()
        assert data_panel.main_panel.isVisible()
        assert not data_panel.audio_panel.isVisible()

    def test_show_panel_programmatic(self, data_panel):
        data_panel._show_panel("audio")
        QApplication.processEvents()

        assert data_panel.audio_panel.isVisible()
        assert not data_panel.main_panel.isVisible()
        assert not data_panel.pose_panel.isVisible()
        assert data_panel.audio_toggle.isChecked()
