"""Closing the primary video dock tears the video down like an extra's close.

The shell's VideoDock previously had no close handling at all: its ✕ merely
hid the dock while the view kept a live plot, decode worker and canvas — and
re-adding "Video (cam)" from the popup then forked an *extra* view over the
invisible primary.
"""

from __future__ import annotations

import pytest
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QWidget

from ethograph.gui.video_manager import VideoArea, VideoManager


class _FakeAppState:
    video = None
    ready = False


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _make(qapp):
    area = VideoArea()
    manager = VideoManager(area, _FakeAppState())
    return area, manager


def test_primary_dock_close_triggers_teardown(qapp):
    area, manager = _make(qapp)
    torn_down = []
    manager.close_primary_video = lambda: torn_down.append(True)
    area.primary_close_requested.disconnect()
    area.primary_close_requested.connect(manager.close_primary_video)

    dock = QWidget()
    dock._is_primary_video_dock = True
    area.eventFilter(dock, QEvent(QEvent.Close))
    qapp.processEvents()  # the teardown is deferred out of the close event
    assert torn_down


def test_extra_dock_close_does_not_touch_the_primary(qapp):
    area, manager = _make(qapp)
    requested = []
    area.primary_close_requested.connect(lambda: requested.append(True))

    dock = QWidget()
    dock._camera_key = "cam-2"
    area.eventFilter(dock, QEvent(QEvent.Close))
    qapp.processEvents()
    assert not requested


def test_close_primary_video_unloads_and_hides_the_dock(qapp):
    area, manager = _make(qapp)

    class _Shell:
        visible = None

        def set_video_dock_visible(self, visible):
            self.visible = visible

    shell = _Shell()
    area.shell = shell
    cleaned = []
    manager._cleanup_primary_video = lambda: cleaned.append(True)
    manager.close_primary_video()
    assert cleaned
    assert shell.visible is False
