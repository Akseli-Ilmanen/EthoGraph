"""Verify real mouse-click delivery on time-axis plots.

The other label tests call ``_on_plot_clicked`` directly, which bypasses the
pyqtgraph ``sigMouseClicked`` -> ``_handle_click`` -> ``plot_clicked`` delivery
path. These tests exercise that path with a synthetic Qt mouse event so a
broken connection (clicks never reaching the labels widget) is caught.
"""

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from ethograph.labels.intervals import empty_intervals

pytestmark = pytest.mark.usefixtures("gui")


def _center_point(plot):
    vp = plot.viewport()
    r = vp.rect()
    return vp, QPoint(r.width() // 2, r.height() // 2)


def test_left_click_emits_plot_clicked(moll2025_gui):
    """A real left mouse click on the line plot fires plot_clicked."""
    _viewer, meta = moll2025_gui
    lp = meta.plot_container.line_plots[0]
    lp.show()
    QApplication.processEvents()

    received = []
    lp.plot_clicked.connect(lambda info: received.append(info))

    vp, pt = _center_point(lp)
    QTest.mouseClick(vp, Qt.LeftButton, pos=pt)
    QApplication.processEvents()

    assert received, "plot_clicked did not fire for a real left click on line_plot"
    assert received[-1]["button"] == Qt.LeftButton


def test_right_click_emits_plot_clicked(moll2025_gui):
    """A real right mouse click on the line plot fires plot_clicked."""
    _viewer, meta = moll2025_gui
    lp = meta.plot_container.line_plots[0]
    lp.show()
    QApplication.processEvents()

    received = []
    lp.plot_clicked.connect(lambda info: received.append(info))

    vp, pt = _center_point(lp)
    QTest.mouseClick(vp, Qt.RightButton, pos=pt)
    QApplication.processEvents()

    assert received, "plot_clicked did not fire for a real right click on line_plot"
    assert received[-1]["button"] == Qt.RightButton


def test_right_click_labels_when_armed(moll2025_gui):
    """Arming a label then right-clicking onset/offset creates an interval."""
    _viewer, meta = moll2025_gui
    lw = meta.labels_widget
    meta.app_state.label_intervals = empty_intervals()

    lw.activate_label(1)
    assert lw.ready_for_label_click, "activate_label did not arm label mode"

    lw._on_plot_clicked({"x": 0.5, "button": Qt.RightButton})
    lw._on_plot_clicked({"x": 1.5, "button": Qt.RightButton})

    df = meta.app_state.label_intervals
    assert df is not None and not df.empty, "right-click onset/offset did not create a label"
    assert not lw.ready_for_label_click, "label mode should exit after placing the interval"
