"""The media/plots separator must be draggable across the whole window.

Qt clamps a dock-separator drag at the minimum size of the widgets on either
side, so this is a test about minimums: the top area (video + space plots) and
the central plot container must both be able to shrink to ~10% of the window.
"""

from __future__ import annotations

from qtpy.QtCore import Qt

from ethograph.gui.app_constants import PANEL_MIN_HEIGHT


def _drag_video_dock_to(shell, qtbot, fraction: float) -> float:
    """Resize the video dock to *fraction* of the window and return the
    fraction it actually got (what a separator drag to that point yields)."""
    height = shell.height()
    shell.resizeDocks([shell._video_dock], [int(height * fraction)], Qt.Vertical)
    qtbot.wait(50)
    return shell._video_dock.height() / height


def test_split_drags_to_both_extremes(gui, qtbot):
    shell, meta = gui
    shell.resize(1200, 1000)
    shell.show()
    qtbot.waitExposed(shell)
    qtbot.wait(50)

    for panel_type in ("audiotrace", "spectrogram", "audiotrace"):
        assert meta.plot_container.add_panel(panel_type) is not None
    qtbot.wait(50)

    assert _drag_video_dock_to(shell, qtbot, 0.9) > 0.85, "plots half blocks the drag"
    assert _drag_video_dock_to(shell, qtbot, 0.1) < 0.15, "media half blocks the drag"


def test_panel_minimums_stay_out_of_the_way(gui, qtbot):
    """No single widget may impose more than a sliver of vertical minimum."""
    shell, meta = gui
    plot = meta.plot_container.add_panel("audiotrace")
    assert plot is not None

    assert meta.plot_container.minimumHeight() <= 100
    assert plot.minimumHeight() == PANEL_MIN_HEIGHT
    assert shell._video_dock.minimumSizeHint().height() <= 100
