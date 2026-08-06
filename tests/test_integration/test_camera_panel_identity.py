"""Every camera view names the camera it shows — the primary included.

The primary used to sit in a dock titled "Video" with no ``camera_name`` at
all, so it was the one panel the user could not identify, and the only one
whose clip/offset came from ``trial_alignment`` rather than from its own
stream. Switching the primary camera left that alignment stale, which showed
up as the "Video" panel drifting out of sync with the cam-N panels.
"""

from __future__ import annotations

from ethograph.gui.video_manager import camera_dock_title


def test_camera_dock_title_names_camera_and_file():
    assert camera_dock_title("cam-1", r"C:\data\front_view.mp4") == "cam-1 (front_view.mp4)"
    assert camera_dock_title("cam-1", "https://host/x/clip.mp4") == "cam-1 (clip.mp4)"
    assert camera_dock_title("cam-1", None) == "cam-1"


def test_primary_view_carries_its_camera_and_title(birdpark_gui):
    shell, meta = birdpark_gui
    vm = meta.data_widget.video_mgr
    view = vm.primary_view
    assert view.has_video

    camera = meta.app_state.primary_camera
    assert view.camera_name == camera, "the primary must name its camera like any extra view"
    assert shell._video_dock.windowTitle() == camera_dock_title(camera, view.source_video_path)
    assert "Video" != shell._video_dock.windowTitle()


def test_primary_camera_switch_rebuilds_the_alignment(birdpark_gui, qtbot):
    _, meta = birdpark_gui
    dw = meta.data_widget
    combo = dw.primary_camera_combo
    if combo.count() < 2:
        return  # single-camera dataset: nothing to switch to

    other = next(combo.itemText(i) for i in range(combo.count()) if combo.itemText(i) != combo.currentText())
    combo.setCurrentText(other)
    qtbot.wait(50)

    sio = meta.app_state.nwb_alignment
    expected = sio.stream_offset_for_trial(meta.app_state.trials_sel, "video", other)
    assert dw.video_mgr.primary_view.camera_name == other
    assert meta.app_state.trial_alignment.video_offset == expected
