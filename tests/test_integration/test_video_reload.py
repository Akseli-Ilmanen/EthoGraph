"""Reloading the video that is already shown must reuse the pynaviz decoder.

Each ``PlotVideo`` owns a spawned worker process that attaches to the parent's
shared-memory frame buffer. Closing one and building another within a couple of
seconds — what a trial change or a camera re-apply used to do — races that
attach on Windows: the worker needs ~1.5-2 s to re-import its modules while
``PlotVideo.close()`` waits only ``join(timeout=2)`` before dropping the
parent's handle, which is what destroys the mapping. The loser dies with
``FileNotFoundError: [WinError 2] ... 'wnsm_…'``.
"""

from __future__ import annotations


def test_same_video_reload_keeps_the_decoder(birdpark_gui, qtbot):
    _, meta = birdpark_gui
    view = meta.data_widget.video_mgr.primary_view
    assert view.has_video
    plot = view.plot
    pid = plot._worker.pid
    hook = plot._update_extra_objects

    meta.data_widget.update_video()
    qtbot.wait(50)

    assert view.plot is plot, "same file reload rebuilt the plot"
    assert view.plot._worker.pid == pid, "same file reload respawned the decoder"
    assert view.plot._worker.is_alive()
    # The overlay hook wraps _update_extra_objects — re-wrapping it on every
    # reload would stack closures and update the overlay N times per frame.
    assert view.plot._update_extra_objects is hook


def test_reused_view_re_clips_and_drops_the_overlay(birdpark_gui, qtbot):
    _, meta = birdpark_gui
    view = meta.data_widget.video_mgr.primary_view
    plot = view.plot
    assert view.ensure_overlay() is not None
    total = plot.data.shape[0]

    view.set_video(view._video_path, fps=view.fps, time_offset=0.25, start_frame=1, end_frame=total - 1)

    assert view.plot is plot
    assert view.start_frame == 1
    assert view.n_frames == total - 2
    assert view.time_offset == 0.25
    assert view.overlay is None, "a reload must start from a clean pose overlay"


def test_clear_forces_a_fresh_decoder(birdpark_gui, qtbot):
    _, meta = birdpark_gui
    view = meta.data_widget.video_mgr.primary_view
    pid = view.plot._worker.pid

    meta.data_widget.video_mgr._cleanup_primary_video()
    assert not view.has_video

    meta.data_widget.update_video()
    qtbot.wait(50)

    assert view.has_video
    assert view.plot._worker.pid != pid
