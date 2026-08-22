"""The label-name overlay drawn on the video is a video display setting, so
its "Hide label" checkbox lives in the sidebar's video context."""

from qtpy.QtWidgets import QApplication

from ethograph.gui.right_context import _CONTEXT_MAP


def test_hide_label_sits_in_the_video_context(moll2025_gui):
    _, meta = moll2025_gui
    cb = meta.labels_widget.hide_label_cb
    gb = meta.data_widget.videolabel_groupbox

    assert cb.parent() is gb
    assert "videolabel" in _CONTEXT_MAP["video"]
    assert cb not in meta.data_widget.overlays_groupbox.findChildren(type(cb))

    meta.context_panel.set_context("video")
    QApplication.processEvents()
    assert gb.isVisibleTo(meta.context_panel)
    assert cb.isVisibleTo(meta.context_panel)
