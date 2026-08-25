"""Closing the window releases the caches a loaded session filled.

Qt sends ``closeEvent`` only to the window, so a cache cleared in a child
widget's ``closeEvent`` is never cleared at all. That leaked a decoded
``AudioLoader`` per dataset load (~80 MB for a 20 MB wav), and a process that
loaded a dozen sessions died with a native access violation on the next large
HDF5 allocation.
"""

from __future__ import annotations

from ethograph.gui.plots_spectrogram import SharedAudioCache


class _Sentinel:
    """Stands in for a cached loader so the test needs no audio file."""


def test_window_close_clears_the_audio_cache(gui):
    shell, meta = gui
    SharedAudioCache._instances["sentinel.wav"] = _Sentinel()
    assert SharedAudioCache._instances

    shell.close()

    assert not SharedAudioCache._instances, "closing the window left the audio cache filled"


def test_data_widget_exposes_cleanup_to_the_window(gui):
    """The window drives the teardown; the child cannot do it on its own."""
    _shell, meta = gui
    assert callable(meta.data_widget.cleanup)
    SharedAudioCache._instances["sentinel.wav"] = _Sentinel()
    meta.data_widget.cleanup()
    assert not SharedAudioCache._instances


def test_close_event_is_not_delivered_to_child_widgets(qapp):
    """The premise the fix rests on — if Qt ever changes this, say so loudly."""
    from qtpy.QtWidgets import QMainWindow, QWidget

    seen = []

    class _Child(QWidget):
        def closeEvent(self, event):  # pragma: no cover - must not run
            seen.append("child")
            super().closeEvent(event)

    win = QMainWindow()
    child = _Child(win)
    win.setCentralWidget(child)
    win.close()
    assert seen == [], "Qt now delivers closeEvent to children; DataWidget.cleanup() can move back"
