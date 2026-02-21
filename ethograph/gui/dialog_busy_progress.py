"""Reusable modal progress dialog for long-running computations."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QProgressBar, QProgressDialog


class BusyProgressDialog(QProgressDialog):
    """Modal dialog with indeterminate progress bar and Cancel button.

    Runs a callable in a background worker so the UI stays responsive.
    On completion the bar fills green and the dialog auto-closes.

    Parameters
    ----------
    label : str
        Text shown above the progress bar.
    parent : QWidget, optional
        Parent widget for modality.
    done_delay_ms : int
        How long to display "Done!" before auto-closing (ms).
    use_process : bool
        If True, use ``ProcessPoolExecutor`` instead of
        ``ThreadPoolExecutor``.  Needed for CPU-bound work that
        doesn't release the GIL (e.g. ruptures).

    Usage::

        dialog = BusyProgressDialog("Computing...", parent=self)
        result, error = dialog.execute(my_func, arg1, kwarg=val)
        if dialog.was_cancelled or error:
            return
        use(result)

    For work that touches Qt widgets (cannot be threaded), use
    ``execute_blocking``::

        dialog = BusyProgressDialog("Applying...", parent=self)
        result, error = dialog.execute_blocking(qt_heavy_func)
    """

    _GREEN_CHUNK = "QProgressBar::chunk { background-color: #4CAF50; }"

    def __init__(
        self,
        label: str,
        parent=None,
        done_delay_ms: int = 600,
        use_process: bool = False,
    ):
        super().__init__(label, "Cancel", 0, 0, parent)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.setMinimumWidth(320)

        self._result: Any = None
        self._error: Exception | None = None
        self._done_delay = done_delay_ms
        self._use_process = use_process
        self.was_cancelled = False

        self._executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None
        self._future: Future | None = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll)
        self.canceled.connect(self._on_cancel)

    def execute(self, fn, *args, **kwargs) -> tuple[Any, Exception | None]:
        """Run *fn* in a background worker, blocking via ``exec_()``.

        Returns ``(result, error)``.  If the user cancelled,
        ``self.was_cancelled`` is ``True`` and *result* is ``None``.
        """
        pool_cls = ProcessPoolExecutor if self._use_process else ThreadPoolExecutor
        self._executor = pool_cls(max_workers=1)
        self._future = self._executor.submit(fn, *args, **kwargs)
        self._poll_timer.start()
        self.exec_()
        self._cleanup()
        return self._result, self._error

    def execute_blocking(self, fn, *args, **kwargs) -> tuple[Any, Exception | None]:
        """Run *fn* on the main thread with a visible progress dialog.

        Use this for operations that touch Qt widgets and cannot be
        threaded.  The dialog is shown, ``processEvents`` is called once
        so the dialog paints, and then *fn* runs synchronously.
        The "Done!" indicator is shown afterwards.
        """
        self.show()
        QApplication.processEvents()
        try:
            self._result = fn(*args, **kwargs)
        except Exception as exc:
            self._error = exc
        self._show_done()
        self.exec_()
        return self._result, self._error

    # ------------------------------------------------------------------

    def _poll(self):
        if self._future is None or not self._future.done():
            return
        self._poll_timer.stop()
        try:
            self._result = self._future.result()
        except Exception as exc:
            self._error = exc
        self._show_done()

    def _show_done(self):
        if self._error:
            self.setLabelText(f"Error: {self._error}")
            self.setCancelButtonText("Close")
            return

        self.setLabelText("Done!")
        self.setRange(0, 1)
        self.setValue(1)
        bar = self.findChild(QProgressBar)
        if bar:
            bar.setStyleSheet(self._GREEN_CHUNK)
        QTimer.singleShot(self._done_delay, self.accept)

    def _on_cancel(self):
        self.was_cancelled = True
        self._poll_timer.stop()
        if self._future:
            self._future.cancel()

    def _cleanup(self):
        self._poll_timer.stop()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._future = None
