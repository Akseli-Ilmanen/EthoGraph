"""Reusable modal progress dialog for long-running computations."""

from __future__ import annotations

import traceback
from typing import Any

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QProgressBar,
    QProgressDialog,
)


class BusyProgressDialog(QProgressDialog):
    """Modal dialog with indeterminate progress bar.

    Runs a callable on the main thread so the UI stays visible.
    On completion the bar fills green and the dialog auto-closes.

    Usage::

        dialog = BusyProgressDialog("Computing...", parent=self)
        result, error = dialog.execute(my_func, arg1, kwarg=val)
        if error:
            return
        use(result)
    """

    _GREEN_CHUNK = "QProgressBar::chunk { background-color: #4CAF50; }"

    _SWEEP_STEPS = 50
    _SWEEP_INTERVAL_MS = 40

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
        self.was_cancelled = False

        self._sweep_timer = QTimer(self)
        self._sweep_timer.setInterval(self._SWEEP_INTERVAL_MS)
        self._sweep_timer.timeout.connect(self._sweep_tick)
        self._sweep_pos = 0

    def execute(self, fn, *args, **kwargs) -> tuple[Any, Exception | None]:
        """Run *fn* on the main thread with a visible progress dialog."""
        self._start_sweep()
        self.show()
        QApplication.processEvents()
        self.repaint()
        QApplication.processEvents()
        try:
            self._result = fn(*args, **kwargs)
        except Exception as exc:
            self._error = exc
            traceback.print_exc()
        self._sweep_timer.stop()
        self._show_done()
        self.exec_()
        return self._result, self._error

    # Kept for call-site compatibility — identical to execute().
    def execute_blocking(self, fn, *args, **kwargs) -> tuple[Any, Exception | None]:
        return self.execute(fn, *args, **kwargs)

    def pump_events(self):
        """Process pending Qt events so the sweep bar and label update."""
        QApplication.processEvents()

    # ------------------------------------------------------------------

    def _start_sweep(self):
        self.setRange(0, self._SWEEP_STEPS)
        self.setValue(0)
        bar = self.findChild(QProgressBar)
        if bar:
            bar.setStyleSheet(self._GREEN_CHUNK)
            bar.setTextVisible(False)
        self._sweep_pos = 0
        self._sweep_timer.start()

    def _sweep_tick(self):
        self._sweep_pos = (self._sweep_pos + 1) % (self._SWEEP_STEPS + 1)
        self.setValue(self._sweep_pos)

    def _show_done(self):
        if self._error:
            print(f"Error: {self._error}")
            short = str(self._error)[:120]
            if len(str(self._error)) > 120:
                short += "…"
            self.setLabelText(f"Error: {short}")
            self.setCancelButtonText("Close")
            return

        self.setLabelText("Done!")
        self.setRange(0, 1)
        self.setValue(1)
        bar = self.findChild(QProgressBar)
        if bar:
            bar.setStyleSheet(self._GREEN_CHUNK)
        QTimer.singleShot(self._done_delay, self.accept)
