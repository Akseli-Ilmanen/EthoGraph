"""Unified notification helpers for the ethograph GUI.

Every user-facing message goes through one of two functions:

- ``notify(msg, severity)``  -- toast overlay + console log
- ``notify_dialog(msg, severity, title, parent)`` -- QMessageBox + console log

Toasts are rendered by :class:`ToastManager`, a lightweight Qt overlay owned by
the main window (registered via :func:`set_toast_host`). Before a host window
exists, ``notify`` only logs.

Set ``SUPPRESS = True`` to disable all popups (used during testing).
"""

from __future__ import annotations

import logging

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QLabel, QMessageBox, QWidget

logger = logging.getLogger(__name__)

SUPPRESS = False

_DIALOG = {
    "error": QMessageBox.critical,
    "warning": QMessageBox.warning,
    "info": QMessageBox.information,
}
_DEFAULT_TITLE = {"error": "Error", "warning": "Warning", "info": "Info"}

_SEVERITY_COLOR = {
    "info": "#4a90d9",
    "warning": "#e0a030",
    "error": "#d94a4a",
}

_TOAST_DURATION_MS = 5000
_TOAST_MARGIN = 12
_TOAST_SPACING = 6


class ToastManager:
    """Stacked, auto-expiring toast labels overlaid on a host window."""

    def __init__(self, host: QWidget):
        self._host = host
        self._toasts: list[QLabel] = []

    def show(self, message: str, severity: str = "info") -> None:
        color = _SEVERITY_COLOR.get(severity, _SEVERITY_COLOR["info"])
        toast = QLabel(message, self._host)
        toast.setWordWrap(True)
        toast.setMaximumWidth(max(300, self._host.width() // 3))
        toast.setAttribute(Qt.WA_TransparentForMouseEvents)
        toast.setStyleSheet(
            "QLabel {"
            "  background: rgba(30, 30, 30, 230);"
            f"  border-left: 4px solid {color};"
            "  color: white;"
            "  padding: 8px 12px;"
            "  border-radius: 4px;"
            "  font-size: 11px;"
            "}"
        )
        toast.adjustSize()
        self._toasts.append(toast)
        self._relayout()
        toast.show()
        toast.raise_()
        QTimer.singleShot(_TOAST_DURATION_MS, lambda: self._remove(toast))

    def _remove(self, toast: QLabel) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
        toast.hide()
        toast.deleteLater()
        self._relayout()

    def _relayout(self) -> None:
        y = self._host.height() - _TOAST_MARGIN
        for toast in reversed(self._toasts):
            y -= toast.height()
            toast.move(self._host.width() - toast.width() - _TOAST_MARGIN, y)
            y -= _TOAST_SPACING


_toast_manager: ToastManager | None = None


def set_toast_host(host: QWidget | None) -> None:
    """Register the main window that toasts are drawn on (None to detach)."""
    global _toast_manager
    _toast_manager = ToastManager(host) if host is not None else None


def notify(message: str, severity: str = "info") -> None:
    """Show a toast notification and log to console."""
    try:
        logger.info("[%s] %s", severity.upper(), message)
        if not SUPPRESS and _toast_manager is not None:
            _toast_manager.show(message, severity)
    except Exception:
        logger.exception("notify failed: %s", message)


def notify_dialog(
    message: str,
    severity: str = "error",
    title: str | None = None,
    parent: object | None = None,
) -> None:
    """Show a modal QMessageBox and log to console."""
    try:
        if severity not in _DIALOG:
            severity = "error"
        title = title or _DEFAULT_TITLE[severity]
        logger.info("[%s] %s", title, message)
        if not SUPPRESS:
            _DIALOG[severity](parent, title, message)
    except Exception:
        logger.exception("notify_dialog failed: %s", message)
