"""Unified notification helpers for the ethograph GUI.

Every user-facing message goes through one of two functions:

- ``notify(msg, severity)``  -- napari toast + console log
- ``notify_dialog(msg, severity, title, parent)`` -- QMessageBox + console log
"""

from __future__ import annotations

import logging

from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)

_TOAST = {"info": show_info, "warning": show_warning, "error": show_error}
_DIALOG = {
    "error": QMessageBox.critical,
    "warning": QMessageBox.warning,
    "info": QMessageBox.information,
}
_DEFAULT_TITLE = {"error": "Error", "warning": "Warning", "info": "Info"}


def notify(message: str, severity: str = "info") -> None:
    """Show a napari toast notification and log to console."""
    logger.info("[%s] %s", severity.upper(), message)
    _TOAST[severity](message)


def notify_dialog(
    message: str,
    severity: str = "error",
    title: str | None = None,
    parent: object | None = None,
) -> None:
    """Show a modal QMessageBox and log to console."""
    title = title or _DEFAULT_TITLE[severity]
    logger.info("[%s] %s", title, message)
    _DIALOG[severity](parent, title, message)
