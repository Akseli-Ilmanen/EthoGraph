"""Standalone ethograph main window (napari-free shell).

Layout
------
- **Central area** — :class:`~ethograph.gui.video_manager.VideoArea`: primary
  pygfx camera view + optional extra camera stack.
- **Bottom dock** — the synced plots (``UnifiedPanelContainer``).
- **Right dock** — the control sidebar (``MetaWidget``), collapsible via the
  toolbar button or ``Ctrl+0``.
- Extra line-plot panels can be added from the View menu (pynaviz-style).

Layout persistence: geometry, dock state and extra line-plot features are
saved to JSON (``File → Save layout``), and auto-restored from the session
file on next launch.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import numpy as np
from qtpy.QtCore import QByteArray, Qt, QTimer
from qtpy.QtGui import QAction, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QScrollArea,
    QWidget,
)

from ethograph.utils.paths import default_config_dir

from .notify import notify, set_toast_host
from .video_manager import VideoArea

logger = logging.getLogger(__name__)

_DOCK_AREAS = {
    "left": Qt.LeftDockWidgetArea,
    "right": Qt.RightDockWidgetArea,
    "top": Qt.TopDockWidgetArea,
    "bottom": Qt.BottomDockWidgetArea,
}


def _session_layout_path() -> Path:
    return default_config_dir() / "window_layout.json"


class EthographMainWindow(QMainWindow):
    """Top-level window hosting video, plots, and the control sidebar."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ethograph")
        self.setObjectName("EthographMainWindow")
        self.resize(1400, 900)
        self.setDockNestingEnabled(True)

        # Corner ownership (matches the old napari arrangement):
        # sidebar spans full height on the right, plots extend left.
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)

        self.video_area = VideoArea()
        self.setCentralWidget(self.video_area)

        set_toast_host(self)

        self.meta_widget = None  # set by attach_meta_widget
        self._sidebar_dock: QDockWidget | None = None
        self._plot_dock: QDockWidget | None = None
        self._shortcuts: list[QShortcut] = []
        self._extra_lineplot_count = 0

        self._create_menus()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def attach_meta_widget(self, meta_widget) -> None:
        """Dock the sidebar (MetaWidget) and the plot container."""
        self.meta_widget = meta_widget

        scroll = QScrollArea()
        scroll.setWidget(meta_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._sidebar_dock = self.add_dock_widget(scroll, area="right", name="ethograph GUI")
        self._sidebar_dock.setObjectName("SidebarDock")

        self._plot_dock = self.add_dock_widget(
            meta_widget.plot_container, area="bottom", name="Plots"
        )
        self._plot_dock.setObjectName("PlotsDock")

        self._sidebar_toggle.setChecked(True)
        # Restore the previous session's window layout once widgets exist.
        QTimer.singleShot(0, self._restore_session_layout)

    def add_dock_widget(self, widget: QWidget, area: str = "right", name: str = "") -> QDockWidget:
        dock = QDockWidget(name, self)
        dock.setWidget(widget)
        dock.setObjectName(name or widget.__class__.__name__)
        self.addDockWidget(_DOCK_AREAS.get(area, Qt.RightDockWidgetArea), dock)
        return dock

    def _create_menus(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("&Save layout…", self._save_layout_dialog)
        file_menu.addAction("&Load layout…", self._load_layout_dialog)
        file_menu.addSeparator()
        file_menu.addAction("&Exit", self.close)

        view_menu = menu_bar.addMenu("&View")
        self._sidebar_toggle = QAction("Show &sidebar", self, checkable=True, checked=True)
        self._sidebar_toggle.setShortcut(QKeySequence("Ctrl+0"))
        self._sidebar_toggle.toggled.connect(self._set_sidebar_visible)
        view_menu.addAction(self._sidebar_toggle)

        self._video_toggle = QAction("Show &video", self, checkable=True, checked=True)
        self._video_toggle.toggled.connect(self.set_video_viewer_visible)
        view_menu.addAction(self._video_toggle)

        view_menu.addSeparator()
        view_menu.addAction("Add &line plot", self.add_lineplot_dock)

    # ------------------------------------------------------------------
    # Sidebar / video visibility
    # ------------------------------------------------------------------

    def _set_sidebar_visible(self, visible: bool):
        if self._sidebar_dock is not None:
            self._sidebar_dock.setVisible(visible)

    def toggle_sidebar(self):
        self._sidebar_toggle.setChecked(not self._sidebar_toggle.isChecked())

    def set_video_viewer_visible(self, visible: bool):
        central = self.centralWidget()
        if central is None:
            return
        # Zero the height rather than hide() so dock geometry stays sane.
        central.setMaximumHeight(16777215 if visible else 0)

    # ------------------------------------------------------------------
    # napari-Viewer-replacement services
    # ------------------------------------------------------------------

    def canvas_widget(self) -> QWidget | None:
        """The primary video canvas widget (for Qt overlays)."""
        return self.video_area.primary.canvas_widget() or self.video_area.primary

    def screenshot(self, canvas_only: bool = True, flash: bool = False) -> np.ndarray:
        """Grab the video canvas (or whole window) as an RGBA uint8 array."""
        target = self.video_area if canvas_only else self
        pixmap = target.grab()
        image = pixmap.toImage()
        image = image.convertToFormat(image.Format.Format_RGBA8888)
        w, h = image.width(), image.height()
        ptr = image.constBits()
        try:
            ptr.setsize(h * w * 4)  # PyQt
        except AttributeError:
            pass  # PySide memoryview already sized
        return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()

    def bind_shortcut(self, key_sequence: str, callback) -> QShortcut:
        shortcut = QShortcut(QKeySequence(key_sequence), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)
        return shortcut

    def clear_shortcuts(self):
        for shortcut in self._shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts = []

    # ------------------------------------------------------------------
    # Extra line plots (pynaviz-style multi-panel)
    # ------------------------------------------------------------------

    def add_lineplot_dock(self, feature: str | None = None):
        if self.meta_widget is None:
            return None
        container = self.meta_widget.plot_container
        panel = container.add_extra_lineplot(feature=feature)
        if panel is None:
            notify("Load a dataset before adding line plots.", "warning")
        return panel

    # ------------------------------------------------------------------
    # Layout persistence
    # ------------------------------------------------------------------

    def _layout_dict(self) -> dict:
        extra_features = []
        if self.meta_widget is not None:
            extra_features = self.meta_widget.plot_container.extra_lineplot_features()
        return {
            "version": 1,
            "geometry_b64": base64.b64encode(bytes(self.saveGeometry())).decode("ascii"),
            "state_b64": base64.b64encode(bytes(self.saveState(version=1))).decode("ascii"),
            "extra_lineplots": extra_features,
            "sidebar_visible": bool(self._sidebar_dock and self._sidebar_dock.isVisible()),
        }

    def save_layout(self, file_name: str | Path | None = None, verbose: bool = True):
        path = Path(file_name) if file_name else _session_layout_path()
        path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._layout_dict(), f, indent=2)
        if verbose:
            notify(f"Layout saved to {path}")

    def restore_layout(self, file_name: str | Path, verbose: bool = True):
        path = Path(file_name)
        if not path.is_file():
            return
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read layout %s: %s", path, e)
            return

        for feature in payload.get("extra_lineplots", []):
            self.add_lineplot_dock(feature=feature)

        try:
            self.restoreGeometry(QByteArray.fromBase64(payload["geometry_b64"].encode("ascii")))
            self.restoreState(
                QByteArray.fromBase64(payload["state_b64"].encode("ascii")),
                payload.get("version", 1),
            )
        except (KeyError, Exception) as e:
            logger.warning("Could not restore layout state: %s", e)
        self._sidebar_toggle.setChecked(payload.get("sidebar_visible", True))
        if verbose:
            notify(f"Layout loaded from {path}")

    def _restore_session_layout(self):
        self.restore_layout(_session_layout_path(), verbose=False)

    def _save_layout_dialog(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Layout", str(Path.cwd() / "ethograph_layout.json"), "Layout Files (*.json)"
        )
        if file_name:
            self.save_layout(file_name)

    def _load_layout_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Layout", "", "Layout Files (*.json)")
        if file_name:
            self.restore_layout(file_name)

    # ------------------------------------------------------------------
    # Close handling
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.meta_widget is not None:
            if not self.meta_widget._check_unsaved_changes(event):
                return
            self.save_layout(_session_layout_path(), verbose=False)
            if hasattr(self.meta_widget.app_state, "stop_auto_save"):
                self.meta_widget.app_state.stop_auto_save()
            data_widget = getattr(self.meta_widget, "data_widget", None)
            if data_widget is not None and getattr(data_widget, "video_mgr", None) is not None:
                data_widget.video_mgr.cleanup()
        set_toast_host(None)
        super().closeEvent(event)
