"""Standalone ethograph main window (napari-free shell).

Layout
------
- **Central area** — :class:`~ethograph.gui.video_manager.VideoArea`: primary
  pygfx camera view + optional extra camera stack.
- **Bottom dock** — the synced plots (``UnifiedPanelContainer``).
- **Right dock** — the control sidebar (``MetaWidget``), collapsible via the
  toolbar button or ``Ctrl+0``.
- Extra line-plot panels are added by dragging a feature from the left sidebar.

Layout persistence: geometry, dock state, plot-panel state (which panels are
open, feature view, splitter sizes) and extra line-plot features are saved to
JSON (``Layout → Save layout``), and auto-restored from the session file on
next launch. A ``template_layout.json`` uploaded to a dataset's GitHub release
is applied automatically after that template loads.
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

# Bump when the dock structure changes so stale saved layouts are ignored.
_LAYOUT_VERSION = 2


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

        self._apply_corner_ownership()

        # Video lives in a top dock (added in attach_meta_widget); the plot
        # container becomes the central widget. Do NOT set video_area as the
        # central widget here — setCentralWidget() deletes the previous central
        # widget, which would destroy the CameraView C++ object.
        self.video_area = VideoArea()

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

        # Plots are the central (expanding) widget; the video sits in a compact
        # top dock so it no longer dominates the window (pynaviz-style stack).
        self.setCentralWidget(meta_widget.plot_container)
        self._video_dock = self.add_dock_widget(self.video_area, area="top", name="Video")
        self._video_dock.setObjectName("VideoDock")
        QTimer.singleShot(
            0, lambda: self.resizeDocks([self._video_dock], [300], Qt.Vertical)
        )

        # Bottom playback bar
        from .widgets_bottom_bar import BottomPlaybackBar
        bottom_bar = BottomPlaybackBar(meta_widget.app_state)
        self._bottom_bar_dock = self.add_dock_widget(bottom_bar, area="bottom", name="Playback")
        self._bottom_bar_dock.setObjectName("BottomBarDock")
        self._bottom_bar_dock.setFeatures(
            self._bottom_bar_dock.DockWidgetFeature.DockWidgetMovable
        )
        self.bottom_bar = bottom_bar

        # Left sidebar: draggable Media / Feature sources — full-height, narrow.
        left = getattr(meta_widget, "left_sidebar", None)
        if left is not None:
            self._left_sidebar_dock = self.add_dock_widget(left, area="left", name="Sources")
            self._left_sidebar_dock.setObjectName("LeftSidebarDock")
            left.setMaximumWidth(220)
            QTimer.singleShot(0, lambda: self.resizeDocks(
                [self._left_sidebar_dock], [150], Qt.Horizontal
            ))

        # Wire bottom bar to data_widget and video sync
        data_widget = getattr(meta_widget, "data_widget", None)
        if data_widget is not None and hasattr(self, "bottom_bar"):
            self.bottom_bar.set_data_widget(data_widget)
            # Connect video sync when it becomes available
            def _wire_video_sync():
                if data_widget.video_mgr and hasattr(data_widget.video_mgr, "video_sync"):
                    self.bottom_bar.connect_video_sync(data_widget.video_mgr.video_sync)

            QTimer.singleShot(100, _wire_video_sync)

        # Build the reorganised top menu bar now that the sidebar sections exist.
        from .top_bar import build_menu_bar

        build_menu_bar(self)

        # Clicking the video shows the pose + playback context in the sidebar.
        if hasattr(self.video_area, "clicked") and hasattr(meta_widget, "focus_video_context"):
            self.video_area.clicked.connect(meta_widget.focus_video_context)

        self._sidebar_toggle.setChecked(True)
        # Restore the previous session's window layout once widgets exist.
        QTimer.singleShot(0, self._restore_session_layout)

    def _apply_corner_ownership(self):
        """Left (Sources) and right (control) sidebars span the full window
        height; the video/plots docks sit between them. Re-applied after any
        state restore, which can otherwise reset corner ownership."""
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

    def add_dock_widget(self, widget: QWidget, area: str = "right", name: str = "") -> QDockWidget:
        dock = QDockWidget(name, self)
        dock.setWidget(widget)
        dock.setObjectName(name or widget.__class__.__name__)
        self.addDockWidget(_DOCK_AREAS.get(area, Qt.RightDockWidgetArea), dock)
        return dock

    def _create_menus(self):
        """Create window-level shortcut actions.

        The visible menu bar (File / Layout / Changepoints / Neural / Help) is
        built by :func:`ethograph.gui.top_bar.build_menu_bar` once the sidebar
        sections exist.  Here we only register the standalone actions whose
        keyboard shortcuts must work regardless of the menu bar.
        """
        self._sidebar_toggle = QAction("Show sidebar", self, checkable=True, checked=True)
        self._sidebar_toggle.setShortcut(QKeySequence("Ctrl+0"))
        self._sidebar_toggle.toggled.connect(self._set_sidebar_visible)
        self.addAction(self._sidebar_toggle)

        self._video_toggle = QAction("Show video", self, checkable=True, checked=True)
        self._video_toggle.toggled.connect(self.set_video_viewer_visible)
        self.addAction(self._video_toggle)

        self._zen_action: QAction | None = None

    # ------------------------------------------------------------------
    # Sidebar / video visibility
    # ------------------------------------------------------------------

    def _set_sidebar_visible(self, visible: bool):
        if self._sidebar_dock is not None:
            self._sidebar_dock.setVisible(visible)

    def toggle_sidebar(self):
        self._sidebar_toggle.setChecked(not self._sidebar_toggle.isChecked())

    def set_video_viewer_visible(self, visible: bool):
        # Video now lives in its own top dock.
        dock = getattr(self, "_video_dock", None)
        if dock is not None:
            dock.setVisible(visible)

    def set_zen_mode(self, on: bool):
        """Hide the left/right sidebars (zen mode). Sidebar updates are skipped.

        Toggled from the Layout menu or ``Ctrl+Z``.  While on, the control
        sidebar (and future left layout sidebar) are hidden and
        ``app_state.zen_mode`` is set so widgets can skip expensive refreshes.
        """
        self._zen_mode = on
        if self._sidebar_dock is not None:
            self._sidebar_dock.setVisible(not on)
        left = getattr(self, "_left_sidebar_dock", None)
        if left is not None:
            left.setVisible(not on)
        if self.meta_widget is not None:
            app_state = getattr(self.meta_widget, "app_state", None)
            if app_state is not None and hasattr(app_state, "zen_mode"):
                app_state.zen_mode = on

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
    # Layout persistence
    # ------------------------------------------------------------------

    def _layout_dict(self) -> dict:
        plot_panels = None
        if self.meta_widget is not None:
            plot_panels = self.meta_widget.plot_container.layout_state()
        return {
            "version": _LAYOUT_VERSION,
            "geometry_b64": base64.b64encode(bytes(self.saveGeometry())).decode("ascii"),
            "state_b64": base64.b64encode(bytes(self.saveState(version=_LAYOUT_VERSION))).decode("ascii"),
            "plot_panels": plot_panels,
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

    def restore_layout(self, file_name: str | Path, verbose: bool = True, panels: bool = True):
        """Restore a saved layout.

        ``panels=False`` restores only window geometry/dock state — used for
        the startup session restore, where no dataset is loaded yet (panel
        state comes from the data load or a template layout instead).
        """
        path = Path(file_name)
        if not path.is_file():
            return
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read layout %s: %s", path, e)
            return

        # Ignore layouts saved by an incompatible (older) window structure — they
        # would clobber the new video-top-dock / full-height-sidebar arrangement.
        if payload.get("version") != _LAYOUT_VERSION:
            logger.info("Ignoring stale window layout (version mismatch).")
            return

        plot_panels = payload.get("plot_panels")
        if panels and plot_panels and self.meta_widget is not None:
            # Recreates open panels + extra line plots (needs a loaded dataset).
            self.meta_widget.plot_container.apply_layout_state(plot_panels)

        try:
            self.restoreGeometry(QByteArray.fromBase64(payload["geometry_b64"].encode("ascii")))
            self.restoreState(
                QByteArray.fromBase64(payload["state_b64"].encode("ascii")),
                _LAYOUT_VERSION,
            )
        except (KeyError, Exception) as e:
            logger.warning("Could not restore layout state: %s", e)
        # restoreState can reset corner ownership — re-assert full-height sidebars.
        self._apply_corner_ownership()
        self._sidebar_toggle.setChecked(payload.get("sidebar_visible", True))
        if verbose:
            notify(f"Layout loaded from {path}")

    def _restore_session_layout(self):
        self.restore_layout(_session_layout_path(), verbose=False, panels=False)

    def _save_layout_dialog(self):
        # Default next to the loaded dataset as template_layout.json — upload
        # that file to a dataset's GitHub release to ship a template layout.
        default = Path.cwd() / "ethograph_layout.json"
        nc_path = getattr(getattr(self.meta_widget, "app_state", None), "nc_file_path", None)
        if nc_path:
            default = Path(nc_path).parent / "template_layout.json"
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Layout", str(default), "Layout Files (*.json)"
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
