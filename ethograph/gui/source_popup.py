"""Add-panel popup: a transient, searchable list of Media / Feature sources.

Opened from the "➕ Add panel" button in the bottom playback bar (or Ctrl+N).
The popup overlays the plot area and vanishes on focus-out. Two workflows:

* **Drag** an entry onto the plot container (:class:`UnifiedPanelContainer`)
  — drops a ``kind|name`` payload; the container calls back into
  :meth:`MetaWidget._on_source_dropped`, which opens :class:`PlotTypePicker`
  (navigable with ↑/↓) to choose the plot type and then creates the panel.
* **Enter / double-click** an entry — same flow, but the panel is created at
  its default location; the user arranges it afterwards by dragging the
  panel's dock title bar.

The filter box has keyboard focus on open; ↑/↓ move the list selection while
typing. Feature plot-type options are gated by data shape ``(T, N)``:

* ``Lineplot`` — always available.
* ``Heatmap`` / ``Space (2D)`` — need ``N >= 2``.
* ``Space (3D)`` — needs ``N >= 3``.
"""

from __future__ import annotations

import logging

from qtpy.QtCore import QMimeData, QPoint, Qt
from qtpy.QtGui import QDrag
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

SOURCE_MIME = "application/x-ethograph-source"

_ROLE_KIND = Qt.UserRole
_ROLE_NAME = Qt.UserRole + 1
_ROLE_HEADER = Qt.UserRole + 2


def feature_ncols(app_state, feature: str) -> int:
    """Number of non-time columns ``N`` for a feature (product of extra dims)."""
    ds = getattr(app_state, "ds", None)
    if ds is None or feature not in getattr(ds, "data_vars", {}):
        return 1
    da = ds[feature]
    n = 1
    for dim, size in zip(da.dims, da.shape):
        if "time" not in str(dim).lower():
            n *= int(size)
    return max(1, n)


def allowed_plot_types(kind: str, name: str, app_state) -> list[str]:
    """Plot types offered for a dropped source, gated by data shape."""
    if kind == "audio":
        return ["Audio Trace", "Spectrogram Trace"]
    if kind == "video":
        return ["Video"]
    if kind == "feature":
        options = ["Lineplot"]
        n = feature_ncols(app_state, name)
        if n >= 2:
            options += ["Heatmap", "Space (2D)"]
        if n >= 3:
            options.append("Space (3D)")
        return options
    return []


class PlotTypePicker(QDialog):
    """Tiny modal list to choose a plot type; ↑/↓ to move, Enter to accept."""

    def __init__(self, options: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot type")
        self.setModal(True)
        self.choice: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._list = QListWidget()
        self._list.addItems(options)
        if options:
            self._list.setCurrentRow(0)
        self._list.itemActivated.connect(self._accept_item)
        self._list.itemDoubleClicked.connect(self._accept_item)
        layout.addWidget(self._list)
        self._list.setFocus()
        self.resize(200, 30 + 22 * max(1, len(options)))

    def _accept_item(self, item):
        self.choice = item.text()
        self.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            item = self._list.currentItem()
            if item is not None:
                self._accept_item(item)
            return
        super().keyPressEvent(event)


class _SourceList(QListWidget):
    """A list whose data items (not headers) are draggable as sources."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def startDrag(self, _actions):
        item = self.currentItem()
        if item is None or item.data(_ROLE_HEADER):
            return
        kind = item.data(_ROLE_KIND)
        name = item.data(_ROLE_NAME)
        mime = QMimeData()
        mime.setData(SOURCE_MIME, f"{kind}|{name}".encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        # Hide the popup so the plot area is unobstructed while dragging.
        self.window().hide()
        drag.exec_(Qt.CopyAction)


class SourcePopup(QWidget):
    """Transient searchable list of Media + Feature sources.

    ``on_activate(kind, name)`` (set by MetaWidget) is called when an entry
    is chosen with Enter or a double-click; dragging out uses the normal
    drag-and-drop path into the plot container.
    """

    _WIDTH = 230
    _MAX_HEIGHT = 420
    _ROW_HEIGHT = 20

    def __init__(self, app_state, parent=None):
        super().__init__(parent, Qt.Popup)
        self.app_state = app_state
        self.on_activate = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        self._filter.installEventFilter(self)
        layout.addWidget(self._filter)

        self._list = _SourceList()
        self._list.itemActivated.connect(self._activate_item)
        self._list.itemDoubleClicked.connect(self._activate_item)
        layout.addWidget(self._list)

        hint = QLabel("Drag onto the plot area, or press Enter.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: rgba(255,255,255,140); font-size: 8pt;")
        layout.addWidget(hint)

    # ------------------------------------------------------------------
    # Population (same catalog-driven list the old left sidebar showed)
    # ------------------------------------------------------------------

    def _add_header(self, text: str):
        item = QListWidgetItem(text)
        item.setData(_ROLE_HEADER, True)
        item.setFlags(Qt.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self._list.addItem(item)

    def _add_source(self, label: str, kind: str, name: str):
        item = QListWidgetItem(f"  {label}")
        item.setData(_ROLE_HEADER, False)
        item.setData(_ROLE_KIND, kind)
        item.setData(_ROLE_NAME, name)
        self._list.addItem(item)

    def refresh(self, catalog=None):
        """Repopulate from the current session (cameras, mics, features).

        *catalog* is the loaded DataCatalog: its ``feature_choices()`` is the
        canonical feature list (same one the features combo displays), so
        every offered feature is representable in the sidebar controls.
        """
        self._list.clear()
        sio = getattr(self.app_state, "nwb_alignment", None)
        cameras = list(getattr(sio, "cameras", []) or []) if sio else []
        mics = list(getattr(sio, "mics", []) or []) if sio else []

        self._add_header("Media")
        for cam in cameras:
            self._add_source(f"Video ({cam})", "video", str(cam))
        for mic in mics:
            self._add_source(f"Audio ({mic})", "audio", str(mic))

        self._add_header("Features")
        features: list[str] = catalog.feature_choices() if catalog is not None else []
        if not features:
            ds = getattr(self.app_state, "ds", None)
            if ds is not None:
                features = list(ds.data_vars)
        for feat in features:
            self._add_source(feat, "feature", str(feat))

        self._apply_filter(self._filter.text())

    # ------------------------------------------------------------------
    # Filtering + keyboard navigation
    # ------------------------------------------------------------------

    def _apply_filter(self, text: str):
        """Hide non-matching data items; hide headers whose group is empty."""
        needle = text.strip().lower()
        header_item = None
        header_has_match = False
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(_ROLE_HEADER):
                if header_item is not None:
                    header_item.setHidden(not header_has_match)
                header_item, header_has_match = item, False
                continue
            match = not needle or needle in item.text().lower()
            item.setHidden(not match)
            header_has_match = header_has_match or match
        if header_item is not None:
            header_item.setHidden(not header_has_match)
        self._select_first_visible()

    def _visible_data_rows(self) -> list[int]:
        return [
            i
            for i in range(self._list.count())
            if not self._list.item(i).isHidden() and not self._list.item(i).data(_ROLE_HEADER)
        ]

    def _select_first_visible(self):
        rows = self._visible_data_rows()
        self._list.setCurrentRow(rows[0] if rows else -1)

    def _move_selection(self, delta: int):
        rows = self._visible_data_rows()
        if not rows:
            return
        current = self._list.currentRow()
        pos = rows.index(current) if current in rows else 0
        self._list.setCurrentRow(rows[(pos + delta) % len(rows)])

    def eventFilter(self, obj, event):
        """↑/↓/Enter in the filter box drive the list selection."""
        if obj is self._filter and event.type() == event.Type.KeyPress:
            key = event.key()
            if key == Qt.Key_Down:
                self._move_selection(+1)
                return True
            if key == Qt.Key_Up:
                self._move_selection(-1)
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                item = self._list.currentItem()
                if item is not None:
                    self._activate_item(item)
                return True
        return super().eventFilter(obj, event)

    def _activate_item(self, item):
        if item is None or item.data(_ROLE_HEADER):
            return
        kind = item.data(_ROLE_KIND)
        name = item.data(_ROLE_NAME)
        self.hide()
        if callable(self.on_activate):
            self.on_activate(kind, name)

    # ------------------------------------------------------------------
    # Showing
    # ------------------------------------------------------------------

    def popup_at(self, global_pos: QPoint, open_upward: bool = False):
        """Show the popup at *global_pos* (top-left; bottom-left if upward)."""
        n_rows = self._list.count()
        height = min(self._MAX_HEIGHT, 80 + self._ROW_HEIGHT * max(3, n_rows))
        self.resize(self._WIDTH, height)
        pos = global_pos - QPoint(0, height) if open_upward else global_pos
        self.move(pos)
        self._filter.clear()
        self.show()
        self._filter.setFocus()
        self._select_first_visible()
