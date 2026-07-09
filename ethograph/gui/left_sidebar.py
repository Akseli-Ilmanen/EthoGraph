"""Left sidebar: drag Media / Feature sources onto the plot area to make panels.

The sidebar lists two groups:

* **Media** — one entry per video camera and per audio microphone.
* **Features** — one entry per catalog feature.

Dragging an entry onto the plot container (:class:`UnifiedPanelContainer`)
drops a ``kind|name`` payload; the container calls back into
:meth:`MetaWidget._on_source_dropped`, which opens :class:`PlotTypePicker`
(navigable with ↑/↓) to choose the plot type and then creates the panel.

Feature plot-type options are gated by data shape ``(T, N)``:

* ``Lineplot`` — always available.
* ``Heatmap`` / ``Space (2D)`` — need ``N >= 2``.
* ``Space (3D)`` — needs ``N >= 3``.
"""

from __future__ import annotations

import logging

from qtpy.QtCore import QMimeData, Qt
from qtpy.QtGui import QDrag
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QLabel,
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
        drag.exec_(Qt.CopyAction)


class LeftSidebar(QWidget):
    """Dockable list of draggable Media + Feature sources."""

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        hint = QLabel("Drag a source onto the plot area to add a panel.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: rgba(255,255,255,140); font-size: 8pt;")
        layout.addWidget(hint)

        self._list = _SourceList()
        layout.addWidget(self._list)

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

    def refresh(self):
        """Repopulate from the current session (cameras, mics, features)."""
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
        ds = getattr(self.app_state, "ds", None)
        features: list[str] = []
        catalog = getattr(self.app_state, "catalog", None)
        if catalog is not None and getattr(catalog, "features", None):
            features = list(catalog.features)
        elif ds is not None:
            features = list(ds.data_vars)
        for feat in features:
            self._add_source(feat, "feature", str(feat))
