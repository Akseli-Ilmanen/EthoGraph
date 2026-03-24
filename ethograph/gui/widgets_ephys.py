"""Ephys widget — trace controls, Kilosort neuron jumping."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from napari.utils.notifications import show_info, show_warning
from napari.viewer import Viewer
from scipy.ndimage import gaussian_filter1d
from qtpy.QtCore import Qt, QItemSelectionModel, QRect, QRectF, QSortFilterProxyModel, Signal
from qtpy.QtGui import QBrush, QColor, QPen, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import pynapple as nap

from ethograph.features.neural import build_tsgroup, compute_pca, firing_rate_to_xarray

from .app_constants import CLUSTER_TABLE_MAX_HEIGHT, CLUSTER_TABLE_ROW_HEIGHT
from .makepretty import find_combo_index, get_combo_value, set_combo_to_value
from .plots_ephystrace import GenericEphysLoader, get_loader as get_ephys_loader
from .plots_timeseriessource import RegularTimeseriesSource

_CLUSTER_COLORS = [
    (228, 26, 28),    # red
    (55, 126, 184),   # blue
    (77, 175, 74),    # green
    (152, 78, 163),   # purple
    (255, 127, 0),    # orange
    (255, 255, 51),   # yellow
    (166, 86, 40),    # brown
    (247, 129, 191),  # pink
    (153, 153, 153),  # grey
    (0, 210, 213),    # cyan
    (180, 210, 36),   # lime
    (240, 60, 100),   # magenta
    (100, 180, 255),  # sky
    (200, 130, 0),    # amber
    (100, 220, 150),  # mint
    (180, 100, 220),  # lavender
    (220, 180, 100),  # sand
    (100, 140, 80),   # olive
    (220, 100, 100),  # coral
    (80, 180, 180),   # teal
]

_RAWIO_TO_DISPLAY = {
    "IntanRawIO": "Intan",
    "OpenEphysBinaryRawIO": "OpenEphys",
    "OpenEphysRawIO": "OpenEphys",
    "NWBIO": "NWB",
    "BlackrockRawIO": "Blackrock",
    "AxonRawIO": "Axon",
    "AxographRawIO": "Axograph",
    "EDFRawIO": "EDF",
    "BrainVisionRawIO": "BrainVision",
    "Spike2RawIO": "Spike2",
    "NeuralynxRawIO": "Neuralynx",
    "MicromedRawIO": "Micromed",
    "PlexonRawIO": "Plexon",
    "Plexon2RawIO": "Plexon2",
    "SpikeGadgetsRawIO": "SpikeGadgets",
    "SpikeGLXRawIO": "SpikeGLX",
    "MedRawIO": "MED",
    "WinEdrRawIO": "WinEDR",
    "WinWcpRawIO": "WinWCP",
    "NeuroNexusRawIO": "NeuroNexus",
    "TdtRawIO": "TDT",
}


_PROBE_COLOR_SELECTED = QColor(0x00, 0xBB, 0xFF)
_PROBE_COLOR_UNSELECTED = QColor(140, 140, 140)
_PROBE_DOT_SIZE = 12
_PROBE_LABEL_FONT = pg.Qt.QtGui.QFont("monospace", 7)


class _RectangleSelector:
    """Rubber-band rectangle drawn on a pyqtgraph ViewBox."""

    def __init__(self, view_box: pg.ViewBox, on_release):
        self._vb = view_box
        self._on_release = on_release
        self._rect_item: pg.QtWidgets.QGraphicsRectItem | None = None
        self._origin = None
        view_box.scene().sigMouseClicked.connect(self._noop)
        view_box.mouseDragEvent = self._drag_event

    @staticmethod
    def _noop(evt):
        pass

    def _drag_event(self, evt):
        evt.accept()
        pos = evt.pos()
        if evt.isStart():
            self._origin = self._vb.mapToView(pos)
            if self._rect_item is not None:
                self._vb.removeItem(self._rect_item)
            self._rect_item = pg.QtWidgets.QGraphicsRectItem()
            self._rect_item.setPen(QPen(QColor(255, 255, 100, 200), 1))
            self._rect_item.setBrush(QBrush(QColor(255, 255, 100, 40)))
            self._vb.addItem(self._rect_item, ignoreBounds=True)
        elif evt.isFinish():
            end = self._vb.mapToView(pos)
            rect = QRectF(self._origin, end).normalized()
            if self._rect_item is not None:
                self._vb.removeItem(self._rect_item)
                self._rect_item = None
            self._on_release(rect)
        else:
            if self._origin is not None:
                current = self._vb.mapToView(pos)
                r = QRectF(self._origin, current).normalized()
                self._rect_item.setRect(r)


class ProbeChannelDialog(QDialog):
    """Interactive probe map for selecting channels by spatial position."""

    def __init__(
        self,
        channel_positions: np.ndarray,
        channel_map: np.ndarray | None,
        hw_names: dict[int, str] | None,
        selected_hw: set[int] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select channels on probe")
        self.resize(800, 600)

        self._positions = channel_positions
        self._channel_map = channel_map if channel_map is not None else np.arange(len(channel_positions))
        self._hw_names = hw_names or {}
        self._n_sites = len(self._channel_map)
        self._selected: set[int] = set(selected_hw) if selected_hw else set()

        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Channel", "Selected"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self._populate_table()
        left_layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(self._select_all)
        deselect_all_btn = QPushButton("Deselect all")
        deselect_all_btn.clicked.connect(self._deselect_all)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(deselect_all_btn)
        left_layout.addLayout(btn_row)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Drag a rectangle to select channels:"))
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setAspectLocked(True)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel('bottom', 'X position')
        self._plot_widget.setLabel('left', 'Y position (depth)')
        right_layout.addWidget(self._plot_widget)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._scatter = None
        self._text_items: list[pg.TextItem] = []
        self._draw_probe()
        vb = self._plot_widget.getPlotItem().getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        self._selector = _RectangleSelector(vb, self._on_rect_select)

    def _channel_label(self, idx: int) -> str:
        hw_ch = int(self._channel_map[idx])
        name = self._hw_names.get(hw_ch)
        if name and name != f"Ch {hw_ch}":
            return f"{name} ({hw_ch})"
        return f"Ch {hw_ch}"

    def _populate_table(self):
        self._table.setRowCount(self._n_sites)
        for i in range(self._n_sites):
            label_item = QTableWidgetItem(self._channel_label(i))
            self._table.setItem(i, 0, label_item)
            status_item = QTableWidgetItem()
            self._table.setItem(i, 1, status_item)
        self._update_table_colors()

    def _update_table_colors(self):
        for i in range(self._n_sites):
            is_sel = i in self._selected
            color = _PROBE_COLOR_SELECTED if is_sel else _PROBE_COLOR_UNSELECTED
            for col in range(2):
                item = self._table.item(i, col)
                if item:
                    item.setBackground(QBrush(color))
            status = self._table.item(i, 1)
            if status:
                status.setText("Yes" if is_sel else "")

    def _draw_probe(self):
        x = self._positions[:, 0] if len(self._positions) > 0 else np.array([])
        y = self._positions[:, 1] if len(self._positions) > 0 else np.array([])

        brushes = []
        for i in range(self._n_sites):
            if i in self._selected:
                brushes.append(pg.mkBrush(_PROBE_COLOR_SELECTED))
            else:
                brushes.append(pg.mkBrush(_PROBE_COLOR_UNSELECTED))

        if self._scatter is not None:
            self._plot_widget.removeItem(self._scatter)

        self._scatter = pg.ScatterPlotItem(
            x=x, y=y,
            size=_PROBE_DOT_SIZE,
            pen=pg.mkPen(color='w', width=0.5),
            brush=brushes,
            hoverable=True,
            hoverSize=_PROBE_DOT_SIZE + 4,
            tip=lambda x, y, data: "",
        )
        self._scatter.sigClicked.connect(self._on_dot_clicked)
        self._plot_widget.addItem(self._scatter)

        for item in self._text_items:
            self._plot_widget.removeItem(item)
        self._text_items.clear()

        for i in range(self._n_sites):
            hw_ch = int(self._channel_map[i])
            ti = pg.TextItem(
                text=str(hw_ch),
                color='w',
                anchor=(0.5, 0.5),
                fill=pg.mkBrush(0, 0, 0, 160),
            )
            ti.setFont(_PROBE_LABEL_FONT)
            ti.setPos(float(x[i]), float(y[i]))
            self._plot_widget.addItem(ti)
            self._text_items.append(ti)

        self._plot_widget.autoRange()

    def _on_dot_clicked(self, _scatter, points, _ev):
        for pt in points:
            idx = pt.index()
            if idx in self._selected:
                self._selected.discard(idx)
            else:
                self._selected.add(idx)
        self._refresh()

    def _on_rect_select(self, rect: QRectF):
        for i in range(self._n_sites):
            px = self._positions[i, 0]
            py = self._positions[i, 1]
            if rect.contains(px, py):
                self._selected.add(i)
        self._refresh()

    def _refresh(self):
        self._update_table_colors()
        brushes = []
        for i in range(self._n_sites):
            if i in self._selected:
                brushes.append(pg.mkBrush(_PROBE_COLOR_SELECTED))
            else:
                brushes.append(pg.mkBrush(_PROBE_COLOR_UNSELECTED))
        if self._scatter is not None:
            self._scatter.setBrush(brushes)

    def _select_all(self):
        self._selected = set(range(self._n_sites))
        self._refresh()

    def _deselect_all(self):
        self._selected.clear()
        self._refresh()

    def get_selected_hw_channels(self) -> np.ndarray | None:
        if not self._selected:
            return None
        y_coords = self._positions[:, 1]
        selected_list = sorted(self._selected, key=lambda i: -y_coords[i])
        return self._channel_map[selected_list].astype(int)


class ParamsDialog(QDialog):
    """Dialog to collect params.py fields when the file is missing or dat_path is invalid."""

    def __init__(self, parent=None, defaults: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Ephys params.py configuration")
        self.resize(500, 260)
        defaults = defaults or {}

        layout = QVBoxLayout(self)

        form = QVBoxLayout()

        # dat_path
        dat_row = QHBoxLayout()
        dat_row.addWidget(QLabel("dat_path:"))
        self.dat_path_edit = QLineEdit(defaults.get("dat_path", ""))
        dat_row.addWidget(self.dat_path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dat)
        dat_row.addWidget(browse_btn)
        form.addLayout(dat_row)

        # n_channels_dat
        nch_row = QHBoxLayout()
        nch_row.addWidget(QLabel("n_channels_dat:"))
        self.n_channels_spin = QSpinBox()
        self.n_channels_spin.setRange(1, 10000)
        self.n_channels_spin.setValue(defaults.get("n_channels_dat", 64))
        nch_row.addWidget(self.n_channels_spin)
        nch_row.addStretch()
        form.addLayout(nch_row)

        # sample_rate
        sr_row = QHBoxLayout()
        sr_row.addWidget(QLabel("sample_rate:"))
        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(1.0, 1_000_000.0)
        self.sample_rate_spin.setDecimals(1)
        self.sample_rate_spin.setValue(defaults.get("sample_rate", 30000.0))
        sr_row.addWidget(self.sample_rate_spin)
        sr_row.addStretch()
        form.addLayout(sr_row)

        # offset
        off_row = QHBoxLayout()
        off_row.addWidget(QLabel("offset:"))
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(0, 1_000_000)
        self.offset_spin.setValue(defaults.get("offset", 0))
        off_row.addWidget(self.offset_spin)
        off_row.addStretch()
        form.addLayout(off_row)

        # dtype
        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel("dtype:"))
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["int16", "float32", "float64", "int32", "uint16"])
        self.dtype_combo.setCurrentText(defaults.get("dtype", "int16"))
        dt_row.addWidget(self.dtype_combo)
        dt_row.addStretch()
        form.addLayout(dt_row)

        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _browse_dat(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select raw ephys data file", "",
            "Raw data (*.dat *.bin *.raw);;All files (*)",
        )
        if path:
            self.dat_path_edit.setText(path)

    def get_params(self) -> dict:
        return {
            "dat_path": self.dat_path_edit.text().strip(),
            "n_channels_dat": self.n_channels_spin.value(),
            "sample_rate": self.sample_rate_spin.value(),
            "offset": self.offset_spin.value(),
            "dtype": self.dtype_combo.currentText(),
        }


def _write_params_py(folder: Path, params: dict):
    """Write a params.py file to the given folder."""
    lines = [
        f"dat_path = r'{params.get('dat_path', '')}'",
        f"n_channels_dat = {params.get('n_channels_dat', 64)}",
        f"dtype = '{params.get('dtype', 'int16')}'",
        f"offset = {params.get('offset', 0)}",
        f"sample_rate = {params.get('sample_rate', 30000.0)}",
        f"hp_filtered = False",
    ]
    (folder / "params.py").write_text("\n".join(lines) + "\n")


_SORT_ROLE = Qt.UserRole + 1
_COLOR_ROLE = Qt.UserRole + 2


class _MultiColumnFilterProxy(QSortFilterProxyModel):
    """Proxy that filters rows by categorical or numeric criteria on multiple columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cat_filters: dict[int, set[str]] = {}
        self._num_filters: dict[int, tuple[str, float]] = {}
        self._visible_channels: set[int] | None = None
        self._ch_col: int | None = None

    def set_cat_filter(self, col: int, allowed: set[str]):
        if not allowed:
            self._cat_filters.pop(col, None)
        else:
            self._cat_filters[col] = allowed
        self.invalidateFilter()

    def set_numeric_filter(self, col: int, op: str | None, value: float | None):
        if op is None or value is None:
            self._num_filters.pop(col, None)
        else:
            self._num_filters[col] = (op, value)
        self.invalidateFilter()

    def set_visible_channel_filter(self, channels: set[int] | None, ch_col: int | None):
        self._visible_channels = channels
        self._ch_col = ch_col
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent):
        model = self.sourceModel()
        for col, allowed in self._cat_filters.items():
            item = model.item(source_row, col)
            if item is None:
                return False
            if item.text() not in allowed:
                return False
        for col, (op, threshold) in self._num_filters.items():
            item = model.item(source_row, col)
            if item is None:
                return False
            try:
                val = float(item.data(_SORT_ROLE))
            except (ValueError, TypeError):
                return False
            if op == ">=" and val < threshold:
                return False
            if op == "<=" and val > threshold:
                return False
        if self._visible_channels is not None and self._ch_col is not None:
            item = model.item(source_row, self._ch_col)
            if item is None:
                return False
            try:
                ch = int(float(item.data(_SORT_ROLE)))
            except (ValueError, TypeError):
                return False
            if ch not in self._visible_channels:
                return False
        return True

    def lessThan(self, left, right):
        left_val = left.data(_SORT_ROLE)
        right_val = right.data(_SORT_ROLE)
        if left_val is not None and right_val is not None:
            return float(left_val) < float(right_val)
        return super().lessThan(left, right)


class _CatFilterDialog(QDialog):
    """Checkbox popup for categorical column filtering."""

    def __init__(self, col: int, all_values: list[str], active: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter")
        self._col = col
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(8, 8, 8, 8)
        self._all_cb = QCheckBox("(All)")
        self._all_cb.setChecked(not active)
        layout.addWidget(self._all_cb)
        self._checks: list[tuple[str, QCheckBox]] = []
        for val in sorted(all_values):
            cb = QCheckBox(val)
            cb.setChecked(not active or val in active)
            layout.addWidget(cb)
            self._checks.append((val, cb))
        self._all_cb.toggled.connect(self._on_all)
        for _, cb in self._checks:
            cb.toggled.connect(self._on_item)
        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def _on_all(self, checked: bool):
        for _, cb in self._checks:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

    def _on_item(self, _):
        self._all_cb.blockSignals(True)
        self._all_cb.setChecked(all(cb.isChecked() for _, cb in self._checks))
        self._all_cb.blockSignals(False)

    def get_allowed(self) -> set[str]:
        checked = {v for v, cb in self._checks if cb.isChecked()}
        return set() if checked == {v for v, _ in self._checks} else checked


class _NumFilterDialog(QDialog):
    """Threshold filter dialog for numeric columns."""

    def __init__(self, col: int, current: tuple[str, float] | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter")
        self._col = col
        self._cleared = False
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        op_row = QHBoxLayout()
        self._op_combo = QComboBox()
        self._op_combo.addItems(["\u2265", "\u2264"])
        op_row.addWidget(self._op_combo)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(-1e9, 1e9)
        self._spin.setDecimals(3)
        op_row.addWidget(self._spin)
        layout.addLayout(op_row)
        if current:
            op, val = current
            self._op_combo.setCurrentText("\u2265" if op == ">=" else "\u2264")
            self._spin.setValue(val)
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        clear_btn = QPushButton("Remove filter")
        clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

    def _clear(self):
        self._cleared = True
        self.accept()

    def get_filter(self) -> tuple[str, float] | None:
        if self._cleared:
            return None
        return (">=" if self._op_combo.currentText() == "\u2265" else "<=", self._spin.value())


class _FilterHeaderView(QHeaderView):
    """Column header that draws filter icons for filterable columns.

    A dedicated zone of width _FILTER_ZONE_W is reserved on the right side of
    each filterable column.  Clicking anywhere in that zone triggers the filter
    dialog; clicking elsewhere triggers the normal sort.
    """

    filter_requested = Signal(int)
    _FILTER_ZONE_W = 20  # px reserved on the right of filterable columns

    def __init__(self, cat_cols: set[int], num_cols: set[int], parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._cat_cols = cat_cols
        self._num_cols = num_cols
        self._active: set[int] = set()
        self.setSectionsClickable(True)

    def set_filterable(self, cat_cols: set[int], num_cols: set[int]):
        self._cat_cols = cat_cols
        self._num_cols = num_cols
        self.viewport().update()

    def set_active_filters(self, active: set[int]):
        self._active = active
        self.viewport().update()

    @property
    def _all_filterable(self) -> set[int]:
        return self._cat_cols | self._num_cols

    def _filter_zone_x(self, logical: int) -> int:
        """Left edge of the filter zone for *logical* column."""
        return self.sectionViewportPosition(logical) + self.sectionSize(logical) - self._FILTER_ZONE_W

    def _icon_rect(self, logical: int) -> QRect:
        s = 11
        zone_x = self._filter_zone_x(logical)
        h = self.height()
        x = zone_x + (self._FILTER_ZONE_W - s) // 2
        return QRect(x, (h - s) // 2, s, s)

    def paintSection(self, painter, rect, logical):
        painter.save()
        super().paintSection(painter, rect, logical)
        painter.restore()
        if logical not in self._all_filterable:
            return
        # Subtle separator at the start of the filter zone
        zone_x = self._filter_zone_x(logical)
        painter.save()
        painter.setPen(QPen(QColor(120, 120, 120, 80), 1))
        painter.drawLine(zone_x, rect.top() + 3, zone_x, rect.bottom() - 3)
        painter.restore()
        # Filter icon (funnel) centred in the zone
        ir = self._icon_rect(logical)
        x, y, s = ir.x(), ir.y(), ir.width()
        color = QColor(255, 215, 0) if logical in self._active else QColor(180, 180, 180)
        painter.save()
        painter.setPen(QPen(color, 1.5))
        painter.drawLine(x, y, x + s, y)
        painter.drawLine(x + 2, y + 4, x + s - 2, y + 4)
        painter.drawLine(x + 4, y + 8, x + s - 4, y + 8)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            logical = self.logicalIndexAt(event.pos())
            if logical in self._all_filterable:
                if event.pos().x() >= self._filter_zone_x(logical):
                    self.filter_requested.emit(logical)
                    return
        super().mousePressEvent(event)


class _ClusterIdDelegate(QStyledItemDelegate):
    """Paints cluster_id cells with a solid background color stored in _COLOR_ROLE.

    This bypasses the QSS `QTableView::item` rules that would otherwise ignore
    the model's BackgroundRole.
    """

    def paint(self, painter, option, index):
        color: QColor | None = index.data(_COLOR_ROLE)
        if color is not None:
            painter.save()
            painter.fillRect(option.rect, QBrush(color))
            if option.state & 0x0002:  # QStyle.State_Selected
                painter.fillRect(option.rect, QBrush(QColor(255, 255, 255, 50)))
            r, g, b = color.red(), color.green(), color.blue()
            text_color = QColor(0, 0, 0) if (r * 0.299 + g * 0.587 + b * 0.114) > 150 else QColor(255, 255, 255)
            painter.setPen(text_color)
            text = str(index.data(Qt.DisplayRole) or "")
            painter.drawText(option.rect.adjusted(4, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
            painter.restore()
        else:
            super().paint(painter, option, index)


class EphysWidget(QWidget):
    """Ephys controls with toggle-button tabs: Ephys trace | Neuron jumping."""

    cluster_selected = Signal(int)   # emitted when a single cluster row is selected

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None
        self.meta_widget = None
        self.data_widget = None
        self.io_widget = None

        self._cluster_df: pd.DataFrame | None = None
        self._spike_clusters: np.ndarray | None = None
        self._spike_samples: np.ndarray | None = None   # raw integer Kilosort sample indices
        self._spike_times_s: np.ndarray | None = None   # float64 seconds, derived on load
        self._channel_positions: np.ndarray | None = None
        self._channel_map: np.ndarray | None = None
        self._probe_channel_order: np.ndarray | None = None
        self._custom_channel_set: np.ndarray | None = None
        self._templates: np.ndarray | None = None
        self._ephys_n_channels = 0
        self._tsgroup = None
        self._current_cluster_id_for_psth: int | None = None
        self._psth_dialog = None
        self._kilosort_sr: float | None = None
        self._fr_cache_key: tuple | None = None
        self._kilosort_params: dict | None = None
        self._phy_reader = None
        self._phy_sr: float | None = None
        self._phy_n_channels: int | None = None

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_toggle_buttons(main_layout)
        self._create_traceview_panel(main_layout)
        self._create_firing_rate_panel(main_layout)

        self._show_panel("traceview")
        self.setEnabled(False)

    # ------------------------------------------------------------------
    # Toggle buttons
    # ------------------------------------------------------------------

    def _create_toggle_buttons(self, main_layout):
        toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_widget.setLayout(toggle_layout)

        toggle_defs = [
            ("traceview_toggle", "TraceView", self._toggle_traceview),
            ("firing_rate_toggle", "Firing rates", self._toggle_firing_rate),
        ]
        for attr, label, callback in toggle_defs:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(callback)
            toggle_layout.addWidget(btn)
            setattr(self, attr, btn)

        main_layout.addWidget(toggle_widget)

    def _show_panel(self, panel_name: str):
        panels = {
            "traceview": (self.traceview_panel, self.traceview_toggle),
            "firing_rate": (self.firing_rate_panel, self.firing_rate_toggle),
        }
        for name, (panel, toggle) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)
        self._refresh_layout()

    def _toggle_traceview(self):
        self._show_panel("traceview" if self.traceview_toggle.isChecked() else "firing_rate")

    def _toggle_firing_rate(self):
        self._show_panel("firing_rate" if self.firing_rate_toggle.isChecked() else "traceview")

    def _open_psth(self):
        from .widgets_psth import PSTHDialog

        if self._psth_dialog is None or not self._psth_dialog.isVisible():
            nav = getattr(self.data_widget, "navigation_widget", None)
            labels_w = getattr(self.data_widget, "labels_widget", None) if self.data_widget else None
            self._psth_dialog = PSTHDialog(
                self.app_state, self, labels_w, nav, parent=self
            )
            self._psth_dialog.trial_jump_requested.connect(self._on_psth_trial_jump)

        self._show_panel("traceview")
        self._psth_dialog.show()
        self._psth_dialog.raise_()
        self._psth_dialog.activateWindow()

    def _on_psth_trial_jump(self, trial_id: str):
        nav = getattr(self.data_widget, "navigation_widget", None)
        if nav is not None:
            nav.navigate_to_trial(trial_id)

    def _refresh_layout(self):
        if self.meta_widget:
            self.meta_widget.refresh_widget_layout(self)

    # ------------------------------------------------------------------
    # Ephys trace panel (channel,  gain, range)
    # ------------------------------------------------------------------

    def _create_traceview_panel(self, main_layout):
        self.traceview_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.traceview_panel.setLayout(layout)

        group = QGroupBox("Ephys trace controls")
        group_layout = QVBoxLayout()
        group.setLayout(group_layout)
        layout.addWidget(group)


        # Channel spinbox
        self.ephys_channel_label = QLabel("Ephys channel:")
        self.ephys_channel_spin = QSpinBox()
        self.ephys_channel_spin.setObjectName("ephys_channel_spin")
        self.ephys_channel_spin.setRange(0, 0)
        self.ephys_channel_spin.setPrefix("Ch ")
        self.ephys_channel_spin.setToolTip("Select ephys channel to display")
        self.ephys_channel_spin.valueChanged.connect(self._on_ephys_channel_changed)

        # Gain spinbox
        self.ephys_gain_label = QLabel("Gain:")
        self.ephys_gain_spin = QDoubleSpinBox()
        self.ephys_gain_spin.setObjectName("ephys_gain_spin")
        self.ephys_gain_spin.setRange(-10.0, 10.0)
        self.ephys_gain_spin.setSingleStep(0.1)
        self.ephys_gain_spin.setDecimals(1)
        self.ephys_gain_spin.setValue(0.0)
        self.ephys_gain_spin.setToolTip(
            "Display gain: negative = amplify, positive = attenuate (Ctrl+Wheel)"
        )
        self.ephys_gain_spin.valueChanged.connect(self._on_ephys_gain_changed)

        self.ephys_auto_gain_cb = QCheckBox("Auto gain")
        self.ephys_auto_gain_cb.setToolTip(
            "Quantile-based auto-scaling (Phy method: 1st/99th percentile after median subtraction)"
        )
        self.ephys_auto_gain_cb.setChecked(True)
        self.ephys_auto_gain_cb.toggled.connect(self._on_auto_gain_toggled)

        ch_row = QHBoxLayout()
        ch_row.addWidget(self.ephys_gain_label)
        ch_row.addWidget(self.ephys_gain_spin)
        ch_row.addWidget(self.ephys_auto_gain_cb)
        ch_row.addStretch()
        group_layout.addLayout(ch_row)

        self._probe_row = QWidget()
        probe_row_layout = QHBoxLayout()
        probe_row_layout.setContentsMargins(0, 0, 0, 0)
        probe_row_layout.setSpacing(5)
        self._probe_row.setLayout(probe_row_layout)

        self.probe_select_btn = QPushButton("Select channels on probe")
        self.probe_select_btn.setToolTip("Open probe map to select channels by spatial position")
        self.probe_select_btn.clicked.connect(self._open_probe_channel_dialog)
        probe_row_layout.addWidget(self.probe_select_btn)

        probe_row_layout.addWidget(QLabel("N closest:"))
        self.n_closest_spin = QSpinBox()
        self.n_closest_spin.setRange(1, 384)
        self.n_closest_spin.setValue(5)
        self.n_closest_spin.setToolTip("Number of spatially closest channels for waveform display. Only works for non split/merged clusters.")
        self.n_closest_spin.valueChanged.connect(self._update_highlight_label)
        probe_row_layout.addWidget(self.n_closest_spin)
        probe_row_layout.addStretch()

        self._probe_row.hide()
        group_layout.addWidget(self._probe_row)

        _sel_row = QHBoxLayout()
        _sel_row.setSpacing(4)
        _sel_row.setContentsMargins(0, 0, 0, 0)

        self._highlight_label = QLabel()
        self._update_highlight_label()
        _sel_row.addWidget(self._highlight_label)


        self._select_visible_btn = QPushButton("All visible rows")
        self._select_visible_btn.setToolTip(
            "Highlight all rows currently visible after filtering, then disable auto-highlight"
        )
        self._select_visible_btn.clicked.connect(self._select_clusters_all_visible)
        _sel_row.addWidget(self._select_visible_btn)

        self._unselect_btn = QPushButton("Unselect")
        self._unselect_btn.setToolTip("Clear spike overlays and table selection")
        self._unselect_btn.clicked.connect(self._unselect_clusters)
        _sel_row.addWidget(self._unselect_btn)

        group_layout.addLayout(_sel_row)

        self._multi_cluster_colors: dict[int, tuple] = {}

        self._cluster_model = QStandardItemModel()
        self._cluster_proxy = _MultiColumnFilterProxy()
        self._cluster_proxy.setSourceModel(self._cluster_model)

        self.cluster_table = QTableView()
        self.cluster_table.setModel(self._cluster_proxy)
        self.cluster_table.verticalHeader().setVisible(False)
        self.cluster_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cluster_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cluster_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cluster_table.setSortingEnabled(True)
        self.cluster_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cluster_table.verticalHeader().setDefaultSectionSize(CLUSTER_TABLE_ROW_HEIGHT)
        self.cluster_table.setMaximumHeight(CLUSTER_TABLE_MAX_HEIGHT)

        self._filter_col_cats: dict[int, list[str]] = {}
        self._filter_cat_active: dict[int, set[str]] = {}
        self._filter_num_active: dict[int, tuple[str, float]] = {}
        self._filter_cat_cols: set[int] = set()
        self._filter_num_cols: set[int] = set()

        self._cluster_id_delegate = _ClusterIdDelegate(self.cluster_table)

        self._filter_header = _FilterHeaderView(set(), set())
        self._filter_header.setDefaultSectionSize(40)
        self._filter_header.setMinimumSectionSize(20)
        self._filter_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self._filter_header.setStretchLastSection(False)
        self._filter_header.filter_requested.connect(self._on_filter_header_clicked)
        self.cluster_table.setHorizontalHeader(self._filter_header)
        self.cluster_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.cluster_table.setStyleSheet("""
            QTableView { gridline-color: transparent; background: #444; color: #fff; }
            QTableView::item { padding: 0px 2px; color: #fff; }
            QTableView::item:selected { background: #3a5070; color: #fff; }
            QHeaderView::section {
                padding: 2px 4px;
                background: #888;
                color: #fff;
                border: none;
                border-right: 1px solid #666;
                font-family: sans-serif;
                font-size: 11px;
            }
            QHeaderView::section:last { border-right: none; }
        """)
        self.cluster_table.selectionModel().selectionChanged.connect(self._on_cluster_row_selected)

        cluster_table_header = QLabel("Cluster Table")
        cluster_table_header.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #aaa; padding: 2px 0px 0px 2px;"
        )
        layout.addWidget(cluster_table_header)
        layout.addWidget(self.cluster_table)

        psth_btn = QPushButton("Open interactive PSTH →")
        psth_btn.setToolTip("Open PSTH popup aligned to labels or trial events.")
        psth_btn.clicked.connect(self._open_psth)
        layout.addWidget(psth_btn)

        main_layout.addWidget(self.traceview_panel)

    # ------------------------------------------------------------------
    # Ephys trace handlers
    # ------------------------------------------------------------------


    def _resolve_phy_loader(self) -> tuple:
        """Return (loader, channel_idx) for the Phy-Viewer panel, or (None, 0).

        Resolution order:
        1. Kilosort .dat reader (_phy_reader) — most specific, set on kilosort load.
        2. Trial alignment — authoritative when session table has ephys media.
        3. Generic loader with kilosort params — when _phy_reader wasn't set but params exist.
        4. Generic loader — any other Neo-supported ephys file.
        """
        if self._phy_reader is not None:
            return self._phy_reader, 0

        ephys_path, stream_id, channel_idx = self.app_state.get_ephys_source()
        if not ephys_path:
            return None, 0

        if (
            self._kilosort_params
            and Path(ephys_path).suffix.lower() in {".dat", ".bin", ".raw"}
        ):
            n_ch = self._kilosort_params.get("n_channels_dat")
            if n_ch is None and self._channel_map is not None:
                n_ch = int(self._channel_map.max()) + 1
            if n_ch is None:
                return None, 0
            return get_ephys_loader(
                ephys_path, stream_id,
                n_channels=n_ch,
                sampling_rate=self._kilosort_params.get("sample_rate"),
            ), channel_idx

        return get_ephys_loader(ephys_path, stream_id), channel_idx

    def configure_ephys_trace_plot(self):
        loader, channel_idx = self._resolve_phy_loader()

        if loader is None:
            return

        self.plot_container.ephys_trace_plot.set_loader(loader, channel_idx)
        ephys_source = RegularTimeseriesSource("ephys", loader, start_time=0.0)
        self.plot_container.ephys_trace_plot.set_source(ephys_source)
        self.plot_container.raster_plot.set_source(ephys_source)

        n_ch = loader.n_channels
        self._ephys_n_channels = n_ch
        self.ephys_gain_label.show()
        self.ephys_gain_spin.show()

        if self.plot_container.is_ephystrace():
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

        if self._tsgroup is not None:
            self._populate_raster_all_spikes()

    def _on_ephys_channel_changed(self, channel: int):
        stream_sel = getattr(self.app_state, 'ephys_stream_sel', None)
        source_map = getattr(self.app_state, 'ephys_source_map', {})
        if stream_sel not in source_map:
            return
        filename, stream_id, _ = source_map[stream_sel]
        source_map[stream_sel] = (filename, stream_id, channel)

        if self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.set_channel(channel)
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

    def set_neural_view(self, mode: str):
        """Switch between '1-ch Trace', 'Multi Trace', 'Raster'."""
        if not self.plot_container:
            return

        ephys_plot = self.plot_container.ephys_trace_plot



        if mode == "Multi Trace":
            self.plot_container.set_neural_panel_mode("trace")
            ephys_plot.auto_channel_spacing()
            if self.ephys_auto_gain_cb.isChecked():
                self._apply_auto_gain()
            ephys_plot.autoscale()
            self.ephys_channel_spin.setEnabled(False)

        elif mode == "Raster":
            self.ephys_channel_spin.setEnabled(False)
            ephys_plot.auto_channel_spacing()
            self.plot_container.set_neural_panel_mode("raster")

        if self.data_widget:
            xmin, xmax = self.plot_container.get_current_xlim()
            self.data_widget.update_main_plot(t0=xmin, t1=xmax)

    def _on_ephys_gain_changed(self, value: float):
        if self.ephys_auto_gain_cb.isChecked():
            self.ephys_auto_gain_cb.blockSignals(True)
            self.ephys_auto_gain_cb.setChecked(False)
            self.ephys_auto_gain_cb.blockSignals(False)
        if self.plot_container and self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.buffer.display_gain = value
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

    def _on_auto_gain_toggled(self, checked: bool):
        if checked:
            self._apply_auto_gain()

    def _apply_auto_gain(self):
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return
        ephys_plot = self.plot_container.ephys_trace_plot
        new_gain = ephys_plot.auto_gain()
        self.ephys_gain_spin.blockSignals(True)
        self.ephys_gain_spin.setValue(new_gain)
        self.ephys_gain_spin.blockSignals(False)

    def _open_probe_channel_dialog(self):
        if self._channel_positions is None or self._channel_map is None:
            show_warning("No channel positions loaded — load a Kilosort folder first.")
            return

        current_selected = None
        if self._custom_channel_set is not None:
            idx_set = set()
            for hw in self._custom_channel_set:
                matches = np.where(self._channel_map == hw)[0]
                if len(matches):
                    idx_set.add(int(matches[0]))
            current_selected = idx_set

        hw_names = self.get_hw_names(self._channel_map)
        dialog = ProbeChannelDialog(
            self._channel_positions,
            self._channel_map,
            hw_names,
            selected_hw=current_selected,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        hw_channels = dialog.get_selected_hw_channels()
        if hw_channels is None or len(hw_channels) == 0:
            self._custom_channel_set = None
            if self.plot_container and self.plot_container.is_ephystrace():
                self.plot_container.ephys_trace_plot.set_custom_channel_set(None)
            self._apply_probe_channel_filter()
            return

        self._custom_channel_set = hw_channels
        self._apply_probe_channel_filter()

        if self.data_widget and hasattr(self.data_widget, 'neural_view_combo'):
            if self.data_widget.neural_view_combo.currentText() != "Multi Trace":
                self.data_widget.neural_view_combo.setCurrentText("Multi Trace")

        if self.plot_container and self.plot_container.is_ephystrace():
            ephys_plot = self.plot_container.ephys_trace_plot
            ephys_plot.set_custom_channel_set(hw_channels)
            ephys_plot.auto_channel_spacing()
            if self.ephys_auto_gain_cb.isChecked():
                self._apply_auto_gain()
            ephys_plot.autoscale()

    def hide_ephys_channel_controls(self):
        self.ephys_channel_label.hide()
        self.ephys_channel_spin.hide()
        self.ephys_gain_label.hide()
        self.ephys_gain_spin.hide()


    # ------------------------------------------------------------------
    # Neuron jumping panel (Kilosort)
    # ------------------------------------------------------------------

    def populate_ephys_default_path(self):
        ks_path = self.io_widget.kilosort_folder_edit.text().strip()
        if ks_path and Path(ks_path).is_dir():
            self._load_kilosort_folder()

    def _browse_kilosort_folder(self):
        ephys_path = getattr(self.app_state, 'ephys_path', '')
        ephys_dir = str(Path(ephys_path).parent) if ephys_path else ''
        start_dir = getattr(self.app_state, 'kilosort_folder', '') or ephys_dir or ''
        folder = QFileDialog.getExistingDirectory(
            self, "Select kilosort4 output folder", start_dir,
        )
        if folder:
            self.io_widget.kilosort_folder_edit.setText(folder)
            self.app_state.kilosort_folder = folder
            self._load_kilosort_folder()

    def _parse_kilosort_params(self, folder: Path) -> dict | None:
        params_file = folder / "params.py"
        if not params_file.exists():
            return None
        try:
            namespace = {}
            exec(params_file.read_text(), namespace)
            sr = namespace.get("sample_rate")
            if sr is None:
                return None
            result = {"sample_rate": float(sr)}
            n_ch = namespace.get("n_channels_dat")
            if n_ch is not None:
                result["n_channels_dat"] = int(n_ch)
            dat_path = namespace.get("dat_path")
            if dat_path is not None:
                result["dat_path"] = str(dat_path)
            result["dtype"] = str(namespace.get("dtype", "int16"))
            return result
        except Exception as e:
            show_warning(f"Failed to parse {params_file.name}: {e}")
            return None

    def _validate_kilosort_sr(self, kilosort_sr: float) -> bool:
        loader = self._get_any_ephys_loader()
        if loader is None:
            return True
        ephys_sr = loader.rate
        if abs(kilosort_sr - ephys_sr) > 1.0:
            show_warning(
                f"Sample rate mismatch: Kilosort params.py says {kilosort_sr:.0f} Hz "
                f"but ephys loader reports {ephys_sr:.0f} Hz. Check your data."
            )
            return False
        return True

    def _load_kilosort_folder(self):
        path_str = self.io_widget.kilosort_folder_edit.text().strip()
        if not path_str:
            return

        folder = Path(path_str)
        if not folder.is_dir():
            show_warning(f"Folder not found: {folder}")
            return

        self.app_state.kilosort_folder = path_str

        required_files = [
            "spike_times.npy",
            "spike_clusters.npy",
            "channel_positions.npy",
            "channel_map.npy",
        ]
        missing = [f for f in required_files if not (folder / f).exists()]
        if missing:
            show_warning(
                f"Kilosort folder is missing required files:\n"
                + "\n".join(f"  - {f}" for f in missing)
            )
            return

        ks_params = self._parse_kilosort_params(folder)
        if ks_params is None:
            # No params.py — prompt user to create one
            dialog = ParamsDialog(self)
            if dialog.exec_() != QDialog.Accepted:
                return
            ks_params = dialog.get_params()
            _write_params_py(folder, ks_params)
            show_info(f"Saved params.py to {folder}")

        # Validate / fix dat_path
        dat_path_str = ks_params.get("dat_path", "")
        if dat_path_str and not Path(dat_path_str).is_file():
            show_warning(
                f"dat_path not found on this machine:\n{dat_path_str}\n\n"
                "Please update the path to the raw data file."
            )
            dialog = ParamsDialog(self, defaults=ks_params)
            if dialog.exec_() != QDialog.Accepted:
                return
            ks_params = dialog.get_params()
            _write_params_py(folder, ks_params)
            show_info(f"Updated params.py in {folder}")

        ks_sr = ks_params.get("sample_rate")
        if ks_sr is None:
            show_warning("No sample_rate in params — cannot load kilosort folder.")
            return
        ks_sr = float(ks_sr)
        if not self._validate_kilosort_sr(ks_sr):
            return
        self._kilosort_sr = ks_sr
        self._kilosort_params = ks_params

        cluster_info_path = folder / "cluster_info.tsv"
        if cluster_info_path.exists():
            self._cluster_df = self._load_file(cluster_info_path, pd.read_csv, sep='\t')
        else:
            show_warning("No cluster_info.tsv found — cluster table will be empty.")
            self._cluster_df = None



        self._spike_clusters = self._load_file(folder / "spike_clusters.npy", np.load, flatten=True)
        self._spike_samples = self._load_file(folder / "spike_times.npy", np.load, flatten=True)
        if self._spike_samples is not None and self._spike_clusters is not None:
            self._spike_times_s = self._spike_samples.astype(np.float64) / ks_sr
            self._tsgroup = build_tsgroup(self._spike_times_s, self._spike_clusters)
        self._channel_positions = self._load_file(folder / "channel_positions.npy", np.load)
        self._channel_map = self._load_file(folder / "channel_map.npy", np.load, flatten=True)
        self._templates = self._load_file(folder / "templates.npy", np.load)
        
        self._reorder_probe_by_position()

        if self._cluster_df is not None:
            self._populate_cluster_table(self._cluster_df)


        if self._channel_positions is not None and self._channel_map is not None:
            self._probe_row.show()

        self._register_dat_fallback(folder)
        if self._phy_reader is not None and self.plot_container:
            self.plot_container.set_ephys_visible(True)
            self.configure_ephys_trace_plot()

        if self._tsgroup is not None:
            self._register_kilosort_features()
            self._populate_raster_all_spikes()
 
        # Show Phy-Viewer checkbox and update Neo stream greying
        if self.data_widget:
            phy_cb = getattr(self.data_widget, 'phy_viewer_checkbox', None)
            if phy_cb:
                phy_cb.show()
                phy_cb.setChecked(True)
            # Re-populate Neo combo to grey out matching streams
            if hasattr(self.data_widget, '_populate_neo_stream_combo'):
                self.data_widget._populate_neo_stream_combo()

    def _trial_ep(self) -> nap.IntervalSet | None:
        alignment = self.app_state.trial_alignment
        ephys_offset = alignment.ephys_offset if alignment is not None else 0.0
        trial_bounds = self.app_state.trial_bounds
        if trial_bounds is None:
            return None
        return nap.IntervalSet(ephys_offset, ephys_offset + trial_bounds.duration)

    def _ephys_offset(self) -> float:
        alignment = self.app_state.trial_alignment
        return alignment.ephys_offset if alignment is not None else 0.0

    def _restrict_to_trial(self, cluster_id: int, sr: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (times_local_s, samples_abs) for cluster_id restricted to current trial."""
        trial_ep = self._trial_ep()
        if trial_ep is None or self._tsgroup is None or cluster_id not in self._tsgroup:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
        offset = self._ephys_offset()
        times_global = self._tsgroup[cluster_id].restrict(trial_ep).times()
        times_local = times_global - offset
        samples_abs = np.round(times_global * sr).astype(np.int64)
        return times_local, samples_abs

    def _populate_raster_all_spikes(self):
        if not self.plot_container or self._tsgroup is None:
            return

        ephys_plot = self.plot_container.ephys_trace_plot
        sr = ephys_plot.buffer.ephys_sr
        if sr is None or sr <= 0:
            return

        trial_ep = self._trial_ep()
        offset = self._ephys_offset()
        if trial_ep is None:
            return

        trial_tsg = self._tsgroup.restrict(trial_ep)
        best_ch_map = self._build_cluster_best_channel_map()

        times_list, channels_list = [], []
        for cid, ts in trial_tsg.items():
            t = ts.times() - offset
            if len(t):
                times_list.append(t)
                channels_list.append(np.full(len(t), best_ch_map.get(int(cid), 0), dtype=np.int32))

        raster = self.plot_container.raster_plot

        all_ch = ephys_plot._all_ordered_channels()
        total = len(all_ch)
        if total > 0:
            spacing = ephys_plot.buffer.channel_spacing
            hw_to_y = {int(hw): (total - 1 - i) * spacing for i, hw in enumerate(all_ch)}
            raster.sync_y_axis(hw_to_y, spacing, total)

        if times_list:
            all_times = np.concatenate(times_list)
            all_channels = np.concatenate(channels_list)
            order = np.argsort(all_times)
            raster.set_spike_data(all_times[order], all_channels[order])

    def _build_cluster_best_channel_map(self) -> dict[int, int]:
        if self._templates is None or self._channel_map is None:
            return {}
        best_map: dict[int, int] = {}
        for cluster_id in range(self._templates.shape[0]):
            template = self._templates[cluster_id]
            amplitude = np.ptp(template, axis=0)
            site_idx = int(np.argmax(amplitude))
            if site_idx < len(self._channel_map):
                best_map[cluster_id] = int(self._channel_map[site_idx])
            else:
                best_map[cluster_id] = site_idx
        return best_map

    def _load_file(self, path: Path, loader, flatten: bool = False, **kwargs):
        if not path.exists():
            return None
        try:
            data = loader(path, **kwargs)
            return data.flatten() if flatten else data
        except Exception as e:
            show_warning(f"Failed to load {path.name}: {e}")
            return None

    def _resolve_dat_path(self, ks_folder: Path) -> Path | None:
        raw_exts = {".dat", ".bin", ".raw"}

        dat_path_str = (self._kilosort_params or {}).get("dat_path")
        if dat_path_str:
            candidate = Path(dat_path_str)
            if candidate.is_file():
                return candidate
            relative = ks_folder / candidate.name
            if relative.is_file():
                return relative
            relative2 = ks_folder / candidate
            if relative2.is_file():
                return relative2

        for folder in [ks_folder]:
            try:
                for entry in folder.iterdir():
                    if entry.is_file() and entry.suffix.lower() in raw_exts:
                        return entry
            except OSError:
                pass

        ephys_path_str = getattr(self.app_state, 'ephys_path', None)
        if ephys_path_str:
            ephys_path = Path(ephys_path_str)
            if ephys_path.is_file() and ephys_path.suffix.lower() in raw_exts:
                return ephys_path

        return None

    def _register_dat_fallback(self, ks_folder: Path):
        """Set up the Phy-Viewer panel using phylib reader for the raw .dat file."""
        if not self._kilosort_params:
            return

        dat_path = self._resolve_dat_path(ks_folder)
        if dat_path is None:
            return

        sr = self._kilosort_params["sample_rate"]
        n_channels = self._kilosort_params.get("n_channels_dat")

        if n_channels is None and self._channel_map is not None:
            n_channels = int(self._channel_map.max()) + 1

        if n_channels is None:
            show_warning("Cannot determine n_channels_dat — skipping Phy viewer.")
            return

        loader = get_ephys_loader(
            dat_path, stream_id="0",
            n_channels=n_channels, sampling_rate=sr,
        )
        if loader is None:
            show_warning(f"Failed to open .dat file: {dat_path}")
            return

        self._phy_reader = loader
        self._phy_sr = sr
        self._phy_n_channels = n_channels

        if not self.app_state.ephys_source_map:
            display_name = "Ephys Waveform"
            self.app_state.ephys_source_map[display_name] = (str(dat_path), "0", 0)
            self.app_state.ephys_stream_sel = display_name

        self.app_state.has_neo = True
        self.app_state.has_kilosort = True
        show_info(f"Phy viewer: {dat_path.name} ({n_channels} ch, {sr:.0f} Hz)")

    def _get_hardware_label(self) -> str:
        ephys_path, stream_id, _ = self.app_state.get_ephys_source()
        if not ephys_path:
            return "Hardware"
        ext = Path(ephys_path).suffix.lower()
        rawio_name = GenericEphysLoader.KNOWN_EXTENSIONS.get(ext)
        return _RAWIO_TO_DISPLAY.get(rawio_name, "Hardware")

    def get_hw_names(self, channel_map: np.ndarray | None) -> dict[int, str] | None:
        if channel_map is None:
            return None
        ephys_path, stream_id, _ = self.app_state.get_ephys_source()
        if not ephys_path:
            return None
        loader = get_ephys_loader(ephys_path, stream_id)
        if loader is None or not hasattr(loader, 'channel_names'):
            return None
        channel_names = loader.channel_names
        if channel_names is None:
            return None
        return {int(ch): channel_names[ch] for ch in channel_map}

    @staticmethod
    def _make_item(text: str, sort_value: float | None = None, user_data=None) -> QStandardItem:
        item = QStandardItem(text)
        item.setEditable(False)
        if sort_value is not None:
            item.setData(sort_value, _SORT_ROLE)
        if user_data is not None:
            item.setData(user_data, Qt.UserRole)
        return item

    @staticmethod
    def _format_value(value) -> tuple[str, float | None]:
        """Return (display_text, numeric_sort_value_or_None).

        Integers display without decimals; floats display with 3 d.p.
        Non-numeric values return (str, None).
        """
        if pd.isna(value):
            return "", None
        try:
            fval = float(value)
            if fval == int(fval):
                return str(int(fval)), fval
            return f"{fval:.3f}", fval
        except (ValueError, TypeError):
            return str(value), None

    def get_spike_times(self, cluster_id: int) -> np.ndarray:
        """All spike times in seconds for *cluster_id* (global, not trial-restricted)."""
        if self._tsgroup is None or cluster_id not in self._tsgroup:
            return np.array([], dtype=np.float64)
        return self._tsgroup[cluster_id].times()

    def _compute_isi_per_cluster(self) -> dict[int, float]:
        """Mean ISI in ms for each cluster."""
        if self._tsgroup is None:
            return {}
        isi_map: dict[int, float] = {}
        for cid in self._tsgroup.keys():
            intervals = np.diff(self._tsgroup[cid].times())
            isi_map[int(cid)] = float(np.mean(intervals) * 1000.0) if len(intervals) > 0 else np.nan
        return isi_map

    def _populate_cluster_table(self, df: pd.DataFrame):
        _PRIORITY = ["cluster_id", "ch", "sh", "KSLabel", "group", "fr", "Amplitude", "n_spikes"]
        _EXCLUDE = {"amp", "id_orig", "group_order"} | {c for c in df.columns if c.startswith("Unnamed")}

        ordered_cols = [c for c in _PRIORITY if c in df.columns]
        extra_cols = [c for c in df.columns if c not in set(_PRIORITY) and c not in _EXCLUDE]
        ordered_cols.extend(extra_cols)

        # Insert ISI column after n_spikes (or after last priority col)
        isi_map = self._compute_isi_per_cluster()
        if isi_map:
            insert_at = ordered_cols.index("n_spikes") + 1 if "n_spikes" in ordered_cols else len(ordered_cols)
            ordered_cols.insert(insert_at, "ISI (ms)")

        has_ch = "ch" in ordered_cols
        hw_names = self.get_hw_names(self._channel_map) if has_ch else None
        has_distinct_hw = hw_names is not None and any(
            name != f"Ch {hw}" for hw, name in hw_names.items()
        )
        hw_label = self._get_hardware_label() if has_distinct_hw else ""

        # Build header labels (group → Human, KSLabel → KS, cluster_id → id, ch → ch(KS) + ch(hw))
        # Trailing space creates a small gap between text and the filter-zone separator.
        header_labels: list[str] = []
        for col in ordered_cols:
            if col == "group":
                header_labels.append("Human  ")
            elif col == "KSLabel":
                header_labels.append("KS  ")
            elif col == "cluster_id":
                header_labels.append("id  ")
            elif col == "ch" and has_distinct_hw:
                header_labels.append("ch (KS)  ")
                header_labels.append(f"ch ({hw_label})  ")
            else:
                header_labels.append(col + "  ")

        self.cluster_table.setSortingEnabled(False)
        model = self._cluster_model
        model.clear()
        model.setHorizontalHeaderLabels(header_labels)

        for _, row in df.iterrows():
            cluster_id = int(row["cluster_id"]) if "cluster_id" in row.index and pd.notna(row["cluster_id"]) else None
            row_items: list[QStandardItem] = []
            for col in ordered_cols:
                if col == "ISI (ms)":
                    isi_val = isi_map.get(cluster_id, np.nan) if cluster_id is not None else np.nan
                    if pd.isna(isi_val):
                        row_items.append(self._make_item(""))
                    else:
                        text, sv = self._format_value(isi_val)
                        row_items.append(self._make_item(text, sv))
                elif col == "ch" and has_distinct_hw:
                    value = row[col]
                    ks_ch = int(value) if pd.notna(value) else 0
                    hw_name = hw_names.get(ks_ch) if hw_names else None
                    hw_display = hw_name if hw_name else str(ks_ch)
                    row_items.append(self._make_item(str(ks_ch), float(ks_ch)))
                    row_items.append(self._make_item(hw_display, float(ks_ch), user_data=ks_ch))
                else:
                    value = row[col] if col in row.index else None
                    text, sv = self._format_value(value)
                    row_items.append(self._make_item(text, sv))
            model.appendRow(row_items)

        self.cluster_table.setSortingEnabled(True)
        self._setup_filter_header(header_labels)
        self._apply_default_human_label_filter(header_labels)
        stripped = [h.strip() for h in header_labels]
        cid_view_col = stripped.index("id") if "id" in stripped else None
        if cid_view_col is not None:
            self.cluster_table.setItemDelegateForColumn(cid_view_col, self._cluster_id_delegate)

    def _setup_filter_header(self, col_names: list[str]):
        # ch/sh/id are always categorical even though they hold integers
        _force_cat = {"ch", "sh", "id"}
        model = self._cluster_model
        cat_cols: set[int] = set()
        num_cols: set[int] = set()
        self._filter_col_cats.clear()

        for col_idx, col_name in enumerate(col_names):
            name = col_name.strip()
            force_cat = name in _force_cat or name.startswith("ch (")
            if force_cat:
                unique_vals: list[str] = []
                for row in range(model.rowCount()):
                    item = model.item(row, col_idx)
                    if item and item.text() and item.text() not in unique_vals:
                        unique_vals.append(item.text())
                self._filter_col_cats[col_idx] = sorted(unique_vals)
                cat_cols.add(col_idx)
            else:
                # Check whether the column has numeric sort values
                is_numeric = any(
                    model.item(row, col_idx) is not None
                    and model.item(row, col_idx).data(_SORT_ROLE) is not None
                    for row in range(model.rowCount())
                )
                if is_numeric:
                    num_cols.add(col_idx)
                else:
                    unique_vals = []
                    for row in range(model.rowCount()):
                        item = model.item(row, col_idx)
                        if item and item.text() and item.text() not in unique_vals:
                            unique_vals.append(item.text())
                    self._filter_col_cats[col_idx] = sorted(unique_vals)
                    cat_cols.add(col_idx)

        self._filter_cat_cols = cat_cols
        self._filter_num_cols = num_cols
        self._filter_header.set_filterable(cat_cols, num_cols)
        self._update_header_active_filters()

    def _apply_default_human_label_filter(self, col_names: list[str]):
        stripped = [h.strip() for h in col_names]
        if "Human" not in stripped:
            return
        col_idx = stripped.index("Human")
        if col_idx not in self._filter_cat_cols:
            return
        vals = self._filter_col_cats.get(col_idx, [])
        if not any(v for v in vals):
            return
        if "good" in vals:
            allowed = {"good"}
            self._filter_cat_active[col_idx] = allowed
            self._cluster_proxy.set_cat_filter(col_idx, allowed)
            self._update_header_active_filters()

    def _on_filter_header_clicked(self, logical_col: int):
        header_item = self._cluster_model.horizontalHeaderItem(logical_col)
        col_name = (header_item.text() if header_item else "").strip()
        if col_name in {"ch", "ch (KS)"} or (col_name.startswith("ch (") and not col_name.startswith("ch (KS")):
            self._open_probe_channel_dialog()
            return
        if logical_col in self._filter_cat_cols:
            values = self._filter_col_cats.get(logical_col, [])
            active = self._filter_cat_active.get(logical_col, set())
            dialog = _CatFilterDialog(logical_col, values, active, self)
            if dialog.exec_() == QDialog.Accepted:
                allowed = dialog.get_allowed()
                if allowed:
                    self._filter_cat_active[logical_col] = allowed
                else:
                    self._filter_cat_active.pop(logical_col, None)
                self._cluster_proxy.set_cat_filter(logical_col, allowed)
                self._update_header_active_filters()
        elif logical_col in self._filter_num_cols:
            current = self._filter_num_active.get(logical_col)
            dialog = _NumFilterDialog(logical_col, current, self)
            if dialog.exec_() == QDialog.Accepted:
                f = dialog.get_filter()
                if f is None:
                    self._filter_num_active.pop(logical_col, None)
                    self._cluster_proxy.set_numeric_filter(logical_col, None, None)
                else:
                    op, val = f
                    self._filter_num_active[logical_col] = f
                    self._cluster_proxy.set_numeric_filter(logical_col, op, val)
                self._update_header_active_filters()

    def _update_header_active_filters(self):
        active = set(self._filter_cat_active.keys()) | set(self._filter_num_active.keys())
        self._filter_header.set_active_filters(active)

    def _find_col_by_header(self, prefix: str, exact: str | None = None) -> int | None:
        model = self._cluster_model
        for col in range(model.columnCount()):
            h = model.horizontalHeaderItem(col)
            if h is None:
                continue
            text = h.text().strip()
            if exact is not None and text == exact:
                return col
            if exact is None and text.startswith(prefix) and text != "ch (KS)":
                return col
        return None

    def _source_item(self, proxy_row: int, col: int) -> QStandardItem | None:
        proxy_idx = self._cluster_proxy.index(proxy_row, col)
        source_idx = self._cluster_proxy.mapToSource(proxy_idx)
        return self._cluster_model.itemFromIndex(source_idx)

    def _on_cluster_row_selected(self, _selected=None, _deselected=None):
        indexes = self.cluster_table.selectionModel().selectedRows()
        if not indexes:
            return

        hw_col_idx = self._find_col_by_header("ch (")
        ch_col_idx = hw_col_idx or self._find_col_by_header("", exact="ch")
        cluster_id_col_idx = self._find_col_by_header("", exact="id")

        # Navigate ephys channel to first selected row
        first_row = indexes[0].row()
        if ch_col_idx is not None:
            ch_item = self._source_item(first_row, ch_col_idx)
            if ch_item is not None:
                channel = ch_item.data(Qt.UserRole)
                if channel is None:
                    try:
                        channel = int(ch_item.text())
                    except (ValueError, TypeError):
                        channel = None
                if channel is not None:
                    self._apply_ephys_channel(int(channel))

        if cluster_id_col_idx is None:
            return

        if len(indexes) == 1:
            self._on_single_cluster_selected(first_row, cluster_id_col_idx, ch_col_idx)
        else:
            self._on_multi_cluster_selected(indexes, cluster_id_col_idx, ch_col_idx)

    def _on_single_cluster_selected(self, proxy_row: int, cid_col: int, hw_col_idx: int | None):
        cid_item = self._source_item(proxy_row, cid_col)
        if cid_item is None:
            return
        try:
            cluster_id = int(cid_item.text())
        except (ValueError, TypeError):
            return

        self._multi_cluster_colors.clear()
        self._multi_cluster_colors[cluster_id] = _CLUSTER_COLORS[0]
        self._apply_cluster_colors_to_table()

        ks_ch = self._get_ks_channel_for_row(proxy_row, hw_col_idx)
        self._draw_spikes_for_cluster(cluster_id, ks_ch)
        self._jump_to_first_spike()
        self._sync_cluster_id_to_combo(cluster_id)

        self._current_cluster_id_for_psth = cluster_id
        self.cluster_selected.emit(cluster_id)

    def _on_multi_cluster_selected(self, indexes, cid_col: int, hw_col_idx: int | None):
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return
        ephys_plot = self.plot_container.ephys_trace_plot
        sr = ephys_plot.buffer.ephys_sr
        if sr is None or sr <= 0:
            return

        self._multi_cluster_colors.clear()
        cluster_entries = []
        cluster_ids = []
        raster_entries = []

        for i, idx in enumerate(indexes):
            proxy_row = idx.row()
            cid_item = self._source_item(proxy_row, cid_col)
            if cid_item is None:
                continue
            try:
                cluster_id = int(cid_item.text())
            except (ValueError, TypeError):
                continue

            color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
            self._multi_cluster_colors[cluster_id] = color
            cluster_ids.append(cluster_id)

            times_local, samples_abs = self._restrict_to_trial(cluster_id, sr)

            ks_ch = self._get_ks_channel_for_row(proxy_row, hw_col_idx)
            channels = self._best_channels_for_cluster(cluster_id, ks_ch)
            cluster_entries.append((times_local, samples_abs, channels, color))

            best_ch = channels[0] if channels else ks_ch
            raster_entries.append((times_local, np.full(len(times_local), best_ch, dtype=np.int32), color))

        self._apply_cluster_colors_to_table()
        ephys_plot.set_multi_cluster_spike_data(cluster_entries)

        raster = self.plot_container.raster_plot
        raster.set_multi_cluster_spike_data(raster_entries)


    def _on_visible_channels_changed(self, _first: int, _last: int):
        pass

    def _apply_probe_channel_filter(self):
        ch_col = (
            self._find_col_by_header("", exact="ch")
            or self._find_col_by_header("", exact="ch (KS)")
        )
        if self._custom_channel_set is not None and ch_col is not None:
            self._cluster_proxy.set_visible_channel_filter(
                set(int(c) for c in self._custom_channel_set), ch_col
            )
        else:
            self._cluster_proxy.set_visible_channel_filter(None, None)

    def _select_clusters_all_visible(self):
        """Select all filtered-visible rows, highlight their spikes, then disable auto-highlight."""
        if self._cluster_proxy.rowCount() == 0:
            show_info("No clusters in current view.")
            return
        self.cluster_table.selectAll()

    def _update_highlight_label(self, _=None):
        n = self.n_closest_spin.value()
        self._highlight_label.setText(f"Highlight clusters (on {n} closest):")

    def _unselect_clusters(self):
        """Clear spike overlays, table selection, and disable auto-highlight."""
        self.cluster_table.clearSelection()
        self._clear_multi_cluster_mode()

    def _clear_multi_cluster_mode(self):
        self._multi_cluster_colors.clear()
        self._apply_cluster_colors_to_table()
        if self.plot_container and self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.clear_spike_overlays()
        if self.plot_container:
            self.plot_container.raster_plot.clear_spike_data()
            self._populate_raster_all_spikes()

    def _apply_cluster_colors_to_table(self):
        model = self._cluster_model
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None:
            return

        for row in range(model.rowCount()):
            item = model.item(row, cid_col)
            if item is None:
                continue
            try:
                cluster_id = int(item.text())
            except (ValueError, TypeError):
                continue

            color = self._multi_cluster_colors.get(cluster_id)
            cid_item = model.item(row, cid_col)
            if cid_item:
                cid_item.setData(QColor(*color[:3]) if color else None, _COLOR_ROLE)

    def _jump_to_first_spike(self):
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return
        self.plot_container.ephys_trace_plot.jump_to_spike(delta=0)

    def _reorder_probe_by_position(self):
        if self._channel_positions is None:
            return
        y_coords = self._channel_positions[:, 1]
        depth_order = np.argsort(y_coords)[::-1]
        if self._channel_map is not None:
            self._probe_channel_order = self._channel_map[depth_order].astype(int)
        else:
            self._probe_channel_order = depth_order.astype(int)
        self.apply_probe_order()

    def apply_probe_order(self):
        if self._probe_channel_order is None:
            return
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return
        self.plot_container.ephys_trace_plot.set_probe_channel_order(self._probe_channel_order)

    @staticmethod
    def _get_closest_channels(
        channel_positions: np.ndarray, channel_index: int, n: int | None = None,
    ) -> np.ndarray:
        """Get the n channels closest to *channel_index* on the probe.

        Direct port of ``phylib.io.model.get_closest_channels``.
        """
        x = channel_positions[:, 0]
        y = channel_positions[:, 1]
        x0, y0 = channel_positions[channel_index]
        d = (x - x0) ** 2 + (y - y0) ** 2
        out = np.argsort(d)
        if n:
            out = out[:n]
        return out

    def _find_best_channels(
        self,
        template: np.ndarray,
        n_closest_channels: int = 12,
        amplitude_threshold: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Find the best channels for a given template.

        Copied: ``phylib.io.model.TemplateModel._find_best_channels``.
        https://github.com/cortex-lab/phylib/blob/master/phylib/io/model.py

        Parameters
        ----------
        template : (n_samples, n_channels) array
            Mean waveform from ``templates.npy[template_id]``.
        n_closest_channels : int
            Max spatial neighbours to consider (Phy default 12).
        amplitude_threshold : float
            Fraction of peak amplitude; channels below are excluded.
            0 keeps all n_closest_channels (Phy default).

        Returns
        -------
        channel_ids : array of int
            Selected channel indices, sorted by descending amplitude.
        amplitude : array of float
            Amplitude on each selected channel.
        best_channel : int
            Channel with maximum amplitude.
        """
        assert template.ndim == 2
        amplitude = template.max(axis=0) - template.min(axis=0)

        best_channel = int(np.argmax(amplitude))
        max_amp = amplitude[best_channel]

        peak_channels = np.nonzero(amplitude >= amplitude_threshold * max_amp)[0]

        close_channels = self._get_closest_channels(
            self._channel_positions, best_channel, n_closest_channels,
        )

        channel_ids = np.intersect1d(peak_channels, close_channels)

        order = np.argsort(amplitude[channel_ids])[::-1]
        channel_ids = channel_ids[order]
        amplitude_out = amplitude[channel_ids]

        return channel_ids, amplitude_out, best_channel

    def _draw_spikes_for_cluster(self, cluster_id: int, channel: int):
        if self._tsgroup is None:
            return
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return

        ephys_plot = self.plot_container.ephys_trace_plot
        sr = ephys_plot.buffer.ephys_sr
        if sr is None or sr <= 0:
            return

        times_local, samples_abs = self._restrict_to_trial(cluster_id, sr)
        channels = self._best_channels_for_cluster(cluster_id, channel)
        ephys_plot.set_spike_data(times_local, samples_abs, channels)

        best_ch = channels[0] if channels else channel
        raster = self.plot_container.raster_plot
        raster.set_spike_data(times_local, np.full(len(times_local), best_ch, dtype=np.int32))

    def _best_channels_for_cluster(self, cluster_id: int, fallback_channel: int) -> list[int]:
        if (
            self._templates is not None
            and self._channel_positions is not None
            and self._channel_map is not None
            and cluster_id < self._templates.shape[0]
        ):
            template = self._templates[cluster_id]
            n_closest = self.n_closest_spin.value()
            site_indices, _amp, _best = self._find_best_channels(
                template, n_closest_channels=n_closest,
            )
            hw_channels = [
                int(self._channel_map[i]) for i in site_indices if i < len(self._channel_map)
            ]
            return hw_channels if hw_channels else [fallback_channel]
        return [fallback_channel]

    def _get_ks_channel_for_row(self, proxy_row: int, hw_col_idx: int | None) -> int:
        if hw_col_idx is not None:
            ch_item = self._source_item(proxy_row, hw_col_idx)
            if ch_item is not None:
                val = ch_item.data(Qt.UserRole)
                if val is not None:
                    return int(val)
                try:
                    return int(ch_item.text())
                except (ValueError, TypeError):
                    pass
        return self.ephys_channel_spin.value()

    def _apply_ephys_channel(self, channel: int):
        stream_sel = getattr(self.app_state, 'ephys_stream_sel', None)
        source_map = getattr(self.app_state, 'ephys_source_map', {})
        if not stream_sel or stream_sel not in source_map:
            return

        filename, stream_id, _ = source_map[stream_sel]
        source_map[stream_sel] = (filename, stream_id, channel)

        self.ephys_channel_spin.blockSignals(True)
        self.ephys_channel_spin.setValue(channel)
        self.ephys_channel_spin.blockSignals(False)

        if self.plot_container and self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.set_channel(channel)
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)



    # ------------------------------------------------------------------
    # Firing rate panel
    # ------------------------------------------------------------------

    def _create_firing_rate_panel(self, main_layout):
        self.firing_rate_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.firing_rate_panel.setLayout(layout)

        group = QGroupBox("Firing rates")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(6)
        group_layout.setContentsMargins(8, 12, 8, 8)
        group.setLayout(group_layout)
        layout.addWidget(group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Clusters:"))
        self.fr_group_combo = QComboBox()
        self.fr_group_combo.addItems(["good", "good + mua", "mua", "Selected in table"])
        self.fr_group_combo.setToolTip("Which clusters to include in firing rate computation")
        self.fr_group_combo.currentTextChanged.connect(self._on_fr_param_changed)
        row1.addWidget(self.fr_group_combo)
        group_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Bin (s):"))
        self.fr_bin_spin = QDoubleSpinBox()
        self.fr_bin_spin.setRange(0.001, 1.0)
        self.fr_bin_spin.setValue(self.app_state.fr_bin_size)
        self.fr_bin_spin.setSingleStep(0.005)
        self.fr_bin_spin.setDecimals(3)
        self.fr_bin_spin.setToolTip("Time bin width in seconds for spike counting")
        self.fr_bin_spin.valueChanged.connect(self._on_fr_param_changed)
        row2.addWidget(self.fr_bin_spin)
        row2.addWidget(QLabel("σ (bins):"))
        self.fr_sigma_spin = QDoubleSpinBox()
        self.fr_sigma_spin.setRange(0.0, 50.0)
        self.fr_sigma_spin.setValue(self.app_state.fr_sigma)
        self.fr_sigma_spin.setSingleStep(0.5)
        self.fr_sigma_spin.setDecimals(1)
        self.fr_sigma_spin.setToolTip("Gaussian smoothing width in bins (0 = no smoothing)")
        self.fr_sigma_spin.valueChanged.connect(self._on_fr_param_changed)
        row2.addWidget(self.fr_sigma_spin)
        group_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.fr_compute_btn = QPushButton("Compute")
        self.fr_compute_btn.setToolTip("Compute firing rates for the current trial")
        self.fr_compute_btn.clicked.connect(lambda: self._compute_firing_rates(force=True))
        row3.addWidget(self.fr_compute_btn)
        self.fr_status_label = QLabel("")
        row3.addWidget(self.fr_status_label)
        row3.addStretch()
        group_layout.addLayout(row3)

        pca_group = QGroupBox("PCA")
        pca_layout = QVBoxLayout()
        pca_layout.setSpacing(6)
        pca_layout.setContentsMargins(8, 12, 8, 8)
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group)

        self.pca_zscore_cb = QCheckBox("Z-score clusters")
        self.pca_zscore_cb.setChecked(True)
        self.pca_zscore_cb.setToolTip("Z-score each cluster's firing rate before PCA")
        pca_layout.addWidget(self.pca_zscore_cb)

        self.pca_btn = QPushButton("Compute PCA")
        self.pca_btn.setToolTip("Project firing rates to PC space via SVD")
        self.pca_btn.clicked.connect(self._compute_pca)
        pca_layout.addWidget(self.pca_btn)

        self.pca_status_label = QLabel("")
        pca_layout.addWidget(self.pca_status_label)

        main_layout.addWidget(self.firing_rate_panel)

    def _on_fr_param_changed(self):
        self.app_state.fr_bin_size = self.fr_bin_spin.value()
        self.app_state.fr_sigma = self.fr_sigma_spin.value()
        self._fr_cache_key = None

    def _get_selected_cluster_ids(self) -> np.ndarray | None:
        selection = self.fr_group_combo.currentText()

        if selection == "Selected in table":
            ids = self._get_table_selected_cluster_ids()
            if ids is None or len(ids) == 0:
                show_warning("No clusters selected in the table.")
                return None
            return ids

        if self._cluster_df is None or "group" not in self._cluster_df.columns:
            return None
        if "cluster_id" not in self._cluster_df.columns:
            return None
        group_map = {
            "good": ["good"],
            "mua": ["mua"],
            "good + mua": ["good", "mua"],
        }
        allowed = group_map.get(selection, ["good"])
        mask = self._cluster_df["group"].isin(allowed)
        ids = self._cluster_df.loc[mask, "cluster_id"].values
        if len(ids) == 0:
            show_warning(f"No clusters with group '{selection}' found in cluster table.")
            return None
        return ids

    def _get_table_selected_cluster_ids(self) -> np.ndarray | None:
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None:
            return None
        indexes = self.cluster_table.selectionModel().selectedRows()
        if not indexes:
            return None
        ids = []
        for idx in indexes:
            item = self._source_item(idx.row(), cid_col)
            if item is not None:
                try:
                    ids.append(int(item.text()))
                except (ValueError, TypeError):
                    pass
        return np.array(ids) if ids else None

    def _compute_firing_rates(self, force: bool = False):
        if self._tsgroup is None:
            return

        trial = self.app_state.trials_sel
        if not trial:
            return

        ephys_sr = self._kilosort_sr
        if ephys_sr is None:
            loader = self._get_any_ephys_loader()
            if loader is None:
                show_warning("No sample rate available — load kilosort folder with params.py or specify an ephys folder.")
                return
            ephys_sr = loader.rate

        cluster_ids = self._get_selected_cluster_ids()
        bin_size = self.fr_bin_spin.value()
        sigma = self.fr_sigma_spin.value()
        group_text = self.fr_group_combo.currentText()

        cache_key = (trial, bin_size, sigma, group_text)
        if not force and self._fr_cache_key == cache_key:
            return
        self._fr_cache_key = cache_key

        ds = self.app_state.dt.trial(trial)
        start_time = self.app_state.dt.session_io.start_time(trial)
        bounds = self.app_state.trial_bounds
        if bounds is None:
            return

        da = firing_rate_to_xarray(
            self._spike_times_s, self._spike_clusters, bin_size,
            t_start=start_time, t_stop=bounds.end_s,
            _tsgroup=self._tsgroup,
        )

        if cluster_ids is not None:
            valid_ids = np.intersect1d(cluster_ids, da.coords["cluster_id"].values)
            da = da.sel(cluster_id=valid_ids)

        if sigma > 0:
            smoothed = gaussian_filter1d(da.values, sigma, axis=1)
            da = da.copy(data=smoothed)

        da = da.assign_coords(time_fr=da.coords["time_fr"].values - start_time)

        new_ds = ds.copy()
        if "firing_rate" in new_ds.data_vars:
            new_ds = new_ds.drop_vars("firing_rate")
        new_ds["firing_rate"] = da

        self.app_state.dt.update_trial(trial, lambda _: new_ds)

        self.app_state.ds = self.app_state.dt.trial(trial)
        n_clusters = len(cluster_ids) if cluster_ids is not None else len(np.unique(self._spike_clusters))
        self.fr_status_label.setText(
            f"{n_clusters} clusters ({group_text})"
        )
        self._enable_feature_item("Firing rate")
        self._update_cluster_id_combo()

        if self.app_state.ready:
            features_combo = self.data_widget.combos.get("features")
            if features_combo is not None:
                set_combo_to_value(features_combo, "firing_rate")
                self.app_state.set_key_sel("features", get_combo_value(features_combo))
                self.data_widget._update_view_mode_items(get_combo_value(features_combo))
                self.data_widget.view_mode_combo.show()
                self.data_widget._apply_view_mode_for_feature()
            self.data_widget.update_main_plot()

    def _register_kilosort_features(self):
        if not self.data_widget:
            return

        features_list = self.data_widget.type_vars_dict.get("features", [])
        features_combo = self.data_widget.combos.get("features")

        _display_to_var = {"Firing rate": "firing_rate", "PCA": "pca"}
        for display_name, var_name in _display_to_var.items():
            if display_name not in features_list:
                features_list.append(display_name)
            if features_combo is not None and find_combo_index(features_combo, var_name) < 0:
                features_combo.addItem(display_name, var_name)
                self._set_combo_item_enabled(features_combo, display_name, False)

        slot1 = getattr(self.data_widget, 'space_view_combo', None)
        if slot1 is not None:
            for label in ("PCA 2D", "PCA 3D"):
                if slot1.findText(label) < 0:
                    slot1.addItem(label)
                    self._set_combo_item_enabled(slot1, label, False)

    def _set_combo_item_enabled(self, combo: QComboBox, text: str, enabled: bool):
        idx = find_combo_index(combo, text)
        if idx < 0:
            return
        model = combo.model()
        item = model.item(idx)
        if item is not None:
            if enabled:
                item.setFlags(item.flags() | Qt.ItemIsEnabled)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)

    def _enable_feature_item(self, display_name: str):
        if not self.data_widget:
            return
        features_combo = self.data_widget.combos.get("features")
        if features_combo is not None:
            self._set_combo_item_enabled(features_combo, display_name, True)

    def _disable_feature_item(self, display_name: str):
        if not self.data_widget:
            return
        features_combo = self.data_widget.combos.get("features")
        if features_combo is not None:
            self._set_combo_item_enabled(features_combo, display_name, False)

    def _update_cluster_id_combo(self):
        if not self.data_widget:
            return
        da = self.app_state.ds.get("firing_rate")
        if da is not None and "cluster_id" in da.dims:
            cluster_ids = [str(c) for c in da.coords["cluster_id"].values]
            if "cluster_id" not in self.data_widget.combos:
                self.data_widget._create_combo_widget("cluster_id", cluster_ids)
            else:
                combo = self.data_widget.combos["cluster_id"]
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(cluster_ids)
                combo.blockSignals(False)

    def _compute_pca(self):
        ds = self.app_state.ds
        if ds is None or "firing_rate" not in ds.data_vars:
            show_warning("Compute firing rates first before running PCA.")
            return

        fr_da = ds["firing_rate"]
        pca_da = compute_pca(fr_da, n_components=3, zscore=self.pca_zscore_cb.isChecked())

        trial = self.app_state.trials_sel
        new_ds = ds.copy()
        if "pca" in new_ds.data_vars:
            new_ds = new_ds.drop_vars("pca")
        new_ds["pca"] = pca_da

        self.app_state.dt.update_trial(trial, lambda _: new_ds)
        self.app_state.ds = self.app_state.dt.trial(trial)

        ev = pca_da.attrs["explained_variance"]
        self.pca_status_label.setText(
            f"PC1: {ev[0]:.1%}  PC2: {ev[1]:.1%}  PC3: {ev[2]:.1%}"
        )

        self._enable_feature_item("PCA")

        slot1 = getattr(self.data_widget, 'space_view_combo', None)
        if slot1 is not None:
            for label in ("PCA 2D", "PCA 3D"):
                self._set_combo_item_enabled(slot1, label, True)
            slot1.setCurrentText("PCA 2D")

        pca_da = self.app_state.ds["pca"]
        if "pc" in pca_da.dims:
            pc_labels = [str(c) for c in pca_da.coords["pc"].values]
            if "pc" not in self.data_widget.combos:
                self.data_widget._create_combo_widget("pc", pc_labels)

        if self.app_state.ready:
            features_combo = self.data_widget.combos.get("features")
            if features_combo is not None:
                set_combo_to_value(features_combo, "pca")
                self.app_state.set_key_sel("features", get_combo_value(features_combo))
            self.data_widget.update_main_plot()

    def _get_any_ephys_loader(self):
        source_map = getattr(self.app_state, 'ephys_source_map', {})
        if not source_map:
            return None
        filename, stream_id, _ = next(iter(source_map.values()))
        if os.path.isabs(filename):
            ephys_path = os.path.normpath(filename)
        else:
            base_ephys_path = getattr(self.app_state, 'ephys_path', None)
            if not base_ephys_path:
                return None
            ephys_path = os.path.normpath(
                os.path.join(os.path.dirname(base_ephys_path), filename)
            )
        return get_ephys_loader(ephys_path, stream_id)

    # ------------------------------------------------------------------
    # Bidirectional cluster_id sync
    # ------------------------------------------------------------------

    def _sync_cluster_id_to_combo(self, cluster_id: int):
        if not self.data_widget:
            return
        combo = self.data_widget.combos.get("cluster_id")
        if combo is None:
            return
        combo.blockSignals(True)
        idx = find_combo_index(combo, str(cluster_id))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)
        self.app_state.set_key_sel("cluster_id", str(cluster_id))

    def select_cluster_in_table(self, cluster_id: int):
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None:
            return
        sel_model = self.cluster_table.selectionModel()
        sel_model.blockSignals(True)
        for proxy_row in range(self._cluster_proxy.rowCount()):
            item = self._source_item(proxy_row, cid_col)
            if item and item.text() == str(cluster_id):
                self.cluster_table.selectRow(proxy_row)
                sel_model.blockSignals(False)
                return
        sel_model.blockSignals(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container
        plot_container.plot_changed.connect(self._on_plot_changed)
        ephys_plot = plot_container.ephys_trace_plot
        if ephys_plot is not None:
            ephys_plot.gain_scroll_requested.connect(self._on_gain_scroll)
            ephys_plot.visible_channels_changed.connect(self._on_visible_channels_changed)

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    def set_data_widget(self, data_widget):
        self.data_widget = data_widget

    def _on_gain_scroll(self, delta: int):
        spin = self.ephys_gain_spin
        new_val = round(spin.value() + delta * 0.1, 1)
        spin.setValue(max(spin.minimum(), min(new_val, spin.maximum())))

    def get_stream_names(self) -> list[str]:
        """Return available ephys stream display names from the source map."""
        source_map = getattr(self.app_state, 'ephys_source_map', {})
        return list(source_map.keys())

    def on_trial_changed(self):
        self._fr_cache_key = None
        if not self.data_widget:
            return
        self._disable_feature_item("Firing rate")
        self._disable_feature_item("PCA")
        slot1 = getattr(self.data_widget, 'space_view_combo', None)
        if slot1 is not None:
            for label in ("PCA 2D", "PCA 3D"):
                self._set_combo_item_enabled(slot1, label, False)
        self.fr_status_label.setText("")
        self.pca_status_label.setText("")

        features_combo = self.data_widget.combos.get("features")
        if features_combo is not None and get_combo_value(features_combo) in ("firing_rate", "pca"):
            features_combo.setCurrentIndex(0)

        self.configure_ephys_trace_plot()
        self._redraw_selected_clusters()
        if self.data_widget:
            self.data_widget._configure_neo_panel()

    def _redraw_selected_clusters(self):
        if not self._multi_cluster_colors:
            return
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None:
            return

        target_ids = set(self._multi_cluster_colors.keys())
        proxy = self._cluster_proxy
        sel_model = self.cluster_table.selectionModel()
        sel_model.blockSignals(True)
        sel_model.clearSelection()
        for row in range(proxy.rowCount()):
            val = proxy.data(proxy.index(row, cid_col))
            try:
                if int(val) in target_ids:
                    sel_model.select(
                        proxy.index(row, 0),
                        QItemSelectionModel.Select | QItemSelectionModel.Rows,
                    )
            except (ValueError, TypeError):
                pass
        sel_model.blockSignals(False)
        self._on_cluster_row_selected()

    def _on_plot_changed(self, plot_type: str):
        if plot_type == 'ephystrace':
            self.apply_probe_order()
