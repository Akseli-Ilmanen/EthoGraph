"""Ephys widget — trace controls, Kilosort neuron jumping, and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import xarray as xr
from napari.utils.notifications import show_info, show_warning
from napari.viewer import Viewer
from scipy.ndimage import gaussian_filter1d
from qtpy.QtCore import Qt, QRectF, QSortFilterProxyModel
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
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import ethograph as eto
from ethograph.features.neural import build_tsgroup, compute_pca, firing_rate_to_xarray

from .app_constants import CLUSTER_TABLE_MAX_HEIGHT, CLUSTER_TABLE_ROW_HEIGHT
from .makepretty import find_combo_index, get_combo_value, set_combo_to_value, styled_link
from .plots_ephystrace import SharedEphysCache

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


_SORT_ROLE = Qt.UserRole + 1
_ALL_FILTER = "All"


class _MultiColumnFilterProxy(QSortFilterProxyModel):
    """Proxy that filters rows by exact match on multiple columns independently."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._col_filters: dict[int, str] = {}
        self._visible_channels: set[int] | None = None
        self._ch_col: int | None = None

    def set_column_filter(self, col: int, value: str):
        if value == _ALL_FILTER or not value:
            self._col_filters.pop(col, None)
        else:
            self._col_filters[col] = value
        self.invalidateFilter()

    def set_visible_channel_filter(self, channels: set[int] | None, ch_col: int | None):
        self._visible_channels = channels
        self._ch_col = ch_col
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent):
        model = self.sourceModel()
        for col, value in self._col_filters.items():
            item = model.item(source_row, col)
            if item is None:
                return False
            if item.text() != value:
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


class EphysWidget(QWidget):
    """Ephys controls with toggle-button tabs: Ephys trace | Neuron jumping | Preprocessing."""

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
        self._spike_times: np.ndarray | None = None
        self._channel_positions: np.ndarray | None = None
        self._channel_map: np.ndarray | None = None
        self._probe_channel_order: np.ndarray | None = None
        self._custom_channel_set: np.ndarray | None = None
        self._templates: np.ndarray | None = None
        self._ephys_n_channels = 0
        self._tsgroup = None
        self._tsgroup_ephys_sr: float | None = None
        self._kilosort_sr: float | None = None
        self._fr_cache_key: tuple | None = None
        self._kilosort_params: dict | None = None

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_toggle_buttons(main_layout)
        self._create_traceview_panel(main_layout)
        self._create_preprocessing_panel(main_layout)
        self._create_firing_rate_panel(main_layout)

        self._enforce_ephys_sequential()
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
            ("preproc_toggle", "Preprocessing", self._toggle_preproc),
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
            "preproc": (self.preproc_panel, self.preproc_toggle),
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
        self._show_panel("traceview" if self.traceview_toggle.isChecked() else "preproc")

    def _toggle_preproc(self):
        self._show_panel("preproc" if self.preproc_toggle.isChecked() else "traceview")

    def _toggle_firing_rate(self):
        self._show_panel("firing_rate" if self.firing_rate_toggle.isChecked() else "traceview")

    def _refresh_layout(self):
        if self.meta_widget:
            self.meta_widget.refresh_widget_layout(self)

    # ------------------------------------------------------------------
    # Ephys trace panel (channel, multichannel, gain, range)
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

        # Stream combo (populated later via populate_stream_combo)
        self._stream_row = QWidget()
        stream_layout = QHBoxLayout()
        stream_layout.setContentsMargins(0, 0, 0, 0)
        stream_layout.setSpacing(5)
        self._stream_row.setLayout(stream_layout)
        self._stream_label = QLabel("Stream:")
        stream_layout.addWidget(self._stream_label)
        self.ephys_stream_combo = QComboBox()
        self.ephys_stream_combo.setObjectName("ephys_stream_combo")
        self.ephys_stream_combo.currentTextChanged.connect(self._on_ephys_stream_changed)
        stream_layout.addWidget(self.ephys_stream_combo)
        stream_layout.addStretch()
        group_layout.addWidget(self._stream_row)
        self._stream_row.hide()

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
        ch_row.addWidget(self.ephys_channel_label)
        ch_row.addWidget(self.ephys_channel_spin)
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
        self.n_closest_spin.setValue(12)
        self.n_closest_spin.setToolTip("Number of spatially closest channels for waveform display")
        probe_row_layout.addWidget(self.n_closest_spin)
        probe_row_layout.addStretch()

        self._probe_row.hide()
        group_layout.addWidget(self._probe_row)

        # Filter + Show all good neurons row
        _good_row = QHBoxLayout()
        _good_row.setContentsMargins(0, 0, 0, 0)
        _good_row.setSpacing(4)

        self._filter_visible_cb = QCheckBox("Filter to visible")
        self._filter_visible_cb.setToolTip("Only show clusters whose best channel is currently visible in the ephys trace")
        self._filter_visible_cb.toggled.connect(self._on_filter_visible_toggled)
        _good_row.addWidget(self._filter_visible_cb)

        self._show_all_good_btn = QPushButton("Show all good neurons")
        self._show_all_good_btn.setToolTip("Overlay spikes from all 'good' clusters with distinct colors")
        self._show_all_good_btn.setCheckable(True)
        self._show_all_good_btn.clicked.connect(self._toggle_show_all_good)
        _good_row.addWidget(self._show_all_good_btn)

        group_layout.addLayout(_good_row)

        self._multi_cluster_colors: dict[int, tuple] = {}

        # Cluster table with filters
        self._filter_row = QWidget()
        self._filter_layout = QHBoxLayout()
        self._filter_layout.setContentsMargins(0, 0, 0, 0)
        self._filter_layout.setSpacing(2)
        self._filter_row.setLayout(self._filter_layout)
        self._filter_combos: list[QComboBox] = []
        self._filter_row.hide()
        layout.addWidget(self._filter_row)

        ref_label_phy = QLabel(styled_link(
            "https://phy.readthedocs.io/en/latest/",
            "TraceView from Phy",
        ))
        ref_label_phy.setOpenExternalLinks(True)
        layout.addWidget(ref_label_phy)

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

        header = self.cluster_table.horizontalHeader()
        header.setDefaultSectionSize(40)
        header.setMinimumSectionSize(20)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

        self.cluster_table.setStyleSheet("""
            QTableView { gridline-color: transparent; }
            QTableView::item { padding: 0px 2px; }
            QHeaderView::section { padding: 0px 2px; }
        """)
        self.cluster_table.selectionModel().selectionChanged.connect(self._on_cluster_row_selected)
        layout.addWidget(self.cluster_table)

        main_layout.addWidget(self.traceview_panel)

    # ------------------------------------------------------------------
    # Ephys trace handlers
    # ------------------------------------------------------------------

    def populate_stream_combo(self):
        stream_names = self.get_stream_names()
        if not stream_names:
            self._stream_row.hide()
            return
        self.ephys_stream_combo.blockSignals(True)
        self.ephys_stream_combo.clear()
        self.ephys_stream_combo.addItems(stream_names)
        self.app_state.ephys_stream_sel = stream_names[0]
        self.ephys_stream_combo.blockSignals(False)
        self._stream_row.setVisible(len(stream_names) > 1)

    def _on_ephys_stream_changed(self, stream_name):
        if not self.app_state.ready or not stream_name:
            return
        source_map = getattr(self.app_state, 'ephys_source_map', {})
        if stream_name not in source_map:
            return
        self.app_state.ephys_stream_sel = stream_name
        self.configure_ephys_trace_plot()
        if self.plot_container:
            self.plot_container.show_ephys_panel()

    def configure_ephys_trace_plot(self):
        ephys_path, stream_id, channel_idx = self.app_state.get_ephys_source()
        if not ephys_path:
            return

        alignment = getattr(self.app_state, 'trial_alignment', None)
        if alignment and "ephys" in alignment.continuous:
            loader = alignment.continuous["ephys"]._loader
        else:
            from .plots_ephystrace import SharedEphysCache
            loader = SharedEphysCache.get_loader(ephys_path, stream_id)

        if loader is None:
            return
        self.plot_container.ephys_trace_plot.set_loader(loader, channel_idx)

        offset = self.app_state.dt.get_start_time(self.app_state.trials_sel)
        t_min, t_max = self.app_state.get_time_bounds()
        duration = t_max - t_min
        self.plot_container.ephys_trace_plot.set_ephys_offset(offset, duration)

        n_ch = loader.n_channels
        self._ephys_n_channels = n_ch
        self.ephys_channel_spin.blockSignals(True)
        self.ephys_channel_spin.setRange(0, max(0, n_ch - 1))
        self.ephys_channel_spin.setValue(channel_idx)
        self.ephys_channel_spin.blockSignals(False)
        self.ephys_channel_label.show()
        self.ephys_channel_spin.show()
        self.ephys_gain_label.show()
        self.ephys_gain_spin.show()

        if self.plot_container.is_ephystrace():
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

        if self._spike_times is not None and self._spike_clusters is not None:
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
            print(f"[widgets_ephys] update_plot_content called (ephys_trace_plot) xmin={xmin}, xmax={xmax}")
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

    def set_neural_view(self, mode: str):
        """Switch between '1-ch Trace', 'Multi Trace', 'Raster'."""
        if not self.plot_container:
            return

        ephys_plot = self.plot_container.ephys_trace_plot

        if mode == "1-ch Trace":
            self.plot_container.set_neural_panel_mode("trace")
            ephys_plot.set_multichannel(False)
            self.ephys_channel_spin.setEnabled(True)

        elif mode == "Multi Trace":
            self.plot_container.set_neural_panel_mode("trace")
            ephys_plot.set_multichannel(True)
            ephys_plot.auto_channel_spacing()
            if self.ephys_auto_gain_cb.isChecked():
                self._apply_auto_gain()
            ephys_plot.autoscale()
            self.ephys_channel_spin.setEnabled(False)

        elif mode == "Raster":
            self.ephys_channel_spin.setEnabled(False)
            if not ephys_plot._multichannel:
                ephys_plot.set_multichannel(True)
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
            return

        self._custom_channel_set = hw_channels

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
        if self.plot_container and getattr(self.plot_container, 'ephys_trace_plot', None) is not None:
            self.plot_container.ephys_trace_plot.set_multichannel(False)

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
            show_warning("No sample_rate found in params.py — cannot load kilosort folder.")
            return
        ks_sr = ks_params["sample_rate"]
        if not self._validate_kilosort_sr(ks_sr):
            return
        self._kilosort_sr = ks_sr
        self._kilosort_params = ks_params

        cluster_info_path = folder / "cluster_info.tsv"
        cluster_group_path = folder / "cluster_group.tsv"
        if cluster_info_path.exists():
            self._cluster_df = self._load_file(cluster_info_path, pd.read_csv, sep='\t')
        elif cluster_group_path.exists():
            self._cluster_df = self._load_file(cluster_group_path, pd.read_csv, sep='\t')
            show_info("Using cluster_group.tsv (cluster_info.tsv not found)")
        else:
            show_warning("No cluster_info.tsv or cluster_group.tsv found — cluster table will be empty.")
            self._cluster_df = None



        self._spike_clusters = self._load_file(folder / "spike_clusters.npy", np.load, flatten=True)
        self._spike_times = self._load_file(folder / "spike_times.npy", np.load, flatten=True)
        self._channel_positions = self._load_file(folder / "channel_positions.npy", np.load)
        self._channel_map = self._load_file(folder / "channel_map.npy", np.load, flatten=True)
        self._templates = self._load_file(folder / "templates.npy", np.load)
        
        self._reorder_probe_by_position()

        if self._cluster_df is not None:
            self._populate_cluster_table(self._cluster_df)


        if self._channel_positions is not None and self._channel_map is not None:
            self._probe_row.show()

        self._register_dat_fallback(folder)

        if self._spike_times is not None and self._spike_clusters is not None:
            self._register_kilosort_features()
            self._populate_raster_all_spikes()

    def _populate_raster_all_spikes(self):
        if not self.plot_container or self._spike_times is None or self._spike_clusters is None:
            return

        ephys_plot = self.plot_container.ephys_trace_plot
        sr = ephys_plot.buffer.ephys_sr
        if sr is None or sr <= 0:
            return

        ephys_offset = ephys_plot._ephys_offset
        trial_duration = ephys_plot._trial_duration
        trial_end = ephys_offset + trial_duration if trial_duration else np.inf

        spike_times_s = self._spike_times.astype(np.float64) / sr
        in_trial = (spike_times_s >= ephys_offset) & (spike_times_s < trial_end)
        trial_spike_times = spike_times_s[in_trial] - ephys_offset
        trial_clusters = self._spike_clusters[in_trial]

        best_ch_map = self._build_cluster_best_channel_map()

        best_channels = np.array(
            [best_ch_map.get(int(c), 0) for c in trial_clusters],
            dtype=np.int32,
        )

        raster = self.plot_container.raster_plot

        all_ch = ephys_plot._all_ordered_channels()
        total = len(all_ch)
        if total > 0:
            spacing = ephys_plot.buffer.channel_spacing
            hw_to_y = {
                int(hw): (total - 1 - i) * spacing
                for i, hw in enumerate(all_ch)
            }
            raster.sync_y_axis(hw_to_y, spacing, total)

        raster.set_spike_data(trial_spike_times, best_channels)

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
        if self.app_state.ephys_source_map:
            return

        if not self._kilosort_params:
            return

        dat_path = self._resolve_dat_path(ks_folder)
        if dat_path is None:
            return

        sr = self._kilosort_params["sample_rate"]
        dtype = self._kilosort_params.get("dtype", "int16")
        n_channels = self._kilosort_params.get("n_channels_dat")

        if n_channels is None and self._channel_map is not None:
            n_channels = int(self._channel_map.max()) + 1

        if n_channels is None:
            show_warning("Cannot determine n_channels_dat for .dat fallback — skipping ephys trace.")
            return

        loader = SharedEphysCache.get_loader(
            dat_path, stream_id="0",
            n_channels=n_channels, sampling_rate=sr,
        )
        if loader is None:
            return

        display_name = "Ephys Waveform"
        self.app_state.ephys_source_map[display_name] = (str(dat_path), "0", 0)
        self.app_state.ephys_stream_sel = display_name

        self.populate_stream_combo()
        self.configure_ephys_trace_plot()

        if self.plot_container:
            self.plot_container.show_ephys_panel()

        show_info(f"Loaded .dat ephys: {dat_path.name} ({n_channels} ch, {sr:.0f} Hz)")

    def _get_hardware_label(self) -> str:
        from .plots_ephystrace import GenericEphysLoader
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
        loader = SharedEphysCache.get_loader(ephys_path, stream_id)
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

    def _populate_cluster_table(self, df: pd.DataFrame):
        display_cols = ["cluster_id", "ch", "group", "fr", "Amplitude", "n_spikes"]
        numeric_cols = {"cluster_id", "ch", "fr", "Amplitude", "n_spikes"}
        available_cols = [c for c in display_cols if c in df.columns]

        has_ch = "ch" in available_cols
        hw_names = self.get_hw_names(self._channel_map) if has_ch else None
        has_distinct_hw = hw_names is not None and any(
            name != f"Ch {hw}" for hw, name in hw_names.items()
        )
        if has_ch and has_distinct_hw:
            hw_label = self._get_hardware_label()
            ch_idx = available_cols.index("ch")
            available_cols[ch_idx] = "ch (KS)"
            available_cols.insert(ch_idx + 1, f"ch ({hw_label})")

        self.cluster_table.setSortingEnabled(False)
        model = self._cluster_model
        model.clear()
        model.setHorizontalHeaderLabels(available_cols)

        for row_idx in range(len(df)):
            row_items: list[QStandardItem] = []
            for col_name in display_cols:
                if col_name not in df.columns:
                    continue
                value = df.iloc[row_idx][col_name]

                if col_name == "ch" and has_ch and has_distinct_hw:
                    ks_ch = int(value) if pd.notna(value) else 0
                    hw_name = hw_names.get(ks_ch) if hw_names else None
                    hw_display = hw_name if hw_name else str(ks_ch)

                    row_items.append(self._make_item(str(ks_ch), float(ks_ch)))
                    row_items.append(self._make_item(hw_display, float(ks_ch), user_data=ks_ch))
                elif col_name in numeric_cols and pd.notna(value):
                    display = f"{float(value):.2f}" if col_name == "fr" else str(int(value))
                    row_items.append(self._make_item(display, float(value)))
                elif pd.isna(value):
                    row_items.append(self._make_item(""))
                else:
                    row_items.append(self._make_item(str(value)))
            model.appendRow(row_items)

        self.cluster_table.setSortingEnabled(True)
        self._build_filter_combos(available_cols)

    def _build_filter_combos(self, col_names: list[str]):
        for combo in self._filter_combos:
            combo.setParent(None)
            combo.deleteLater()
        self._filter_combos.clear()
        self._cluster_proxy.set_column_filter(-1, "")

        filterable = {"KSLabel", "group"}
        model = self._cluster_model

        for col_idx, col_name in enumerate(col_names):
            if col_name not in filterable:
                spacer = QComboBox()
                spacer.setVisible(False)
                spacer.setFixedWidth(0)
                self._filter_layout.addWidget(spacer)
                self._filter_combos.append(spacer)
                continue

            unique_vals = set()
            for row in range(model.rowCount()):
                item = model.item(row, col_idx)
                if item and item.text():
                    unique_vals.add(item.text())

            combo = QComboBox()
            combo.addItem(_ALL_FILTER)
            combo.addItems(sorted(unique_vals))
            combo.setToolTip(f"Filter by {col_name}")
            combo.setProperty("filter_col", col_idx)
            combo.currentTextChanged.connect(self._on_filter_changed)
            self._filter_layout.addWidget(combo)
            self._filter_combos.append(combo)

        has_visible = any(c.isVisible() for c in self._filter_combos)
        self._filter_row.setVisible(has_visible)

    def _on_filter_changed(self, value: str):
        combo = self.sender()
        if combo is None:
            return
        col = combo.property("filter_col")
        if col is None:
            return
        self._cluster_proxy.set_column_filter(int(col), value)

    def _find_col_by_header(self, prefix: str, exact: str | None = None) -> int | None:
        model = self._cluster_model
        for col in range(model.columnCount()):
            h = model.horizontalHeaderItem(col)
            if h is None:
                continue
            text = h.text()
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
        cluster_id_col_idx = self._find_col_by_header("", exact="cluster_id")

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
            self._on_single_cluster_selected(first_row, cluster_id_col_idx, hw_col_idx)
        else:
            self._on_multi_cluster_selected(indexes, cluster_id_col_idx, hw_col_idx)

    def _on_single_cluster_selected(self, proxy_row: int, cid_col: int, hw_col_idx: int | None):
        cid_item = self._source_item(proxy_row, cid_col)
        if cid_item is None:
            return
        try:
            cluster_id = int(cid_item.text())
        except (ValueError, TypeError):
            return

        self._show_all_good_btn.setChecked(False)
        self._multi_cluster_colors.clear()
        self._apply_cluster_colors_to_table()

        ks_ch = self._get_ks_channel_for_row(proxy_row, hw_col_idx)
        self._draw_spikes_for_cluster(cluster_id, ks_ch)
        self._jump_to_first_spike()
        self._sync_cluster_id_to_combo(cluster_id)

    def _on_multi_cluster_selected(self, indexes, cid_col: int, hw_col_idx: int | None):
        self._show_all_good_btn.setChecked(False)
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return
        ephys_plot = self.plot_container.ephys_trace_plot
        sr = ephys_plot.buffer.ephys_sr
        if sr is None or sr <= 0:
            return

        ephys_offset = ephys_plot._ephys_offset
        trial_duration = ephys_plot._trial_duration
        trial_end = ephys_offset + trial_duration if trial_duration else np.inf

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

            mask = self._spike_clusters == cluster_id
            spike_samples = self._spike_times[mask]
            spike_times_s = spike_samples.astype(np.float64) / sr

            in_trial = (spike_times_s >= ephys_offset) & (spike_times_s < trial_end)
            trial_spike_times = spike_times_s[in_trial] - ephys_offset
            trial_spike_samples = spike_samples[in_trial]

            order = np.argsort(trial_spike_times)
            trial_spike_times = trial_spike_times[order]
            trial_spike_samples = trial_spike_samples[order]

            ks_ch = self._get_ks_channel_for_row(proxy_row, hw_col_idx)
            channels = self._best_channels_for_cluster(cluster_id, ks_ch)
            cluster_entries.append((trial_spike_times, trial_spike_samples, channels, color))

            best_ch = channels[0] if channels else ks_ch
            raster_entries.append((trial_spike_times, np.full(len(trial_spike_times), best_ch, dtype=np.int32), color))

        self._apply_cluster_colors_to_table()
        ephys_plot.set_multi_cluster_spike_data(cluster_entries)

        raster = self.plot_container.raster_plot
        raster.set_multi_cluster_spike_data(raster_entries)


    def _on_filter_visible_toggled(self, checked: bool):
        if checked:
            self._apply_visible_channel_filter()
        else:
            self._cluster_proxy.set_visible_channel_filter(None, None)

    def _on_visible_channels_changed(self, _first: int, _last: int):
        if self._filter_visible_cb.isChecked():
            self._apply_visible_channel_filter()

    def _apply_visible_channel_filter(self):
        if not self.plot_container or not self.plot_container.is_ephystrace():
            self._cluster_proxy.set_visible_channel_filter(None, None)
            return
        ephys_plot = self.plot_container.ephys_trace_plot
        visible_hw = set(int(ch) for ch in ephys_plot._visible_hw_channels())
        ch_col = (
            self._find_col_by_header("", exact="ch")
            or self._find_col_by_header("", exact="ch (KS)")
        )
        self._cluster_proxy.set_visible_channel_filter(visible_hw, ch_col)

    def _toggle_show_all_good(self):
        if self._show_all_good_btn.isChecked():
            self._show_all_good_neurons()
        else:
            self._clear_multi_cluster_mode()

    def _show_all_good_neurons(self):
        if self._cluster_df is None or "group" not in self._cluster_df.columns:
            show_warning("No cluster table loaded or 'group' column missing.")
            self._show_all_good_btn.setChecked(False)
            return
        if not self.plot_container or not self.plot_container.is_ephystrace():
            show_warning("Switch to ephys trace view first.")
            self._show_all_good_btn.setChecked(False)
            return

        good_mask = self._cluster_df["group"] == "good"
        good_df = self._cluster_df[good_mask]
        if len(good_df) == 0:
            show_warning("No clusters with group 'good' found.")
            self._show_all_good_btn.setChecked(False)
            return

        ephys_plot = self.plot_container.ephys_trace_plot
        sr = ephys_plot.buffer.ephys_sr
        if sr is None or sr <= 0:
            return

        ephys_offset = ephys_plot._ephys_offset
        trial_duration = ephys_plot._trial_duration
        trial_end = ephys_offset + trial_duration if trial_duration else np.inf

        self._multi_cluster_colors.clear()
        cluster_entries = []

        for i, (_, row) in enumerate(good_df.iterrows()):
            cluster_id = int(row["cluster_id"])
            color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
            self._multi_cluster_colors[cluster_id] = color

            mask = self._spike_clusters == cluster_id
            spike_samples = self._spike_times[mask]
            spike_times_s = spike_samples.astype(np.float64) / sr

            in_trial = (spike_times_s >= ephys_offset) & (spike_times_s < trial_end)
            trial_spike_times = spike_times_s[in_trial] - ephys_offset
            trial_spike_samples = spike_samples[in_trial]

            order = np.argsort(trial_spike_times)
            trial_spike_times = trial_spike_times[order]
            trial_spike_samples = trial_spike_samples[order]

            ks_ch = int(row["ch"]) if "ch" in row.index and pd.notna(row["ch"]) else 0
            channels = self._best_channels_for_cluster(cluster_id, ks_ch)

            cluster_entries.append((trial_spike_times, trial_spike_samples, channels, color))

        self._apply_cluster_colors_to_table()
        ephys_plot.set_multi_cluster_spike_data(cluster_entries)

        raster = self.plot_container.raster_plot
        raster_entries = []
        for spike_t, _spike_s, channels, color in cluster_entries:
            best_ch = channels[0] if channels else 0
            best_arr = np.full(len(spike_t), best_ch, dtype=np.int32)
            raster_entries.append((spike_t, best_arr, color))
        raster.set_multi_cluster_spike_data(raster_entries)

        n = len(good_df)
        show_info(f"Showing spikes from {n} good neuron{'s' if n != 1 else ''}")

    def _clear_multi_cluster_mode(self):
        self._multi_cluster_colors.clear()
        self._show_all_good_btn.setChecked(False)
        self._apply_cluster_colors_to_table()
        if self.plot_container and self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.clear_spike_overlays()
        if self.plot_container:
            self.plot_container.raster_plot.clear_spike_data()
            self._populate_raster_all_spikes()

    def _apply_cluster_colors_to_table(self):
        model = self._cluster_model
        cid_col = self._find_col_by_header("", exact="cluster_id")
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
            if color:
                brush = QBrush(QColor(*color))
                r, g, b = color[:3]
                text_color = QColor(0, 0, 0) if (r * 0.299 + g * 0.587 + b * 0.114) > 150 else QColor(255, 255, 255)
                for col in range(model.columnCount()):
                    col_item = model.item(row, col)
                    if col_item:
                        col_item.setBackground(brush)
                        col_item.setForeground(QBrush(text_color))
            else:
                for col in range(model.columnCount()):
                    col_item = model.item(row, col)
                    if col_item:
                        col_item.setBackground(QBrush())
                        col_item.setForeground(QBrush())

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
        if self._spike_times is None or self._spike_clusters is None:
            return
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return

        ephys_plot = self.plot_container.ephys_trace_plot
        sr = ephys_plot.buffer.ephys_sr
        if sr is None or sr <= 0:
            return

        mask = self._spike_clusters == cluster_id
        spike_samples = self._spike_times[mask]
        spike_times_s = spike_samples.astype(np.float64) / sr

        ephys_offset = ephys_plot._ephys_offset
        trial_duration = ephys_plot._trial_duration

        start_time = ephys_offset
        trial_end = ephys_offset + trial_duration if trial_duration else np.inf
        in_trial = (spike_times_s >= start_time) & (spike_times_s < trial_end)
        trial_spike_times = spike_times_s[in_trial] - ephys_offset
        trial_spike_samples = spike_samples[in_trial]

        channels = self._best_channels_for_cluster(cluster_id, channel)
        ephys_plot.set_spike_data(trial_spike_times, trial_spike_samples, channels)

        best_ch = channels[0] if channels else channel
        raster = self.plot_container.raster_plot
        best_channels_arr = np.full(len(trial_spike_times), best_ch, dtype=np.int32)
        raster.set_spike_data(trial_spike_times, best_channels_arr)

    def _best_channels_for_cluster(self, cluster_id: int, fallback_channel: int) -> list[int]:
        if (
            self._templates is not None
            and self._channel_positions is not None
            and cluster_id < self._templates.shape[0]
        ):
            template = self._templates[cluster_id]
            n_closest = self.n_closest_spin.value()
            channel_ids, _amp, _best = self._find_best_channels(
                template, n_closest_channels=n_closest,
            )
            return channel_ids.tolist()
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
    # Ephys preprocessing panel
    # ------------------------------------------------------------------

    def _create_preprocessing_panel(self, main_layout):
        self.preproc_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.preproc_panel.setLayout(layout)

        ephys_group = QGroupBox("Ephys pre-processing")
        ephys_layout = QVBoxLayout()
        ephys_layout.setSpacing(6)
        ephys_layout.setContentsMargins(8, 12, 8, 8)
        ephys_group.setLayout(ephys_layout)
        layout.addWidget(ephys_group)

        self.ephys_subtract_mean_cb = QCheckBox("1. Subtract channel mean")
        self.ephys_subtract_mean_cb.setToolTip("Remove DC offset from each channel")
        self.ephys_car_cb = QCheckBox("2. Common average reference (CAR)")
        self.ephys_car_cb.setToolTip("Subtract median across channels at each time point")
        self.ephys_temporal_filter_cb = QCheckBox("3. Temporal filtering")
        self.ephys_temporal_filter_cb.setToolTip("3rd-order Butterworth highpass filter")
        self.ephys_hp_cutoff_edit = QLineEdit("300")
        self.ephys_hp_cutoff_edit.setFixedWidth(50)
        self.ephys_hp_cutoff_edit.setToolTip("Highpass cutoff frequency in Hz")
        self.ephys_hp_cutoff_label = QLabel("Hz highpass")
        self.ephys_whitening_cb = QCheckBox("4. (Global) channel whitening")
        self.ephys_whitening_cb.setToolTip("Decorrelate channels via SVD-based whitening")

        self._ephys_checkboxes = [
            self.ephys_subtract_mean_cb,
            self.ephys_car_cb,
            self.ephys_temporal_filter_cb,
            self.ephys_whitening_cb,
        ]

        ephys_layout.addWidget(self.ephys_subtract_mean_cb)
        ephys_layout.addWidget(self.ephys_car_cb)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.addWidget(self.ephys_temporal_filter_cb)
        filter_row.addWidget(self.ephys_hp_cutoff_edit)
        filter_row.addWidget(self.ephys_hp_cutoff_label)
        filter_row.addStretch()
        filter_row_widget = QWidget()
        filter_row_widget.setLayout(filter_row)
        ephys_layout.addWidget(filter_row_widget)

        ephys_layout.addWidget(self.ephys_whitening_cb)

        for cb in self._ephys_checkboxes:
            cb.toggled.connect(self._on_ephys_checkbox_toggled)
        self.ephys_hp_cutoff_edit.editingFinished.connect(self._on_ephys_checkbox_toggled)

        ref_label_ks = QLabel(styled_link(
            "https://www.nature.com/articles/s41592-024-02232-7#Sec10",
            "Adapted from Kilosort4 methods",
        ))
        ref_label_ks.setOpenExternalLinks(True)
        ephys_layout.addWidget(ref_label_ks)

        main_layout.addWidget(self.preproc_panel)

    def _on_ephys_checkbox_toggled(self, _checked=None):
        self._enforce_ephys_sequential()
        self._apply_ephys_preprocessing()

    def _enforce_ephys_sequential(self):
        for i, cb in enumerate(self._ephys_checkboxes):
            if i == 0:
                cb.setEnabled(True)
                continue
            prev_checked = self._ephys_checkboxes[i - 1].isChecked()
            cb.setEnabled(prev_checked)
            if not prev_checked and cb.isChecked():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)

    def _parse_hp_cutoff(self) -> float:
        try:
            return max(1.0, float(self.ephys_hp_cutoff_edit.text()))
        except (ValueError, TypeError):
            return 300.0

    def get_ephys_preprocessing_flags(self) -> dict:
        return {
            "subtract_mean": self.ephys_subtract_mean_cb.isChecked(),
            "car": self.ephys_car_cb.isChecked(),
            "temporal_filter": self.ephys_temporal_filter_cb.isChecked(),
            "hp_cutoff": self._parse_hp_cutoff(),
            "whitening": self.ephys_whitening_cb.isChecked(),
        }

    def _apply_ephys_preprocessing(self):
        if not self.plot_container:
            return
        ephys_plot = self.plot_container.ephys_trace_plot
        if ephys_plot is None:
            return
        flags = self.get_ephys_preprocessing_flags()
        ephys_plot.buffer.set_preprocessing(flags)
        if ephys_plot.current_range:
            print(f"[widgets_ephys] update_plot_content called (ephys_plot) current_range={ephys_plot.current_range}")
            ephys_plot.update_plot_content(*ephys_plot.current_range)

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
        cid_col = self._find_col_by_header("", exact="cluster_id")
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
        if self._spike_times is None or self._spike_clusters is None:
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

        spike_times_s = self._spike_times.astype(np.float64) / ephys_sr

        if self._tsgroup is None or self._tsgroup_ephys_sr != ephys_sr:
            self._tsgroup = build_tsgroup(spike_times_s, self._spike_clusters)
            self._tsgroup_ephys_sr = ephys_sr

        ds = self.app_state.dt.trial(trial)
        start_time = self.app_state.dt.get_start_time(trial)
        _, stop_time = self.app_state.get_trial_bounds()

            
            

        da = firing_rate_to_xarray(
            spike_times_s, self._spike_clusters, bin_size,
            t_start=start_time, t_stop=stop_time,
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
                set_combo_to_value(features_combo, "Firing rate")
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

        for display_name in ("Firing rate", "PCA"):
            if display_name not in features_list:
                features_list.append(display_name)
            if features_combo is not None and find_combo_index(features_combo, display_name) < 0:
                features_combo.addItem(display_name, display_name)
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
                set_combo_to_value(features_combo, "PCA")
                self.app_state.set_key_sel("features", get_combo_value(features_combo))
            self.data_widget.update_main_plot()

    def _get_any_ephys_loader(self):
        import os

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
        return SharedEphysCache.get_loader(ephys_path, stream_id)

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
        cid_col = self._find_col_by_header("", exact="cluster_id")
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
        if features_combo is not None and get_combo_value(features_combo) in ("Firing rate", "PCA"):
            features_combo.setCurrentIndex(0)

    def _on_plot_changed(self, plot_type: str):
        if plot_type == 'ephystrace':
            self.apply_probe_order()
