"""Ephys widget — trace controls, Kilosort neuron jumping, and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

from .app_constants import CLUSTER_TABLE_MAX_HEIGHT, CLUSTER_TABLE_ROW_HEIGHT
from .plots_ephystrace import SharedEphysCache

_RAWIO_TO_DISPLAY = {
    "IntanRawIO": "Intan",
    "OpenEphysBinaryRawIO": "OpenEphys",
    "NWBIO": "NWB",
    "BlackrockRawIO": "Blackrock",
    "AxonRawIO": "Axon",
    "EdfRawIO": "EDF",
    "BrainVisionRawIO": "BrainVision",
}


class _NumericTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of lexicographically."""

    def __init__(self, sort_value: float, display_text: str):
        super().__init__(display_text)
        self._sort_value = sort_value

    def __lt__(self, other):
        if isinstance(other, _NumericTableItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


class EphysWidget(QWidget):
    """Ephys controls with toggle-button tabs: Ephys trace | Neuron jumping | Preprocessing."""

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None
        self.meta_widget = None
        self.data_widget = None

        self._cluster_df: pd.DataFrame | None = None
        self._spike_clusters: np.ndarray | None = None
        self._spike_times: np.ndarray | None = None
        self._channel_positions: np.ndarray | None = None
        self._channel_map: np.ndarray | None = None
        self._probe_channel_order: np.ndarray | None = None
        self._ephys_n_channels = 0

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_toggle_buttons(main_layout)
        self._create_trace_panel(main_layout)
        self._create_neuron_panel(main_layout)
        self._create_preprocessing_panel(main_layout)

        self._enforce_ephys_sequential()
        self._show_panel("trace")
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
            ("trace_toggle", "Ephys trace", self._toggle_trace),
            ("neuron_toggle", "Neuron jumping", self._toggle_neuron),
            ("preproc_toggle", "Preprocessing", self._toggle_preproc),
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
            "trace": (self.trace_panel, self.trace_toggle),
            "neuron": (self.neuron_panel, self.neuron_toggle),
            "preproc": (self.preproc_panel, self.preproc_toggle),
        }
        for name, (panel, toggle) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)
        self._refresh_layout()

    def _toggle_trace(self):
        self._show_panel("trace" if self.trace_toggle.isChecked() else "neuron")

    def _toggle_neuron(self):
        self._show_panel("neuron" if self.neuron_toggle.isChecked() else "trace")

    def _toggle_preproc(self):
        self._show_panel("preproc" if self.preproc_toggle.isChecked() else "trace")

    def _refresh_layout(self):
        if self.meta_widget and hasattr(self.meta_widget, "collapsible_widgets"):
            for collapsible in self.meta_widget.collapsible_widgets:
                if hasattr(collapsible, "content_widget"):
                    content = collapsible.content_widget
                    if content and self in content.findChildren(QWidget):
                        collapsible.collapse()
                        QApplication.processEvents()
                        collapsible.expand()

    # ------------------------------------------------------------------
    # Ephys trace panel (channel, multichannel, gain, range)
    # ------------------------------------------------------------------

    def _create_trace_panel(self, main_layout):
        self.trace_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.trace_panel.setLayout(layout)

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

        # Multichannel checkbox
        self.ephys_multichannel_cb = QCheckBox("Multi-channel")
        self.ephys_multichannel_cb.setObjectName("ephys_multichannel_cb")
        self.ephys_multichannel_cb.setToolTip("Show all channels stacked (Ctrl+X to toggle)")
        self.ephys_multichannel_cb.stateChanged.connect(self._on_ephys_multichannel_changed)

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

        ch_row = QHBoxLayout()
        ch_row.addWidget(self.ephys_channel_label)
        ch_row.addWidget(self.ephys_channel_spin)
        ch_row.addWidget(self.ephys_multichannel_cb)
        ch_row.addWidget(self.ephys_gain_label)
        ch_row.addWidget(self.ephys_gain_spin)
        ch_row.addStretch()
        group_layout.addLayout(ch_row)

        # Channel range slider
        self.ephys_range_label = QLabel("Channel range:")
        self.ephys_range_slider = QRangeSlider(Qt.Horizontal)
        self.ephys_range_slider.setObjectName("ephys_range_slider")
        self.ephys_range_slider.setRange(0, 0)
        self.ephys_range_slider.setValue((0, 0))
        self.ephys_range_slider.setToolTip(
            "Channel range for multi-channel view (Wheel to scroll, Shift+Wheel to resize)"
        )
        self.ephys_range_slider.valueChanged.connect(self._on_ephys_range_changed)

        range_row = QHBoxLayout()
        range_row.addWidget(self.ephys_range_label)
        range_row.addWidget(self.ephys_range_slider)
        group_layout.addLayout(range_row)

        main_layout.addWidget(self.trace_panel)

    # ------------------------------------------------------------------
    # Ephys trace handlers
    # ------------------------------------------------------------------

    def configure_ephys_trace_plot(self):
        from .plots_ephystrace import SharedEphysCache

        ephys_path, stream_id, channel_idx = self.app_state.get_ephys_source()
        if not ephys_path:
            return
        loader = SharedEphysCache.get_loader(ephys_path, stream_id)
        if loader is None:
            return
        self.plot_container.ephys_trace_plot.set_loader(loader, channel_idx)

        offset = float(self.app_state.ds.attrs.get('trial_onset', 0.0))
        trial_duration = float(self.app_state.ds.time.values[-1]) + 1.0 / self.app_state.ds.attrs['fps']
        self.plot_container.ephys_trace_plot.set_ephys_offset(offset, trial_duration)

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
        if n_ch > 1:
            self.ephys_multichannel_cb.show()
        else:
            self.ephys_multichannel_cb.hide()

        self.ephys_range_slider.blockSignals(True)
        self.ephys_range_slider.setRange(0, max(0, n_ch - 1))
        self.ephys_range_slider.setValue((0, max(0, n_ch - 1)))
        self.ephys_range_slider.blockSignals(False)
        if n_ch > 10:
            self._show_range_slider()
        else:
            self.ephys_range_label.hide()
            self.ephys_range_slider.hide()

    def _on_ephys_channel_changed(self, channel: int):
        feature_sel = getattr(self.app_state, 'features_sel', None)
        source_map = getattr(self.app_state, 'ephys_source_map', {})
        if feature_sel not in source_map:
            return
        filename, stream_id, _ = source_map[feature_sel]
        source_map[feature_sel] = (filename, stream_id, channel)

        if self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.set_channel(channel)
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

    def _on_ephys_multichannel_changed(self, state: int):
        enabled = state == Qt.Checked
        self.ephys_channel_label.setVisible(not enabled)
        self.ephys_channel_spin.setVisible(not enabled)
        self.ephys_range_slider.setEnabled(enabled)

        if self.plot_container and self.plot_container.is_ephystrace():
            ephys_plot = self.plot_container.ephys_trace_plot
            ephys_plot.set_multichannel(enabled)
            if enabled:
                ephys_plot.auto_channel_spacing()
            ephys_plot.autoscale()
            xmin, xmax = self.plot_container.get_current_xlim()
            if self.data_widget:
                self.data_widget.update_main_plot(t0=xmin, t1=xmax)


    def _on_ephys_gain_changed(self, value: float):
        if self.plot_container and self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.buffer.display_gain = value
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

    def _on_ephys_range_changed(self, value):
        ch_start, ch_end = int(value[0]), int(value[1])
        if self.plot_container and self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.set_channel_range(ch_start, ch_end)

    def _show_range_slider(self):
        self.ephys_range_label.show()
        self.ephys_range_slider.show()
        self.ephys_range_slider.setEnabled(self.ephys_multichannel_cb.isChecked())

    def hide_ephys_channel_controls(self):
        self.ephys_channel_label.hide()
        self.ephys_channel_spin.hide()
        self.ephys_multichannel_cb.hide()
        self.ephys_gain_label.hide()
        self.ephys_gain_spin.hide()
        self.ephys_range_label.hide()
        self.ephys_range_slider.hide()
        if self.plot_container and hasattr(self.plot_container, 'ephys_trace_plot'):
            self.plot_container.ephys_trace_plot.set_multichannel(False)
        self.ephys_multichannel_cb.blockSignals(True)
        self.ephys_multichannel_cb.setChecked(False)
        self.ephys_multichannel_cb.blockSignals(False)

    # ------------------------------------------------------------------
    # Neuron jumping panel (Kilosort)
    # ------------------------------------------------------------------

    def _create_neuron_panel(self, main_layout):
        self.neuron_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.neuron_panel.setLayout(layout)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("kilosort4_folder:"))
        self.kilosort_folder_edit = QLineEdit()
        self.kilosort_folder_edit.setPlaceholderText("kilosort4 output folder")
        self.kilosort_folder_edit.returnPressed.connect(self._load_kilosort_folder)
        path_row.addWidget(self.kilosort_folder_edit)

        self.cluster_browse_btn = QPushButton("Browse")
        self.cluster_browse_btn.clicked.connect(self._browse_kilosort_folder)
        path_row.addWidget(self.cluster_browse_btn)

        self.cluster_load_btn = QPushButton("Load")
        self.cluster_load_btn.clicked.connect(self._load_kilosort_folder)
        path_row.addWidget(self.cluster_load_btn)
        layout.addLayout(path_row)

        self.cluster_table = QTableWidget()
        self.cluster_table.setColumnCount(5)
        self.cluster_table.setHorizontalHeaderLabels(["cluster_id", "ch", "KSLabel", "group", "fr"])
        self.cluster_table.verticalHeader().setVisible(False)
        self.cluster_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cluster_table.setSelectionMode(QAbstractItemView.SingleSelection)
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
            QTableWidget { gridline-color: transparent; }
            QTableWidget::item { padding: 0px 2px; }
            QHeaderView::section { padding: 0px 2px; }
        """)
        self.cluster_table.itemSelectionChanged.connect(self._on_cluster_row_selected)
        layout.addWidget(self.cluster_table)

        main_layout.addWidget(self.neuron_panel)

    def populate_ephys_default_path(self):
        ephys_folder = getattr(self.app_state, 'ephys_folder', None)
        if not ephys_folder:
            return
        default_folder = Path(ephys_folder) / "kilosort4"
        self.kilosort_folder_edit.setText(str(default_folder))
        if default_folder.is_dir():
            self._load_kilosort_folder()

    def _browse_kilosort_folder(self):
        start_dir = getattr(self.app_state, 'ephys_folder', '') or ''
        folder = QFileDialog.getExistingDirectory(
            self, "Select kilosort4 output folder", start_dir,
        )
        if folder:
            self.kilosort_folder_edit.setText(folder)
            self._load_kilosort_folder()

    def _load_kilosort_folder(self):
        path_str = self.kilosort_folder_edit.text().strip()
        if not path_str:
            return

        folder = Path(path_str)
        if not folder.is_dir():
            show_warning(f"Folder not found: {folder}")
            return

        self._cluster_df = self._load_file(folder / "cluster_info.tsv", pd.read_csv, sep='\t')
        self._populate_cluster_table(self._cluster_df)

        self._spike_clusters = self._load_file(folder / "spike_clusters.npy", np.load, flatten=True)
        self._spike_times = self._load_file(folder / "spike_times.npy", np.load, flatten=True)
        self._channel_positions = self._load_file(folder / "channel_positions.npy", np.load)
        self._channel_map = self._load_file(folder / "channel_map.npy", np.load, flatten=True)
        self._reorder_probe_by_position()

    def _load_file(self, path: Path, loader, flatten: bool = False, **kwargs):
        if not path.exists():
            return None
        try:
            data = loader(path, **kwargs)
            return data.flatten() if flatten else data
        except Exception as e:
            show_warning(f"Failed to load {path.name}: {e}")
            return None

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

    def _populate_cluster_table(self, df: pd.DataFrame):
        display_cols = ["cluster_id", "ch", "KSLabel", "group", "Amplitude", "fr", "n_spikes"]
        numeric_cols = {"cluster_id", "ch", "Amplitude", "fr", "n_spikes"}
        available_cols = [c for c in display_cols if c in df.columns]

        hw_label = self._get_hardware_label()

        has_ch = "ch" in available_cols
        if has_ch:
            ch_idx = available_cols.index("ch")
            available_cols[ch_idx] = "ch (KS)"
            available_cols.insert(ch_idx + 1, f"ch ({hw_label})")

        self.cluster_table.setSortingEnabled(False)
        self.cluster_table.setColumnCount(len(available_cols))
        self.cluster_table.setHorizontalHeaderLabels(available_cols)
        self.cluster_table.setRowCount(len(df))

        for row_idx in range(len(df)):
            col_idx = 0
            for col_name in display_cols:
                if col_name not in df.columns:
                    continue
                value = df.iloc[row_idx][col_name]

                if col_name == "ch" and has_ch:
                    ks_ch = int(value) if pd.notna(value) else 0
                    hw_names = self.get_hw_names(self._channel_map)
                    hw_name = hw_names.get(ks_ch) if hw_names else None
                    hw_display = hw_name if hw_name else str(ks_ch)

                    ks_item = _NumericTableItem(float(ks_ch), str(ks_ch))
                    self.cluster_table.setItem(row_idx, col_idx, ks_item)
                    col_idx += 1

                    hw_item = _NumericTableItem(float(ks_ch), hw_display)
                    hw_item.setData(Qt.UserRole, ks_ch)
                    self.cluster_table.setItem(row_idx, col_idx, hw_item)
                    col_idx += 1
                elif col_name in numeric_cols and pd.notna(value):
                    display = f"{float(value):.2f}" if col_name == "fr" else str(int(value))
                    item = _NumericTableItem(float(value), display)
                    self.cluster_table.setItem(row_idx, col_idx, item)
                    col_idx += 1
                elif pd.isna(value):
                    self.cluster_table.setItem(row_idx, col_idx, QTableWidgetItem(""))
                    col_idx += 1
                else:
                    self.cluster_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
                    col_idx += 1

        self.cluster_table.setSortingEnabled(True)

    def _on_cluster_row_selected(self):
        selected = self.cluster_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        hw_col_idx = None
        cluster_id_col_idx = None
        for col in range(self.cluster_table.columnCount()):
            header_text = self.cluster_table.horizontalHeaderItem(col).text()
            if header_text.startswith("ch (") and header_text != "ch (KS)":
                print(header_text)
                hw_col_idx = col
            if header_text == "cluster_id":
                cluster_id_col_idx = col

        if hw_col_idx is not None:
            ch_item = self.cluster_table.item(row, hw_col_idx)
            if ch_item is not None:
                channel = ch_item.data(Qt.UserRole)
                if channel is not None:
                    self._apply_ephys_channel(int(channel))

        if hw_col_idx is not None:
            self._center_range_slider_on_channel(
                self._get_ks_channel_for_row(row, hw_col_idx)
            )

        if cluster_id_col_idx is not None:
            cid_item = self.cluster_table.item(row, cluster_id_col_idx)
            if cid_item is not None:
                try:
                    cluster_id = int(cid_item.text())
                    from .phy_bridge import phy_select_clusters
                    phy_select_clusters([cluster_id])
                    ks_ch = self._get_ks_channel_for_row(row, hw_col_idx)
                    self._draw_spikes_for_cluster(cluster_id, ks_ch)
                    self._jump_to_first_spike()
                except (ValueError, TypeError):
                    pass

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

        trial_start = ephys_offset
        trial_end = ephys_offset + trial_duration if trial_duration else np.inf
        in_trial = (spike_times_s >= trial_start) & (spike_times_s < trial_end)
        trial_spike_times = spike_times_s[in_trial] - ephys_offset
        trial_spike_samples = spike_samples[in_trial]

        ephys_plot.set_spike_data(trial_spike_times, trial_spike_samples, [channel])

    def _center_range_slider_on_channel(self, channel: int):
        slider = self.ephys_range_slider
        slider_min, slider_max = slider.minimum(), slider.maximum()
        cur_lo, cur_hi = int(slider.value()[0]), int(slider.value()[1])
        span = cur_hi - cur_lo
        print(f"[range-slider] ch={channel} slider=[{slider_min},{slider_max}] "
              f"cur=({cur_lo},{cur_hi}) span={span} enabled={slider.isEnabled()}")
        if span >= slider_max - slider_min:
            return
        half = span // 2
        new_lo = max(slider_min, min(channel - half, slider_max - span))
        new_hi = new_lo + span
        print(f"[range-slider] -> setting ({new_lo},{new_hi})")
        slider.setValue((new_lo, new_hi))

    def _get_ks_channel_for_row(self, row: int, hw_col_idx: int | None) -> int:
        if hw_col_idx is not None:
            ch_item = self.cluster_table.item(row, hw_col_idx)
            if ch_item is not None:
                val = ch_item.data(Qt.UserRole)
                if val is not None:
                    return int(val)
        return self.ephys_channel_spin.value()

    def _apply_ephys_channel(self, channel: int):
        feature_sel = getattr(self.app_state, 'features_sel', None)
        source_map = getattr(self.app_state, 'ephys_source_map', {})
        if not feature_sel or feature_sel not in source_map:
            return

        filename, stream_id, _ = source_map[feature_sel]
        source_map[feature_sel] = (filename, stream_id, channel)

        self.ephys_channel_spin.blockSignals(True)
        self.ephys_channel_spin.setValue(channel)
        self.ephys_channel_spin.blockSignals(False)

        if self.plot_container and self.plot_container.is_ephystrace():
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

        ref_label_ks = QLabel(
            '<a href="https://www.nature.com/articles/s41592-024-02232-7#Sec10" '
            'style="color: #87CEEB; text-decoration: none;">Adapted from Kilosort4 methods</a>'
        )
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
            ephys_plot.update_plot_content(*ephys_plot.current_range)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container
        plot_container.plot_changed.connect(self._on_plot_changed)
        ephys_plot = plot_container.ephys_trace_plot
        if ephys_plot is not None:
            ephys_plot.channel_scroll_requested.connect(self._on_channel_scroll)
            ephys_plot.gain_scroll_requested.connect(self._on_gain_scroll)
            ephys_plot.range_resize_requested.connect(self._on_range_resize)

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    def set_data_widget(self, data_widget):
        self.data_widget = data_widget

    def _on_channel_scroll(self, delta: int):
        slider = self.ephys_range_slider
        cur_lo, cur_hi = int(slider.value()[0]), int(slider.value()[1])
        span = cur_hi - cur_lo
        if span >= slider.maximum() - slider.minimum():
            return
        new_lo = max(slider.minimum(), min(cur_lo + delta, slider.maximum() - span))
        new_hi = new_lo + span
        slider.setValue((new_lo, new_hi))

    def _on_gain_scroll(self, delta: int):
        spin = self.ephys_gain_spin
        new_val = round(spin.value() + delta * 0.1, 1)
        spin.setValue(max(spin.minimum(), min(new_val, spin.maximum())))

    def _on_range_resize(self, delta: int):
        slider = self.ephys_range_slider
        cur_lo, cur_hi = int(slider.value()[0]), int(slider.value()[1])
        new_lo = max(slider.minimum(), cur_lo - delta)
        new_hi = min(slider.maximum(), cur_hi + delta)
        if new_hi - new_lo < 1:
            return
        slider.setValue((new_lo, new_hi))

    def _on_plot_changed(self, plot_type: str):
        if plot_type == 'ephystrace':
            self.apply_probe_order()
