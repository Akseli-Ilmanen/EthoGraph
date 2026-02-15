"""Axes control widget for plot settings and overlay management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyqtgraph as pg
from napari.viewer import Viewer
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

HEATMAP_COLORMAPS = [
    "RdBu_r",
    "viridis",
    "inferno",
    "coolwarm",
    "plasma",
    "magma",
    "cividis",
]


# ---------------------------------------------------------------------------
# OverlayManager — manages scaled overlays with independent autoscale
# ---------------------------------------------------------------------------

@dataclass
class _ScaledOverlayEntry:
    name: str
    item: pg.PlotCurveItem
    host_plot: object
    raw_time: np.ndarray
    raw_data: np.ndarray
    data_min: float
    data_max: float
    tick_format: str = "{:.3g}"


class OverlayManager:
    """Manages scaled overlays on pyqtgraph plots with correct autoscale.

    Overlays (confidence, envelope) are added to the host ViewBox with
    ``ignoreBounds=True`` so pyqtgraph's autorange considers only the
    primary data.  When the main y-range changes, the manager linearly
    rescales every overlay to span the new range and updates the right-axis
    tick labels.

    Usage from PlotContainer::

        mgr = OverlayManager()
        host.vb.sigYRangeChanged.connect(lambda: mgr.rescale_for_plot(host))
        item = pg.PlotCurveItem(...)
        mgr.add_scaled_overlay("envelope", host, item, time, data)
        mgr.remove_overlay("envelope")
    """

    def __init__(self):
        self._entries: dict[str, _ScaledOverlayEntry] = {}
        self._rescaling = False

    def add_scaled_overlay(
        self,
        name: str,
        host_plot,
        item: pg.PlotCurveItem,
        raw_time: np.ndarray,
        raw_data: np.ndarray,
        *,
        data_min: float | None = None,
        data_max: float | None = None,
        tick_format: str = "{:.3g}",
    ):
        """Register *item* as a scaled overlay on *host_plot*.

        The item is added to the host's ViewBox with ``ignoreBounds=True``
        and immediately scaled to the current y-range.
        """
        self.remove_overlay(name)

        if data_min is None:
            data_min = float(np.nanmin(raw_data))
        if data_max is None:
            data_max = float(np.nanmax(raw_data))

        entry = _ScaledOverlayEntry(
            name=name,
            item=item,
            host_plot=host_plot,
            raw_time=raw_time,
            raw_data=raw_data,
            data_min=data_min,
            data_max=data_max,
            tick_format=tick_format,
        )
        self._entries[name] = entry

        host_plot.vb.addItem(item, ignoreBounds=True)

        main_range = host_plot.plot_item.viewRange()[1]
        self._rescale_entry(entry, main_range)
        self._update_right_axis(host_plot)

    def remove_overlay(self, name: str):
        """Remove a registered overlay by name."""
        entry = self._entries.pop(name, None)
        if entry is None:
            return
        try:
            entry.host_plot.vb.removeItem(entry.item)
        except (RuntimeError, AttributeError, ValueError):
            pass
        self._update_right_axis(entry.host_plot)

    def update_overlay_data(
        self,
        name: str,
        raw_time: np.ndarray,
        raw_data: np.ndarray,
        *,
        data_min: float | None = None,
        data_max: float | None = None,
    ):
        """Replace the raw data for a scaled overlay and rescale."""
        entry = self._entries.get(name)
        if entry is None:
            return
        entry.raw_time = raw_time
        entry.raw_data = raw_data
        entry.data_min = data_min if data_min is not None else float(np.nanmin(raw_data))
        entry.data_max = data_max if data_max is not None else float(np.nanmax(raw_data))

        main_range = entry.host_plot.plot_item.viewRange()[1]
        self._rescale_entry(entry, main_range)
        self._update_right_axis(entry.host_plot)

    def rescale_for_plot(self, host_plot):
        """Rescale all overlays on *host_plot* to its current y-range.

        Connect this to ``host.vb.sigYRangeChanged``.
        """
        if self._rescaling:
            return
        self._rescaling = True
        try:
            main_range = host_plot.plot_item.viewRange()[1]
            for entry in self._entries.values():
                if entry.host_plot is host_plot:
                    self._rescale_entry(entry, main_range)
            self._update_right_axis(host_plot)
        finally:
            self._rescaling = False

    def has_overlay(self, name: str) -> bool:
        return name in self._entries

    def clear_plot(self, host_plot):
        """Remove all overlays from a specific host plot."""
        to_remove = [n for n, e in self._entries.items() if e.host_plot is host_plot]
        for name in to_remove:
            self.remove_overlay(name)

    def clear_all(self):
        for name in list(self._entries):
            self.remove_overlay(name)

    # --- private helpers ---

    def _rescale_entry(self, entry: _ScaledOverlayEntry, main_range: list):
        data_range = entry.data_max - entry.data_min
        if data_range <= 0:
            data_range = 1.0
        main_ymin, main_ymax = main_range
        main_span = main_ymax - main_ymin
        if main_span <= 0:
            return
        scaled = ((entry.raw_data - entry.data_min) / data_range) * main_span + main_ymin
        entry.item.setData(entry.raw_time, scaled)

    def _update_right_axis(self, host_plot):
        entries_on_plot = [e for e in self._entries.values() if e.host_plot is host_plot]
        try:
            right_axis = host_plot.plot_item.getAxis('right')
        except Exception:
            return

        if not entries_on_plot:
            right_axis.hide()
            return

        entry = entries_on_plot[0]
        right_axis.setStyle(showValues=True)
        right_axis.show()

        main_range = host_plot.plot_item.viewRange()[1]
        data_range = entry.data_max - entry.data_min
        if data_range <= 0:
            data_range = 1.0

        ticks = []
        for val in np.linspace(entry.data_min, entry.data_max, 5):
            main_val = ((val - entry.data_min) / data_range) * (main_range[1] - main_range[0]) + main_range[0]
            ticks.append((main_val, entry.tick_format.format(val)))
        right_axis.setTicks([ticks])


class AxesWidget(QWidget):
    """Axes controls for line plots and general plot settings.

    Keys used in gui_settings.yaml (via app_state):
      - ymin, ymax
      - percentile_ylim
      - window_size
      - lock_axes
    """

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        group_box = QGroupBox("Axes Controls")
        group_layout = QGridLayout()
        group_box.setLayout(group_layout)
        main_layout.addWidget(group_box)

        self.ymin_edit = QLineEdit()
        self.ymax_edit = QLineEdit()

        self.percentile_ylim_edit = QLineEdit()
        validator = QDoubleValidator(95.0, 100, 2, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.percentile_ylim_edit.setValidator(validator)

        self.window_s_edit = QLineEdit()

        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset")

        self.autoscale_checkbox = QCheckBox("Autoscale Y")
        self.lock_axes_checkbox = QCheckBox("Lock Axes")

        row = 0
        group_layout.addWidget(QLabel("Y min:"), row, 0)
        group_layout.addWidget(self.ymin_edit, row, 1)
        group_layout.addWidget(QLabel("Y max:"), row, 2)
        group_layout.addWidget(self.ymax_edit, row, 3)

        row += 1
        group_layout.addWidget(QLabel("Percentile Y-lim:"), row, 0)
        group_layout.addWidget(self.percentile_ylim_edit, row, 1)
        group_layout.addWidget(QLabel("Window (s):"), row, 2)
        group_layout.addWidget(self.window_s_edit, row, 3)

        row += 1
        group_layout.addWidget(self.autoscale_checkbox, row, 0)
        group_layout.addWidget(self.lock_axes_checkbox, row, 1)
        group_layout.addWidget(self.apply_button, row, 2)
        group_layout.addWidget(self.reset_button, row, 3)

        self.ymin_edit.editingFinished.connect(self._on_edited)
        self.ymax_edit.editingFinished.connect(self._on_edited)
        self.percentile_ylim_edit.editingFinished.connect(self._on_edited)
        self.window_s_edit.editingFinished.connect(self._on_edited)

        self.apply_button.clicked.connect(self._on_edited)
        self.reset_button.clicked.connect(self._reset_to_defaults)
        self.autoscale_checkbox.toggled.connect(self._autoscale_y_toggle)
        self.lock_axes_checkbox.toggled.connect(self._on_lock_axes_toggled)

        self._create_heatmap_controls(main_layout)
        self._restore_or_set_default_selections()

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def _create_heatmap_controls(self, main_layout):
        hm_group = QGroupBox("Heatmap Display")
        hm_layout = QGridLayout()
        hm_group.setLayout(hm_layout)
        main_layout.addWidget(hm_group)

        hm_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.heatmap_colormap_combo = QComboBox()
        self.heatmap_colormap_combo.addItems(HEATMAP_COLORMAPS)
        self.heatmap_colormap_combo.currentTextChanged.connect(self._on_heatmap_colormap_changed)
        hm_layout.addWidget(self.heatmap_colormap_combo, 0, 1)

        hm_layout.addWidget(QLabel("Excl. percentile:"), 0, 2)
        self.heatmap_percentile_spin = QDoubleSpinBox()
        self.heatmap_percentile_spin.setRange(50.0, 100.0)
        self.heatmap_percentile_spin.setSingleStep(1.0)
        self.heatmap_percentile_spin.setDecimals(1)
        self.heatmap_percentile_spin.setToolTip("Percentile of abs(z-scores) for symmetric color range")
        self.heatmap_percentile_spin.valueChanged.connect(self._on_heatmap_percentile_changed)
        hm_layout.addWidget(self.heatmap_percentile_spin, 0, 3)

    def _restore_or_set_default_selections(self):
        for attr, edit in [
            ("ymin", self.ymin_edit),
            ("ymax", self.ymax_edit),
            ("percentile_ylim", self.percentile_ylim_edit),
            ("window_size", self.window_s_edit),
        ]:
            value = getattr(self.app_state, attr, None)
            if value is None:
                value = self.app_state.get_with_default(attr)
                setattr(self.app_state, attr, value)
            edit.setText("" if value is None else str(value))

        lock_axes = self.app_state.get_with_default("lock_axes")
        self.lock_axes_checkbox.setChecked(lock_axes)

        cmap = self.app_state.get_with_default("heatmap_colormap")
        if cmap in HEATMAP_COLORMAPS:
            self.heatmap_colormap_combo.setCurrentText(cmap)

        self.heatmap_percentile_spin.setValue(
            self.app_state.get_with_default("heatmap_exclusion_percentile")
        )

    def _parse_float(self, text: str) -> Optional[float]:
        s = (text or "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _autoscale_y_toggle(self, checked: bool):
        if not self.plot_container:
            return

        if checked:
            self.plot_container.vb.enableAutoRange(x=False, y=True)
            self.lock_axes_checkbox.setChecked(False)
        else:
            self.plot_container.vb.disableAutoRange()

    def _on_lock_axes_toggled(self, checked: bool):
        self.app_state.lock_axes = checked
        if self.plot_container:
            self.plot_container.toggle_axes_lock()
        if checked:
            self.autoscale_checkbox.setChecked(False)

    def _on_edited(self):
        if not self.plot_container:
            return

        edits = {
            "ymin": self.ymin_edit,
            "ymax": self.ymax_edit,
            "percentile_ylim": self.percentile_ylim_edit,
            "window_size": self.window_s_edit,
        }

        values = {}
        for attr, edit in edits.items():
            val = self._parse_float(edit.text())
            if val is None:
                val = self.app_state.get_with_default(attr)
            values[attr] = val
            setattr(self.app_state, attr, val)

        user_set_yrange = (self._parse_float(self.ymin_edit.text()) is not None or
                           self._parse_float(self.ymax_edit.text()) is not None)

        if not self.plot_container.is_spectrogram() and not self.autoscale_checkbox.isChecked():
            if user_set_yrange:
                current_plot = self.plot_container.get_current_plot()
                if hasattr(current_plot, 'vb'):
                    current_plot.vb.setLimits(yMin=None, yMax=None, minYRange=None, maxYRange=None)
            self.plot_container.apply_y_range(values["ymin"], values["ymax"])

        if not user_set_yrange and not self.autoscale_checkbox.isChecked() and "percentile_ylim" in values:
            current_plot = self.plot_container.get_current_plot()
            if hasattr(current_plot, '_apply_zoom_constraints'):
                current_plot._apply_zoom_constraints()

        new_xmin, new_xmax = self._calculate_new_window_size()
        if new_xmin is not None and new_xmax is not None:
            self.plot_container.set_x_range(mode='preserve', curr_xlim=(new_xmin, new_xmax))

    def _calculate_new_window_size(self):
        if not self.plot_container:
            return None, None

        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return None, None

        current_time = self.app_state.current_frame / self.app_state.ds.fps
        window_size = self.app_state.get_with_default("window_size")
        half_window = window_size / 2

        new_xmin = current_time - half_window
        new_xmax = current_time + half_window
        return new_xmin, new_xmax

    def _on_heatmap_colormap_changed(self, colormap_name: str):
        self.app_state.heatmap_colormap = colormap_name
        if self.plot_container:
            heatmap = self.plot_container.heatmap_plot
            heatmap.update_colormap(colormap_name)
            if self.plot_container.is_heatmap():
                heatmap._clear_buffer()
                heatmap.update_plot_content()

    def _on_heatmap_percentile_changed(self, value: float):
        self.app_state.heatmap_exclusion_percentile = value
        if self.plot_container and self.plot_container.is_heatmap():
            heatmap = self.plot_container.heatmap_plot
            heatmap._clear_buffer()
            heatmap.update_plot_content()

    def _reset_to_defaults(self):
        for attr, edit in [
            ("ymin", self.ymin_edit),
            ("ymax", self.ymax_edit),
            ("percentile_ylim", self.percentile_ylim_edit),
            ("window_size", self.window_s_edit),
        ]:
            value = self.app_state.get_with_default(attr)
            edit.setText("" if value is None else str(value))
            setattr(self.app_state, attr, value)

        self.lock_axes_checkbox.setChecked(False)
        self.app_state.lock_axes = False

        self._on_edited()


# Backward compatibility alias
PlotsWidget = AxesWidget
