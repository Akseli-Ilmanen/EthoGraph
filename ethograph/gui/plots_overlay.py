"""Overlay manager for scaled overlays on pyqtgraph plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg


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
        to_remove = [n for n, e in self._entries.items() if e.host_plot is host_plot]
        for name in to_remove:
            self.remove_overlay(name)

    def clear_all(self):
        for name in list(self._entries):
            self.remove_overlay(name)

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
