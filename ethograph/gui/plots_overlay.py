"""Overlay manager for scaled and ViewBox-based overlays on pyqtgraph plots."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QTimer


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


@dataclass
class _ViewBoxOverlayEntry:
    name: str
    host_plot: object
    viewbox: pg.ViewBox
    host_items: list = field(default_factory=list)
    geometry_updater: object | None = None


class OverlayManager:
    """Manages overlays on pyqtgraph plots.

    Two overlay types:

    **Scaled overlays** — data rescaled into the host's y-range with custom
    right-axis tick labels.  Good for dimensionless indicators (confidence).

    **ViewBox overlays** — a separate ``pg.ViewBox`` with its own y-axis,
    linked to the host's x-axis.  Good for dual-axis plots (envelopes,
    threshold lines).  The caller adds items to the returned ViewBox.

    Usage::

        mgr = OverlayManager()
        host.vb.sigYRangeChanged.connect(lambda: mgr.rescale_for_plot(host))

        # Scaled overlay
        item = pg.PlotCurveItem(...)
        mgr.add_scaled_overlay("confidence", host, item, time, data)

        # ViewBox overlay
        vb = mgr.add_viewbox_overlay("envelope", host, axis_label="Env")
        vb.addItem(pg.PlotDataItem(time, data, ...))
        vb.setYRange(0, 1)

        mgr.remove_overlay("confidence")
    """

    def __init__(self):
        self._entries: dict[str, _ScaledOverlayEntry] = {}
        self._vb_entries: dict[str, _ViewBoxOverlayEntry] = {}
        self._rescaling = False

    # ------------------------------------------------------------------
    # Scaled overlays (existing API)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # ViewBox overlays
    # ------------------------------------------------------------------

    def add_viewbox_overlay(
        self,
        name: str,
        host_plot,
        *,
        host_items: list | None = None,
        axis_label: str = "",
        axis_color: str | None = None,
    ) -> pg.ViewBox:
        """Create a ViewBox overlay with its own right axis.

        Returns the ``ViewBox``.  The caller adds items (curves, threshold
        lines) and sets the y-range as needed.

        Parameters
        ----------
        name
            Unique overlay name (used for ``remove_overlay``).
        host_plot
            The ``BasePlot`` that hosts this overlay.
        host_items
            Optional items added directly to the host (not the ViewBox)
            that should be removed when the overlay is removed.
        axis_label
            Label for the right axis.
        axis_color
            Color string for the right axis pen and text.
        """
        self.remove_overlay(name)

        vb = pg.ViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        host_plot.plot_item.scene().addItem(vb)
        vb.setXLink(host_plot.plot_item.vb)

        host_plot.plot_item.showAxis('right')
        right_axis = host_plot.plot_item.getAxis('right')
        if axis_color:
            right_axis.setPen(pg.mkPen(color=axis_color))
            right_axis.setTextPen(pg.mkPen(color=axis_color))
        if axis_label:
            right_axis.setLabel(axis_label, color=axis_color or 'k')
        right_axis.linkToView(vb)

        def update_geometry():
            rect = host_plot.plot_item.vb.sceneBoundingRect()
            if rect.width() > 0 and rect.height() > 0:
                vb.setGeometry(rect)

        QTimer.singleShot(0, update_geometry)
        QTimer.singleShot(100, update_geometry)
        host_plot.plot_item.vb.sigResized.connect(update_geometry)

        entry = _ViewBoxOverlayEntry(
            name=name,
            host_plot=host_plot,
            viewbox=vb,
            host_items=list(host_items or []),
            geometry_updater=update_geometry,
        )
        self._vb_entries[name] = entry
        return vb

    def get_viewbox(self, name: str) -> pg.ViewBox | None:
        entry = self._vb_entries.get(name)
        return entry.viewbox if entry else None

    # ------------------------------------------------------------------
    # Common API
    # ------------------------------------------------------------------

    def remove_overlay(self, name: str):
        scaled = self._entries.pop(name, None)
        if scaled is not None:
            try:
                scaled.host_plot.vb.removeItem(scaled.item)
            except (RuntimeError, AttributeError, ValueError):
                pass
            self._update_right_axis(scaled.host_plot)
            return

        vb_entry = self._vb_entries.pop(name, None)
        if vb_entry is None:
            return

        host = vb_entry.host_plot

        if vb_entry.geometry_updater:
            try:
                host.plot_item.vb.sigResized.disconnect(vb_entry.geometry_updater)
            except (RuntimeError, TypeError):
                pass

        for item in vb_entry.host_items:
            try:
                host.removeItem(item)
            except (RuntimeError, ValueError):
                pass

        try:
            host.plot_item.scene().removeItem(vb_entry.viewbox)
        except (RuntimeError, AttributeError, ValueError):
            pass

        if not self._any_overlay_on_plot(host):
            try:
                host.plot_item.hideAxis('right')
            except (RuntimeError, AttributeError):
                pass
        else:
            self._update_right_axis(host)

    def has_overlay(self, name: str) -> bool:
        return name in self._entries or name in self._vb_entries

    def clear_plot(self, host_plot):
        to_remove = [n for n, e in self._entries.items() if e.host_plot is host_plot]
        to_remove += [n for n, e in self._vb_entries.items() if e.host_plot is host_plot]
        for name in to_remove:
            self.remove_overlay(name)

    def clear_all(self):
        for name in list(self._entries) + list(self._vb_entries):
            self.remove_overlay(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _any_overlay_on_plot(self, host_plot) -> bool:
        for e in self._entries.values():
            if e.host_plot is host_plot:
                return True
        for e in self._vb_entries.values():
            if e.host_plot is host_plot:
                return True
        return False

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
        scaled_on_plot = [e for e in self._entries.values() if e.host_plot is host_plot]
        right_axis = host_plot.plot_item.getAxis('right')

        if scaled_on_plot:
            entry = scaled_on_plot[0]
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
        elif not self._any_overlay_on_plot(host_plot):
            right_axis.hide()
