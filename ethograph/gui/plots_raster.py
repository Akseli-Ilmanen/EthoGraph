"""Spike raster plot — one dot per spike at (time, best_channel_y).

Uses pg.ScatterPlotItem for hardware-accelerated rendering with
viewport-culled, debounced updates so the Qt event queue never floods
during zoom/pan.  Supports single-color (all spikes gray) and per-cluster
coloring when multiple neurons are selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy.QtCore import Signal
from qtpy.QtGui import QColor

from .app_constants import Z_INDEX_TIME_MARKER, RASTER_DEBOUNCE_MS, DEFAULT_BUFFER_MULTIPLIER, BUFFER_COVERAGE_MARGIN
from .plots_base import BasePlot, ThrottleDebounce

if TYPE_CHECKING:
    from ethograph.gui.plots_timeseriessource import SpikeEventSource

_PHY_BG = '#000000'
_PHY_AXIS = '#AAAAAA'
_DOT_COLOR = QColor(180, 180, 180, 200)
_DOT_WIDTH = 3
_MAX_DOTS_PER_GROUP = 50_000


class RasterPlot(BasePlot):
    """Spike raster: dots at (spike_time, channel_y_position).

    Mirrors the ephys trace y-coordinate space so spikes align visually
    with their best channels.
    """

    y_range_changed = Signal()

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        self.setBackground(_PHY_BG)
        for axis_name in ('left', 'bottom'):
            axis = self.plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen(_PHY_AXIS))
            axis.setTextPen(pg.mkPen(_PHY_AXIS))
        self.time_marker.setPen(pg.mkPen('#FF4444', width=2, style=pg.QtCore.Qt.DotLine))

        self.plot_item.getAxis('left').hide()

        self._hw_to_global_y: dict[int, float] = {}
        self._channel_spacing: float = 1.0
        self._total_channels: int = 0

        # One ScatterPlotItem per color group; rebuilt on viewport change.
        self._scatter_items: list[pg.ScatterPlotItem] = []
        self._scatter_t0: float | None = None
        self._scatter_t1: float | None = None

        # Full sorted spike arrays (source of truth).
        self._spike_times: NDArray | None = None
        self._best_channels: NDArray | None = None
        self._multi_entries: list[tuple[NDArray, NDArray, tuple]] | None = None

        # Debounce viewport-driven rebuilds so rapid zoom/pan doesn't flood
        # the Qt event queue with expensive setData() calls.
        self._td = ThrottleDebounce(
            debounce_ms=RASTER_DEBOUNCE_MS,
            throttle_cb=self._update_visible_dots,
            debounce_cb=self._update_visible_dots,
        )

        self.vb.sigRangeChanged.connect(self._on_range_changed)
        self.vb.sigYRangeChanged.connect(self._emit_y_range)

    def _on_range_changed(self):
        if self._spike_times is not None or self._multi_entries is not None:
            self._td.trigger()

    def _emit_y_range(self):
        self.y_range_changed.emit()

    # ------------------------------------------------------------------
    # Y-axis sync
    # ------------------------------------------------------------------

    def sync_y_axis(
        self,
        hw_to_global_y: dict[int, float],
        spacing: float,
        total_channels: int,
    ):
        self._hw_to_global_y = hw_to_global_y
        self._channel_spacing = spacing
        self._total_channels = total_channels

        if total_channels > 0:
            margin = spacing * 0.5
            y_max = (total_channels - 1) * spacing + margin
            self.vb.setLimits(yMin=-margin, yMax=y_max)

        self._update_visible_dots()

    # ------------------------------------------------------------------
    # Spike data API
    # ------------------------------------------------------------------

    def set_spike_data(self, spike_times: NDArray, best_channels: NDArray):
        self._multi_entries = None
        if len(spike_times) == 0:
            self._spike_times = None
            self._best_channels = None
            self._clear_scatter_items()
            return

        order = np.argsort(spike_times)
        self._spike_times = spike_times[order]
        self._best_channels = best_channels[order]
        self._scatter_t0 = None
        self._update_visible_dots()

    def set_multi_cluster_spike_data(
        self,
        entries: list[tuple[NDArray, NDArray, tuple]],
    ):
        # Pre-sort each cluster by time so searchsorted works correctly.
        sorted_entries = []
        for times, channels, color in entries:
            if len(times) == 0:
                continue
            order = np.argsort(times)
            sorted_entries.append((times[order], channels[order], color))
        self._multi_entries = sorted_entries
        self._spike_times = None
        self._best_channels = None
        self._scatter_t0 = None
        self._update_visible_dots()

    def set_spike_source(self, source: SpikeEventSource):
        """Populate from a SpikeEventSource (from TrialAlignment)."""
        if source.channels is not None:
            self.set_spike_data(source._times, source.channels)
        else:
            self.set_spike_data(source._times, np.zeros(len(source._times), dtype=int))

    def clear_spike_data(self):
        self._spike_times = None
        self._best_channels = None
        self._multi_entries = None
        self._clear_scatter_items()

    # ------------------------------------------------------------------
    # BasePlot overrides
    # ------------------------------------------------------------------

    def update_plot_content(self, t0=None, t1=None):
        pass  # range changes handled via sigRangeChanged → ThrottleDebounce

    def apply_y_range(self, ymin=None, ymax=None):
        if ymin is not None and ymax is not None:
            self.vb.setYRange(ymin, ymax, padding=0)

    def _apply_y_constraints(self):
        if self._total_channels > 0:
            margin = self._channel_spacing * 0.5
            y_max = (self._total_channels - 1) * self._channel_spacing + margin
            self.vb.setLimits(yMin=-margin, yMax=y_max)

    def _get_time_bounds(self):
        if self._spike_times is not None and len(self._spike_times) > 0:
            return float(self._spike_times[0]), float(self._spike_times[-1])
        return super()._get_time_bounds()

    # ------------------------------------------------------------------
    # Internal – scatter management
    # ------------------------------------------------------------------

    def _clear_scatter_items(self):
        for item in self._scatter_items:
            try:
                self.vb.removeItem(item)
            except (RuntimeError, ValueError):
                pass
        self._scatter_items.clear()
        self._scatter_t0 = None
        self._scatter_t1 = None

    def _covers_range(self, x_lo: float, x_hi: float) -> bool:
        if self._scatter_t0 is None:
            return False
        margin = (x_hi - x_lo) * BUFFER_COVERAGE_MARGIN
        return self._scatter_t0 <= x_lo - margin and self._scatter_t1 >= x_hi + margin

    def _add_scatter(self, x: NDArray, y: NDArray, color):
        scatter = pg.ScatterPlotItem(
            x=x, y=y,
            pen=None,
            brush=pg.mkBrush(color),
            size=_DOT_WIDTH,
            symbol='o',
            useCache=True,
        )
        scatter.setZValue(Z_INDEX_TIME_MARKER - 1)
        self.vb.addItem(scatter, ignoreBounds=True)
        self._scatter_items.append(scatter)

    # ------------------------------------------------------------------
    # Viewport-culled update (called via ThrottleDebounce or directly)
    # ------------------------------------------------------------------

    def _update_visible_dots(self):
        if not self._hw_to_global_y:
            return

        (x_lo, x_hi), (y_lo, y_hi) = self.vb.viewRange()

        if self._covers_range(x_lo, x_hi):
            return

        self._clear_scatter_items()

        x_span = x_hi - x_lo
        x_buf = x_span * DEFAULT_BUFFER_MULTIPLIER / 2
        draw_x0 = x_lo - x_buf
        draw_x1 = x_hi + x_buf
        self._scatter_t0 = draw_x0
        self._scatter_t1 = draw_x1

        if self._multi_entries is not None:
            for times, channels, color in self._multi_entries:
                self._draw_visible(times, channels, draw_x0, draw_x1, y_lo, y_hi,
                                   QColor(*color))
        elif self._spike_times is not None and self._best_channels is not None:
            self._draw_visible(self._spike_times, self._best_channels,
                               draw_x0, draw_x1, y_lo, y_hi, _DOT_COLOR)

    def _draw_visible(
        self,
        times: NDArray, channels: NDArray,
        x_lo: float, x_hi: float,
        y_lo: float, y_hi: float,
        color,
    ):
        # 1. X-cull via searchsorted (O(log n), times is pre-sorted).
        i0 = int(np.searchsorted(times, x_lo, side='left'))
        i1 = int(np.searchsorted(times, x_hi, side='right'))
        if i1 <= i0:
            return
        t_vis = times[i0:i1]
        ch_vis = channels[i0:i1]

        # 2. Downsample BEFORE channel mapping so the map runs on ≤50K rows.
        if len(t_vis) > _MAX_DOTS_PER_GROUP:
            step = max(1, len(t_vis) // _MAX_DOTS_PER_GROUP)
            t_vis = t_vis[::step]
            ch_vis = ch_vis[::step]

        # 3. Map hardware channel indices → global Y positions.
        y_pos, valid = self._map_channels_to_y(ch_vis)
        t_vis = t_vis[valid]
        y_pos = y_pos[valid]
        if len(t_vis) == 0:
            return

        # 4. Y-cull.
        y_mask = (y_pos >= y_lo) & (y_pos <= y_hi)
        t_vis = t_vis[y_mask]
        y_pos = y_pos[y_mask]
        if len(t_vis) == 0:
            return

        self._add_scatter(t_vis, y_pos, color)

    def _map_channels_to_y(self, channels: NDArray) -> tuple[NDArray, NDArray]:
        hw_map = self._hw_to_global_y
        unique_chs = np.unique(channels)
        max_ch = int(unique_chs.max()) if len(unique_chs) > 0 else 0
        lookup = np.full(max_ch + 1, np.nan, dtype=np.float64)
        for ch in unique_chs:
            y = hw_map.get(int(ch))
            if y is not None:
                lookup[int(ch)] = y

        clipped = np.clip(channels.astype(int), 0, len(lookup) - 1)
        y_positions = lookup[clipped]
        valid = ~np.isnan(y_positions)
        return y_positions, valid
