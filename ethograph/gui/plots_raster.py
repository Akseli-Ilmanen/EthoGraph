"""Spike raster plot — one dot per spike at (time, best_channel_y).

Uses a custom QGraphicsObject with QPainter.drawPoints() for maximum
throughput.  Supports single-color (all spikes gray) and per-cluster
coloring when multiple neurons are selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy.QtCore import QPointF, QRectF, Signal
from qtpy.QtGui import QColor, QPen, QPolygonF

from .app_constants import Z_INDEX_TIME_MARKER
from .plots_base import BasePlot

if TYPE_CHECKING:
    from ethograph.gui.plots_timeseriessource import SpikeEventSource

_PHY_BG = '#000000'
_PHY_AXIS = '#AAAAAA'
_DOT_COLOR = QColor(180, 180, 180, 200)
_DOT_WIDTH = 3
_MAX_DOTS_PER_GROUP = 50_000


def _make_polygon(sx: NDArray, sy: NDArray, n: int) -> QPolygonF:
    try:
        from pyqtgraph.functions import ndarray_to_qpolygonf
        xy = np.empty((n, 2), dtype=np.float64)
        xy[:, 0] = sx
        xy[:, 1] = sy
        return ndarray_to_qpolygonf(xy)
    except (ImportError, AttributeError):
        return QPolygonF([QPointF(float(sx[i]), float(sy[i])) for i in range(n)])


class _RasterDots(pg.GraphicsObject):
    """Draws spike dots grouped by color.

    Each group is a (x_sorted, y, pen) tuple.  ``paint()`` viewport-culls
    each group independently and issues one ``drawPoints`` call per color.
    """

    def __init__(self, viewbox=None):
        super().__init__()
        self._viewbox = viewbox
        # List of (x_sorted, y, QPen) — one per color group
        self._groups: list[tuple[NDArray, NDArray, QPen]] = []
        self._bounding = QRectF()

    def set_data(self, x: NDArray, y: NDArray):
        pen = QPen(_DOT_COLOR, _DOT_WIDTH)
        pen.setCosmetic(True)
        self._set_groups([(x, y, pen)])

    def set_multi_data(
        self,
        entries: list[tuple[NDArray, NDArray, tuple]],
    ):
        groups = []
        for times, channels, color in entries:
            if len(times) == 0:
                continue
            pen = QPen(QColor(*color), _DOT_WIDTH)
            pen.setCosmetic(True)
            order = np.argsort(times)
            groups.append((times[order], channels[order], pen))
        self._set_groups(groups)

    def _set_groups(self, groups: list[tuple[NDArray, NDArray, QPen]]):
        self.prepareGeometryChange()
        self._groups = groups
        if groups:
            all_x = np.concatenate([g[0] for g in groups])
            all_y = np.concatenate([g[1] for g in groups])
            if len(all_x) > 0:
                self._bounding = QRectF(
                    float(all_x.min()), float(all_y.min()),
                    float(np.ptp(all_x)), float(np.ptp(all_y)),
                )
            else:
                self._bounding = QRectF()
        else:
            self._bounding = QRectF()
        self.update()

    def clear(self):
        self.prepareGeometryChange()
        self._groups = []
        self._bounding = QRectF()
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounding

    def paint(self, painter, _option, _widget):
        if not self._groups:
            return

        vb = self._viewbox
        if vb is None:
            return
        vr = vb.viewRange()
        x_lo, x_hi = vr[0]
        y_lo, y_hi = vr[1]

        tr = painter.transform()
        m11, m22 = tr.m11(), tr.m22()
        dx, dy = tr.m31(), tr.m32()
        painter.resetTransform()

        for x, y, pen in self._groups:
            i0 = int(np.searchsorted(x, x_lo, side='left'))
            i1 = int(np.searchsorted(x, x_hi, side='right'))
            xv = x[i0:i1]
            yv = y[i0:i1]

            if len(xv) == 0:
                continue

            y_mask = (yv >= y_lo) & (yv <= y_hi)
            xv = xv[y_mask]
            yv = yv[y_mask]

            n = len(xv)
            if n == 0:
                continue

            if n > _MAX_DOTS_PER_GROUP:
                step = n // _MAX_DOTS_PER_GROUP
                xv = xv[::step]
                yv = yv[::step]
                n = len(xv)

            sx = np.ascontiguousarray(xv * m11 + dx, dtype=np.float64)
            sy = np.ascontiguousarray(yv * m22 + dy, dtype=np.float64)
            poly = _make_polygon(sx, sy, n)

            painter.setPen(pen)
            painter.drawPoints(poly)


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

        self._dots = _RasterDots(viewbox=self.vb)
        self._dots.setZValue(Z_INDEX_TIME_MARKER - 1)
        self.vb.addItem(self._dots, ignoreBounds=True)

        self._spike_times: NDArray | None = None
        self._best_channels: NDArray | None = None
        self._multi_entries: list[tuple[NDArray, NDArray, tuple]] | None = None

        self.vb.sigYRangeChanged.connect(self._emit_y_range)

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

        self._rebuild_dots()

    # ------------------------------------------------------------------
    # Spike data API
    # ------------------------------------------------------------------

    def set_spike_data(self, spike_times: NDArray, best_channels: NDArray):
        self._multi_entries = None
        if len(spike_times) == 0:
            self._spike_times = None
            self._best_channels = None
            self._dots.clear()
            return

        order = np.argsort(spike_times)
        self._spike_times = spike_times[order]
        self._best_channels = best_channels[order]
        self._rebuild_dots()

    def set_multi_cluster_spike_data(
        self,
        entries: list[tuple[NDArray, NDArray, tuple]],
    ):
        self._multi_entries = entries
        self._spike_times = None
        self._best_channels = None
        self._rebuild_dots()

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
        self._dots.clear()

    # ------------------------------------------------------------------
    # BasePlot overrides
    # ------------------------------------------------------------------

    def update_plot_content(self, t0=None, t1=None):
        pass  # dots self-cull in paint()

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
    # Internal
    # ------------------------------------------------------------------

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

    def _rebuild_dots(self):
        if not self._hw_to_global_y:
            self._dots.clear()
            return

        if self._multi_entries is not None:
            self._rebuild_multi_dots()
            return

        if self._spike_times is None or self._best_channels is None:
            self._dots.clear()
            return

        y_positions, valid = self._map_channels_to_y(self._best_channels)
        self._dots.set_data(self._spike_times[valid], y_positions[valid])

    def _rebuild_multi_dots(self):
        groups = []
        for times, channels, color in self._multi_entries:
            if len(times) == 0:
                continue
            y_positions, valid = self._map_channels_to_y(channels)
            if np.any(valid):
                groups.append((times[valid], y_positions[valid], color))
        if groups:
            self._dots.set_multi_data(groups)
        else:
            self._dots.clear()
