"""Enhanced line plot inheriting from BasePlot."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg

import ethograph as eto
from ethograph.io.catalog import PlotData

logger = logging.getLogger(__name__)

from ethograph.io.plot_sources import WindowedBuffer, XarraySource  # noqa: E402

from .app_constants import DEFAULT_BUFFER_MULTIPLIER, LINEPLOT_DEBOUNCE_MS  # noqa: E402
from .make_pretty import clean_display_labels  # noqa: E402
from .plots_base import BasePlot, ThrottleDebounce  # noqa: E402


class LinePlot(BasePlot):
    """Line plot with lazy loading and shared sync/marker functionality."""

    def __init__(self, napari_viewer, app_state, parent=None):
        super().__init__(app_state, parent)
        self.viewer = napari_viewer

        self.setLabel("left", "Value")

        self.plot_items = []
        self.label_items = []

        self._buffer = WindowedBuffer(buffer_multiplier=DEFAULT_BUFFER_MULTIPLIER)
        self._current_feature = None
        self._current_trial = None
        self._current_ds_kwargs_hash = None

        self._td = ThrottleDebounce(
            debounce_ms=LINEPLOT_DEBOUNCE_MS,
            throttle_cb=self._do_range_update,
            debounce_cb=self._do_range_update,
        )

        self.vb.sigRangeChanged.connect(self._on_view_range_changed)

    def _get_selections_hash(self) -> str:
        selections = self.app_state.get_selections()
        return str(sorted(selections.items()))

    def _context_changed(self) -> bool:
        feature = getattr(self.app_state, "features_sel", None)
        trial = getattr(self.app_state, "trials_sel", None)
        sel_hash = self._get_selections_hash()

        return (
            feature != self._current_feature or trial != self._current_trial or sel_hash != self._current_ds_kwargs_hash
        )

    def _update_context(self):
        self._current_feature = getattr(self.app_state, "features_sel", None)
        self._current_trial = getattr(self.app_state, "trials_sel", None)
        self._current_ds_kwargs_hash = self._get_selections_hash()

    def _ensure_source(self):
        """Create/update the XarraySource from current app_state."""
        ds = self.app_state.ds
        time_coord = self.app_state.time_coord
        if ds is None or time_coord is None:
            self._buffer.set_source(None)
            return
        bounds = self.app_state.window_bounds
        source = XarraySource(ds, time_coord.name)
        self._buffer.set_source(source, bounds=bounds)

    def _get_buffered_ds(self, t0: float, t1: float):
        """Get buffered dataset slice for the visible time range."""
        if self._context_changed():
            self._buffer.invalidate()
            self._update_context()
            self._ensure_source()

        if self._buffer.source is None:
            self._ensure_source()

        return self._buffer.get(t0, t1)

    @property
    def _store(self):
        return getattr(self.app_state, "data_loader", None)

    def _on_view_range_changed(self):
        if not self.isVisible():
            return
        if self._store is None and (not hasattr(self.app_state, "ds") or self.app_state.ds is None):
            return
        self._td.trigger()

    def _do_range_update(self):
        if not self.isVisible():
            return
        t0, t1 = self.get_current_xlim()
        self._update_plot(t0, t1)

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        clear_plot_items(self.plot_item, self.plot_items)

        if not hasattr(self.app_state, "features_sel"):
            return

        if t0 is None or t1 is None:
            t0, t1 = self.get_current_xlim()

        self._update_plot(t0, t1)

    def _update_plot(self, t0: float, t1: float):
        clear_plot_items(self.plot_item, self.plot_items)

        if not hasattr(self.app_state, "features_sel"):
            return

        feature_sel = self.app_state.features_sel
        selections = self.app_state.get_selections()
        color_var = None
        if hasattr(self.app_state, "colors_sel") and self.app_state.colors_sel != "None":
            color_var = self.app_state.colors_sel
        show_cp = getattr(self.app_state, "show_changepoints", False)

        store = self._store
        if store is not None:
            plot_data = store.select(
                feature_sel,
                selections,
                t0=t0,
                t1=t1,
                color_variable=color_var,
            )
        else:
            buffered_ds = self._get_buffered_ds(t0, t1)
            if buffered_ds is None:
                self.app_state.plot_has_changepoints = False
                return
            plot_data = select_feature(buffered_ds, feature_sel, selections, color_var)

        if plot_data is None:
            self.app_state.plot_has_changepoints = False
            return

        self.app_state.plot_has_changepoints = bool(plot_data.changepoints)
        self.plot_items = render_plot_data(self.plot_item, plot_data, show_changepoints=show_cp)

        for item in self.plot_items:
            if hasattr(item, "setDownsampling"):
                item.setDownsampling(auto=True, method="peak")

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _apply_y_constraints(self):
        """Apply y-axis constraints based on current feature data."""
        if not hasattr(self.app_state, "features_sel"):
            return

        feature_sel = self.app_state.features_sel
        selections = self.app_state.get_selections()

        try:
            store = self._store
            if store is not None:
                wb = self.app_state.window_bounds
                if wb is None:
                    return
                pd = store.select(feature_sel, selections, t0=wb.start_s, t1=wb.end_s)
                if pd is None:
                    return
                data = pd.data
            else:
                data, _ = eto.sel_valid(self.app_state.ds[feature_sel], selections)

            percentile_ylim = self.app_state.get_with_default("percentile_ylim")
            y_min = np.nanpercentile(data, 100 - percentile_ylim)
            y_max = np.nanpercentile(data, percentile_ylim)
            y_range = y_max - y_min
            y_buffer = y_range * 0.2

            if y_range > 0:
                self.vb.setLimits(
                    yMin=y_min - y_buffer,
                    yMax=y_max + y_buffer,
                    minYRange=y_range * 0.1,
                    maxYRange=y_range + y_buffer,
                )
        except (KeyError, AttributeError, ValueError):
            pass


class MultiColoredLineItem(pg.GraphicsObject):
    """Efficient multi-colored line for PyQtGraph."""

    def __init__(self, x, y, colors, width=2):
        super().__init__()
        self.x = x
        self.y = y
        self.colors = colors
        self.width = width
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self.picture)
        painter.setCompositionMode(pg.QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)

        for i in range(len(self.x) - 1):
            if i < len(self.colors):
                color = self.colors[i]
                if max(color) <= 1:
                    color = tuple(int(c * 255) for c in color)
            else:
                color = (255, 255, 255)

            pen = pg.mkPen(color=color, width=self.width)
            painter.setPen(pen)
            painter.drawLine(
                pg.QtCore.QPointF(self.x[i], self.y[i]),
                pg.QtCore.QPointF(self.x[i + 1], self.y[i + 1]),
            )

        painter.end()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


def plot_multidim(plot_item, time, data, coord_labels=None, existing_curves=None):
    """
    Plot multi-dimensional data (e.g., pos, vel) over time using PyQtGraph.

    Args:
        plot_item: PyQtGraph PlotItem to plot on
        time: time array
        data: shape (time, space)
        coord_labels: list of labels for each dimension (e.g., ['x', 'y', 'z'])
        existing_curves: list to append created curves to

    Returns:
        list of PlotDataItem objects
    """
    if existing_curves is None:
        existing_curves = []

    colors = [
        "#1f77b4",  # Blue (replaces white)
        "#d62728",  # Red
        "#2ca02c",  # Green
        "#ff7f0e",  # Orange
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ]

    for i in range(data.shape[1]):
        label = coord_labels[i] if coord_labels is not None else f"dim {i}"
        color = colors[i % len(colors)]

        curve = plot_item.plot(time, data[:, i], pen=pg.mkPen(color=color, width=2), name=label)
        existing_curves.append(curve)

    return existing_curves


def plot_singledim(
    plot_item,
    time,
    data,
    color_data=None,
    changepoints_dict=None,
    existing_items=None,
    show_changepoints=True,
):
    if existing_items is None:
        existing_items = []

    if color_data is not None and color_data.ndim == 2 and color_data.shape[1] == 3:
        multi_line = MultiColoredLineItem(time, data, color_data)
        plot_item.addItem(multi_line)
        existing_items.append(multi_line)
    else:
        curve = plot_item.plot(
            time,
            data,
            pen=pg.mkPen(color="k", width=2),
        )
        existing_items.append(curve)

    # Add changepoints as scatter plots, each with its own color and label
    if changepoints_dict is not None and show_changepoints:
        cmap = plt.get_cmap("tab10")

        colors = [tuple(int(c * 255) for c in cmap.colors[i][:3]) for i in range(len(cmap.colors))]

        for i, (cp_name, cp_array) in enumerate(changepoints_dict.items()):
            idxs = np.where(cp_array)[0]
            color = colors[(i + 5) % len(colors)]  # offset to match original
            if len(idxs) > 0:
                scatter = pg.ScatterPlotItem(
                    x=time[idxs],
                    y=data[idxs],
                    pen=pg.mkPen(color=color, width=2),
                    brush=None,
                    symbol="o",
                    size=10,
                    name=cp_name,
                )
                plot_item.addItem(scatter)
                existing_items.append(scatter)

    return existing_items


def select_feature(ds, variable, ds_kwargs, color_variable=None) -> PlotData | None:
    """Extract plot-ready data from an xarray Dataset via XarrayLoader."""
    from ethograph.io.catalog import XarrayLoader

    return XarrayLoader(ds).select(variable, ds_kwargs, color_variable=color_variable)


def render_plot_data(plot_item, plot_data: PlotData, show_changepoints=True) -> list:
    """Render a PlotData to a pyqtgraph PlotItem. Source-agnostic."""
    if hasattr(plot_item, "legend") and plot_item.legend is not None:
        plot_item.removeItem(plot_item.legend)
        plot_item.legend = None

    items = []
    dim_labels = plot_data.dim_labels
    if dim_labels:
        dim_labels = clean_display_labels(dim_labels)

    if plot_data.data.ndim == 2:
        plot_item.legend = plot_item.addLegend(offset=(10, 10))
        items = plot_multidim(
            plot_item,
            plot_data.time,
            plot_data.data,
            dim_labels,
            items,
        )
    elif plot_data.data.ndim == 1:
        if plot_data.changepoints and show_changepoints:
            plot_item.legend = plot_item.addLegend(offset=(10, 10))

        items = plot_singledim(
            plot_item,
            plot_data.time,
            plot_data.data,
            color_data=plot_data.color_data,
            changepoints_dict=plot_data.changepoints,
            existing_items=items,
            show_changepoints=show_changepoints,
        )
    else:
        logger.warning("Data ndim=%d not supported for plotting", plot_data.data.ndim)

    plot_item.setLabel("bottom", "Time", units="s")
    plot_item.setLabel("left", plot_data.ylabel, Fontsize="14pt")
    plot_item.setTitle(plot_data.title)

    return items


def plot_ds_variable(plot_item, ds, ds_kwargs, variable, color_variable=None, show_changepoints=True):
    """Plot a variable from an xarray Dataset.

    Delegates to :func:`select_feature` + :func:`render_plot_data`.
    """
    plot_data = select_feature(ds, variable, ds_kwargs, color_variable)
    if plot_data is None:
        return []
    return render_plot_data(plot_item, plot_data, show_changepoints=show_changepoints)


def clear_plot_items(plot_item, items_list):
    """Helper function to clear specific plot items from a plot with proper cleanup."""
    for item in items_list:
        plot_item.removeItem(item)

        if isinstance(item, MultiColoredLineItem):
            if hasattr(item, "picture"):
                item.picture = None
            item.x = None
            item.y = None
            item.colors = None

        elif isinstance(item, pg.ScatterPlotItem):
            item.clear()
            item.setData([], [])

        elif isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
            item.clear()

        elif isinstance(item, pg.InfiniteLine):
            if hasattr(item, "sigPositionChanged"):
                try:
                    item.sigPositionChanged.disconnect()
                except (TypeError, RuntimeError):
                    pass

        item.setParentItem(None)
        if hasattr(item, "deleteLater"):
            item.deleteLater()

    items_list.clear()
