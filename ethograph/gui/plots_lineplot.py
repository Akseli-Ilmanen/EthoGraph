"""Enhanced line plot inheriting from BasePlot."""

from typing import Optional

import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from qtpy.QtCore import QTimer

from ethograph.utils.data_utils import sel_valid, get_time_coord

from .plots_base import BasePlot


class LinePlot(BasePlot):
    """Line plot with lazy loading and shared sync/marker functionality."""

    def __init__(self, napari_viewer, app_state, parent=None):
        super().__init__(app_state, parent)
        self.viewer = napari_viewer

        self.setLabel('left', 'Value')

        self.plot_items = []
        self.label_items = []

        # Buffer state for lazy loading
        self._buffer_multiplier = 5.0
        self._buffered_ds = None
        self._buffer_t0 = 0.0
        self._buffer_t1 = 0.0
        self._current_feature = None
        self._current_trial = None
        self._current_ds_kwargs_hash = None

        # Debounce timer for view range changes
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(50)
        self._debounce_timer.timeout.connect(self._debounced_update)
        self._pending_range = None

        self.vb.sigRangeChanged.connect(self._on_view_range_changed)

    def _get_ds_kwargs_hash(self) -> str:
        ds_kwargs = self.app_state.get_ds_kwargs()
        return str(sorted(ds_kwargs.items()))

    def _context_changed(self) -> bool:
        feature = getattr(self.app_state, 'features_sel', None)
        trial = getattr(self.app_state, 'trials_sel', None)
        ds_kwargs_hash = self._get_ds_kwargs_hash()

        return (feature != self._current_feature or
                trial != self._current_trial or
                ds_kwargs_hash != self._current_ds_kwargs_hash)

    def _update_context(self):
        self._current_feature = getattr(self.app_state, 'features_sel', None)
        self._current_trial = getattr(self.app_state, 'trials_sel', None)
        self._current_ds_kwargs_hash = self._get_ds_kwargs_hash()

    def _clear_buffer(self):
        self._buffered_ds = None
        self._buffer_t0 = 0.0
        self._buffer_t1 = 0.0

    def _get_buffered_ds(self, t0: float, t1: float):
        """Get buffered dataset slice for the visible time range."""
        if self._context_changed():
            self._clear_buffer()
            self._update_context()

        margin = (t1 - t0) * 0.2
        if (self._buffered_ds is not None and
            self._buffer_t0 <= t0 - margin and
            self._buffer_t1 >= t1 + margin):
            return self._buffered_ds

        ds = self.app_state.ds
        time = self.app_state.time.values

        window_size = t1 - t0
        buffer_size = window_size * self._buffer_multiplier
        load_t0 = max(time[0], t0 - buffer_size / 2)
        load_t1 = min(time[-1], t1 + buffer_size / 2)

        time_coord_name = self.app_state.time.name
        self._buffered_ds = ds.sel({time_coord_name: slice(load_t0, load_t1)})
        

        
        self._buffer_t0 = load_t0
        self._buffer_t1 = load_t1

        return self._buffered_ds

    def _on_view_range_changed(self):
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return

        self._pending_range = self.get_current_xlim()
        self._debounce_timer.start()

    def _debounced_update(self):
        if self._pending_range is None:
            return

        t0, t1 = self._pending_range
        self._pending_range = None
        self._update_plot(t0, t1)

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        clear_plot_items(self.plot_item, self.plot_items)

        if not hasattr(self.app_state, 'features_sel'):
            return

        if t0 is None or t1 is None:
            t0, t1 = self.get_current_xlim()

        self._update_plot(t0, t1)

    def _update_plot(self, t0: float, t1: float):
        clear_plot_items(self.plot_item, self.plot_items)

        ds_kwargs = self.app_state.get_ds_kwargs()
        feature_sel = self.app_state.features_sel

        # Skip audio-specific features - they use dedicated plots
        if feature_sel == "Audio Waveform":
            return

        color_var = None
        if hasattr(self.app_state, 'colors_sel') and self.app_state.colors_sel != "None":
            color_var = self.app_state.colors_sel

        buffered_ds = self._get_buffered_ds(t0, t1)

        show_cp = getattr(self.app_state, 'show_changepoints', False)
        self.plot_items = plot_ds_variable(
            self.plot_item,
            buffered_ds,
            ds_kwargs,
            feature_sel,
            color_variable=color_var,
            show_changepoints=show_cp
        )

        for item in self.plot_items:
            if hasattr(item, 'setDownsampling'):
                item.setDownsampling(auto=True, method='peak')

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        """Apply y-axis range for line plot."""
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _apply_y_constraints(self):
        """Apply y-axis constraints based on current feature data."""
        if not hasattr(self.app_state, 'features_sel'):
            return

        feature_sel = self.app_state.features_sel
        if feature_sel == "Audio Waveform":
            return

        ds_kwargs = self.app_state.get_ds_kwargs()

        try:
            data, _ = sel_valid(self.app_state.ds[feature_sel], ds_kwargs)

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
                    maxYRange=y_range + y_buffer
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
                pg.QtCore.QPointF(self.x[i+1], self.y[i+1])
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
        '#1f77b4',  # Blue (replaces white)
        '#d62728',  # Red 
        '#2ca02c',  # Green
        '#ff7f0e',  # Orange
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    for i in range(data.shape[1]):
        label = coord_labels[i] if coord_labels is not None else f"dim {i}"
        color = colors[i % len(colors)]
        
        curve = plot_item.plot(
            time, data[:, i], 
            pen=pg.mkPen(color=color, width=2),
            name=label
        )
        existing_curves.append(curve)
    
    return existing_curves


def plot_singledim(plot_item, time, data, color_data=None, changepoints_dict=None, existing_items=None, show_changepoints=True):
    if existing_items is None:
        existing_items = []

    if color_data is not None and color_data.ndim == 2 and color_data.shape[1] == 3:
        multi_line = MultiColoredLineItem(time, data, color_data)
        plot_item.addItem(multi_line)
        existing_items.append(multi_line)
    else:
        curve = plot_item.plot(
            time, data,
            pen=pg.mkPen(color='k', width=2),
        )
        existing_items.append(curve)

    # Add changepoints as scatter plots, each with its own color and label
    if changepoints_dict is not None and show_changepoints:
        # Use tab10 color palette from matplotlib, converted to 0-255 RGB
        
        
        
        cmap = plt.get_cmap('tab10')
        
        colors = [tuple(int(c*255) for c in cmap.colors[i][:3]) for i in range(len(cmap.colors))]
        
        
        for i, (cp_name, cp_array) in enumerate(changepoints_dict.items()):
            idxs = np.where(cp_array)[0]
            color = colors[(i+5) % len(colors)]  # offset to match original
            if len(idxs) > 0:
                scatter = pg.ScatterPlotItem(
                    x=time[idxs],
                    y=data[idxs],
                    pen=pg.mkPen(color=color, width=2),
                    brush=None,
                    symbol='o',
                    size=10,
                    name=cp_name
                )
                plot_item.addItem(scatter)
                existing_items.append(scatter)

    return existing_items


def plot_ds_variable(plot_item, ds, ds_kwargs, variable, color_variable=None, show_changepoints=True):
    """
    Plot a variable from ds for a given trial and keypoint using PyQtGraph.
    Handles both multi-dimensional (e.g., pos, vel) and single-dimensional (e.g., speed) variables.

    Args:
        plot_item: PyQtGraph PlotItem to plot on
        ds: xarray Dataset
        ds_kwargs: dict with selection criteria (e.g., {keypoints="beakTip"})
        variable: variable name to plot
        color_variable: optional variable name for coloring
        show_changepoints: whether to draw changepoint markers

    Returns:
        list of created plot items
    """

    # Clear existing legend if present
    if hasattr(plot_item, 'legend') and plot_item.legend is not None:
        plot_item.removeItem(plot_item.legend)
        plot_item.legend = None

    # NOTE: Don't clear all items here - clear_plot_items() handles targeted
    # clearing before this function is called. Blanket clearing removes labels.

    var = ds[variable]
    time = get_time_coord(var).values

    data, filt_kwargs = sel_valid(var, ds_kwargs)
    var = var.sel(**filt_kwargs)
    plot_items = []

    if data.ndim == 2:
        # data is (time, other_dim) after sel_valid transpose
        # Find the non-time dimension for coordinate labels
        non_time_dim = next((d for d in var.dims if 'time' not in d.lower()), None)
        if non_time_dim and non_time_dim in var.coords:
            coord_labels = var.coords[non_time_dim].values
        else:
            coord_labels = [str(i) for i in range(data.shape[1])]
        plot_item.legend = plot_item.addLegend(offset=(10, 10))
        plot_items = plot_multidim(plot_item, time, data, coord_labels, plot_items)

    elif data.ndim == 1:
        if color_variable and color_variable in ds.data_vars:
            # Exclude RGB from kwargs to keep all 3 color channels
            color_kwargs = {k: v for k, v in ds_kwargs.items() if k != 'RGB'}
            color_data, _ = sel_valid(ds[color_variable], color_kwargs)
        else:
            color_data = None


        # Build changepoints_dict from ds attributes, inspired by plots.py
        changepoints_dict = {}
        if hasattr(ds, 'filter_by_attrs'):
            cp_ds = ds.filter_by_attrs(type="changepoints")
            for cp_var_name in cp_ds.data_vars:
                cp_var = cp_ds[cp_var_name]
                cp_data, _ = sel_valid(cp_var, ds_kwargs)
                if cp_var.attrs.get("target_feature") == variable and not np.isnan(cp_data).all():
                    changepoints_dict[cp_var_name] = cp_data

        # Add legend if changepoints will be shown
        if changepoints_dict and show_changepoints:
            plot_item.legend = plot_item.addLegend(offset=(10, 10))

        plot_items = plot_singledim(
            plot_item, time, data,
            color_data=color_data,
            changepoints_dict=changepoints_dict if changepoints_dict else None,
            existing_items=plot_items,
            show_changepoints=show_changepoints
        )
    else:
        print(f"Variable '{variable}' not supported for plotting.")
    
    # Add boundary events as vertical lines
    if hasattr(ds, "boundary_events"):
        boundary_events_raw = ds["boundary_events"].values
        valid_events = boundary_events_raw[~np.isnan(boundary_events_raw)]
        eventsIdxs = valid_events.astype(int)
        eventsIdxs = eventsIdxs[(eventsIdxs >= 0) & (eventsIdxs < len(time))]
        
        for event in eventsIdxs:
            vline = pg.InfiniteLine(
                pos=time[event], 
                angle=90, 
                pen=pg.mkPen('k', width=2)
            )
            plot_item.addItem(vline)
            plot_items.append(vline)
    
    # Set labels and title - use filt_kwargs which contains the applied selections
    ylabel = var.attrs.get("ylabel", variable)
    title_parts = [f"Trial: {ds.attrs.get('trial')}"]
    title_parts.extend(f"{k}={v}" for k, v in filt_kwargs.items())
    title = ", ".join(title_parts)
    
    plot_item.setLabel('bottom', 'Time', units='s')
    plot_item.setLabel('left', ylabel, Fontsize='12pt')
    plot_item.setTitle(title)

    return plot_items


def clear_plot_items(plot_item, items_list):
    """Helper function to clear specific plot items from a plot with proper cleanup."""
    for item in items_list:
        try:
            plot_item.removeItem(item)


            if isinstance(item, MultiColoredLineItem):
                if hasattr(item, 'picture'):
                    item.picture = None
                item.x = None
                item.y = None
                item.colors = None

      
            elif isinstance(item, pg.ScatterPlotItem):
                item.clear()
                item.setData([], [])


            elif isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
                item.clear()
                if hasattr(item, 'xData'):
                    item.xData = None
                if hasattr(item, 'yData'):
                    item.yData = None

       
            elif isinstance(item, pg.InfiniteLine):
                if hasattr(item, 'sigPositionChanged'):
                    try:
                        item.sigPositionChanged.disconnect()
                    except:
                        pass

 
            item.setParentItem(None)
            item.deleteLater() if hasattr(item, 'deleteLater') else None

        except Exception as e:
            pass  

    items_list.clear()


