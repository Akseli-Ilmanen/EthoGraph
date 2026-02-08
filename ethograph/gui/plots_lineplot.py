"""Enhanced line plot inheriting from BasePlot."""

from typing import Optional

import numpy as np
from qtpy.QtCore import QTimer

from ethograph.plots.lineplot_qtgraph import clear_plot_items, plot_ds_variable
from ethograph.utils.data_utils import sel_valid

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
        if feature_sel in ("Spectrogram", "Waveform"):
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
        if feature_sel in ("Spectrogram", "Waveform"):
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
