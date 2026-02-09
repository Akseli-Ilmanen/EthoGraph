"""Heatmap plot for visualizing feature sub-dimensions as color-coded rows."""

from typing import Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QTimer

from ethograph.features.preprocessing import z_normalize
from ethograph.utils.data_utils import sel_valid

from .app_constants import (
    DEFAULT_BUFFER_MULTIPLIER,
    Z_INDEX_BACKGROUND,
    HEATMAP_EXCLUSION_PERCENTILE,
)
from .plots_base import BasePlot


class HeatmapPlot(BasePlot):
    """MNE-style stacked heatmap rendering feature data as color-coded rows.

    Each row corresponds to a sub-dimension (e.g., 10 keypoints)
    1D data is rendered as a single row. Uses RdBu_r diverging colormap
    with per-row z-score normalization.
    """

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        self.setLabel('left', 'Channel')

        self.image_item = pg.ImageItem(autoDownsample=True)
        self.image_item.setZValue(Z_INDEX_BACKGROUND)
        self.addItem(self.image_item)

        self._init_colormap()

        self.label_items = []
        self._n_channels = 1
        self._channel_labels = []

        # Buffer state for lazy loading
        self._buffer_multiplier = DEFAULT_BUFFER_MULTIPLIER
        self._buffered_data = None
        self._buffered_time = None
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

    def _init_colormap(self):
        cmap = pg.colormap.get('RdBu_r', source='matplotlib')
        self.image_item.setColorMap(cmap)

    # --- Context tracking (same pattern as LinePlot) ---

    def _get_ds_kwargs_hash(self) -> str:
        ds_kwargs = self.app_state.get_ds_kwargs()
        return str(sorted(ds_kwargs.items()))

    def _context_changed(self) -> bool:
        feature = getattr(self.app_state, 'features_sel', None)
        trial = getattr(self.app_state, 'trials_sel', None)
        ds_kwargs_hash = self._get_ds_kwargs_hash()
        return (
            feature != self._current_feature
            or trial != self._current_trial
            or ds_kwargs_hash != self._current_ds_kwargs_hash
        )

    def _update_context(self):
        self._current_feature = getattr(self.app_state, 'features_sel', None)
        self._current_trial = getattr(self.app_state, 'trials_sel', None)
        self._current_ds_kwargs_hash = self._get_ds_kwargs_hash()

    def _clear_buffer(self):
        self._buffered_data = None
        self._buffered_time = None
        self._buffer_t0 = 0.0
        self._buffer_t1 = 0.0

    # --- Buffered data loading ---

    def _get_buffered_data(self, t0: float, t1: float):
        """Load and cache feature data for the visible time range with buffer."""
        if self._context_changed():
            self._clear_buffer()
            self._update_context()

        margin = (t1 - t0) * 0.2
        if (
            self._buffered_data is not None
            and self._buffer_t0 <= t0 - margin
            and self._buffer_t1 >= t1 + margin
        ):
            return self._buffered_data, self._buffered_time

        ds = self.app_state.ds
        time = self.app_state.time.values
        feature_sel = self.app_state.features_sel

        if feature_sel == "Audio Waveform":
            return None, None

        window_size = t1 - t0
        buffer_size = window_size * self._buffer_multiplier
        load_t0 = max(float(time[0]), t0 - buffer_size / 2)
        load_t1 = min(float(time[-1]), t1 + buffer_size / 2)

        time_coord_name = self.app_state.time.name
        buffered_ds = ds.sel({time_coord_name: slice(load_t0, load_t1)})

        ds_kwargs = self.app_state.get_ds_kwargs()
        da = buffered_ds[feature_sel]
        data, _ = sel_valid(da, ds_kwargs)

        if data.ndim == 1:
            data = data[:, np.newaxis]

        # Extract channel labels from coordinates
        da_full = ds[feature_sel]
        dims_after_sel = [d for d in da_full.dims if 'time' not in d and d not in ds_kwargs]
        if dims_after_sel and dims_after_sel[0] in da_full.coords:
            self._channel_labels = [str(v) for v in da_full.coords[dims_after_sel[0]].values]
        elif data.shape[1] > 1:
            self._channel_labels = [str(i) for i in range(data.shape[1])]
        else:
            self._channel_labels = [feature_sel]

        self._n_channels = data.shape[1]

        buffered_time = buffered_ds.coords[time_coord_name].values

        self._buffered_data = data
        self._buffered_time = buffered_time
        self._buffer_t0 = load_t0
        self._buffer_t1 = load_t1

        return data, buffered_time

    # --- Rendering ---

    def _compute_symmetric_levels(self, data: np.ndarray) -> tuple[float, float]:
        """Compute symmetric color range from 98th percent. of abs. values, clipping extremes."""
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return -1.0, 1.0
        vmax = np.percentile(np.abs(valid), HEATMAP_EXCLUSION_PERCENTILE)
        if vmax < 1e-10:
            vmax = 1.0
        return -vmax, vmax

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        if not hasattr(self.app_state, 'features_sel'):
            return

        feature_sel = self.app_state.features_sel
        if feature_sel == "Audio Waveform":
            return

        if t0 is None or t1 is None:
            t0, t1 = self.get_current_xlim()

        self._render_heatmap(t0, t1)

    def _render_heatmap(self, t0: float, t1: float):
        result = self._get_buffered_data(t0, t1)
        if result[0] is None:
            return

        data, time_vals = result
        normalized = z_normalize(data)
        vmin, vmax = self._compute_symmetric_levels(normalized)

        # pyqtgraph ImageItem uses img[x, y]: first index = horizontal (time),
        # second index = vertical (channels). normalized is (n_time, n_channels)
        # which already matches this convention — no transpose needed.
        self.image_item.setImage(normalized, autoLevels=False)
        self.image_item.setLevels([vmin, vmax])

        buf_t0 = float(time_vals[0])
        buf_t1 = float(time_vals[-1])
        duration = buf_t1 - buf_t0
        n_channels = self._n_channels

        self.image_item.setRect(pg.QtCore.QRectF(buf_t0, 0, duration, n_channels))
        self.plot_item.setYRange(0, n_channels, padding=0)

        self._update_y_axis_ticks()

    def _update_y_axis_ticks(self):
        """Set y-axis tick labels to channel names."""
        left_axis = self.plot_item.getAxis('left')
        ticks = [(i + 0.5, label) for i, label in enumerate(self._channel_labels)]
        left_axis.setTicks([ticks])

    # --- View range handling ---

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
        self._render_heatmap(t0, t1)

    # --- Y-axis management ---

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _apply_y_constraints(self):
        self.vb.setLimits(yMin=-0.5, yMax=self._n_channels + 0.5)
