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
        self.vb.invertY(True)

        self._init_colormap()
        self._init_colorbar()

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
        self._debounce_timer.setInterval(150)
        self._debounce_timer.timeout.connect(self._debounced_update)
        self._pending_range = None
        self._rendering = False

        self.vb.sigXRangeChanged.connect(self._on_view_range_changed)

    def _init_colormap(self):
        colormap_name = self.app_state.get_with_default('heatmap_colormap')
        try:
            self._cmap = pg.colormap.get(colormap_name, source='matplotlib')
        except (KeyError, ValueError, TypeError):
            self._cmap = pg.colormap.get('RdBu_r', source='matplotlib')
        self.image_item.setColorMap(self._cmap)

    def _init_colorbar(self):
        self.colorbar = pg.ColorBarItem(
            values=(-1, 1),
            colorMap=self._cmap,
            interactive=False,
            width=15,
        )
        self.colorbar.setImageItem(self.image_item, insert_in=self.plot_item)

    def update_colormap(self, name: str):
        try:
            cmap = pg.colormap.get(name, source='matplotlib')
            self._cmap = cmap
            self.image_item.setColorMap(cmap)
            self.colorbar.setColorMap(cmap)
        except (KeyError, ValueError, TypeError):
            pass

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

    # --- Audio envelope loading ---

    def _get_buffered_audio_envelope(self, t0: float, t1: float):
        """Load audio, compute per-channel envelope using selected metric, and cache."""
        from .plots_spectrogram import SharedAudioCache

        audio_path = getattr(self.app_state, 'audio_path', None)
        if not audio_path:
            return None, None

        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            return None, None

        fs = loader.rate
        total_duration = len(loader) / fs

        window_size = t1 - t0
        buffer_size = window_size * self._buffer_multiplier
        load_t0 = max(0.0, t0 - buffer_size / 2)
        load_t1 = min(total_duration, t1 + buffer_size / 2)

        margin = (t1 - t0) * 0.2
        if (
            self._buffered_data is not None
            and self._buffer_t0 <= t0 - margin
            and self._buffer_t1 >= t1 + margin
        ):
            return self._buffered_data, self._buffered_time

        start_idx = max(0, int(load_t0 * fs))
        stop_idx = min(len(loader), int(load_t1 * fs))
        if stop_idx <= start_idx:
            return None, None

        audio_data = np.array(loader[start_idx:stop_idx], dtype=np.float64)
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis]

        n_channels = audio_data.shape[1]
        metric = self.app_state.get_with_default('energy_metric')

        from .widgets_transform import compute_energy_envelope

        env_channels = []
        for ch in range(n_channels):
            _, ch_env = compute_energy_envelope(audio_data[:, ch], fs, metric, self.app_state)
            env_channels.append(ch_env)

        # Align channels to same length (may differ slightly between metrics)
        min_len = min(len(e) for e in env_channels)
        env_data = np.stack([e[:min_len] for e in env_channels], axis=1)
        env_time = np.linspace(load_t0, load_t1, env_data.shape[0])

        self._channel_labels = [f"Ch {i}" for i in range(n_channels)]
        self._n_channels = n_channels

        self._buffered_data = env_data
        self._buffered_time = env_time
        self._buffer_t0 = load_t0
        self._buffer_t1 = load_t1

        return env_data, env_time

    # --- Ephys envelope loading ---

    def _get_buffered_ephys_envelope(self, t0: float, t1: float):
        """Load ephys data, compute per-channel envelope, and cache."""
        from .plots_ephystrace import SharedEphysCache

        ephys_path, stream_id, _ = self.app_state.get_ephys_source()
        if not ephys_path:
            return None, None

        loader = SharedEphysCache.get_loader(ephys_path, stream_id)
        if loader is None:
            return None, None

        fs = loader.rate
        total_duration = len(loader) / fs

        window_size = t1 - t0
        buffer_size = window_size * self._buffer_multiplier
        load_t0 = max(0.0, t0 - buffer_size / 2)
        load_t1 = min(total_duration, t1 + buffer_size / 2)

        margin = (t1 - t0) * 0.2
        if (
            self._buffered_data is not None
            and self._buffer_t0 <= t0 - margin
            and self._buffer_t1 >= t1 + margin
        ):
            return self._buffered_data, self._buffered_time

        start_idx = max(0, int(load_t0 * fs))
        stop_idx = min(len(loader), int(load_t1 * fs))
        if stop_idx <= start_idx:
            return None, None

        raw = np.array(loader[start_idx:stop_idx], dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]

        n_channels = raw.shape[1]

        # Compute amplitude envelope per channel via RMS in short windows
        win_samples = max(1, int(0.01 * fs))  # 10 ms windows
        n_windows = raw.shape[0] // win_samples
        if n_windows == 0:
            return None, None

        usable = n_windows * win_samples
        reshaped = raw[:usable].reshape(n_windows, win_samples, n_channels)
        env_data = np.sqrt(np.mean(reshaped ** 2, axis=1))  # (n_windows, n_channels)
        env_time = np.linspace(load_t0, load_t1, env_data.shape[0])

        if hasattr(loader, 'channel_names'):
            self._channel_labels = loader.channel_names[:n_channels]
        else:
            self._channel_labels = [f"Ch {i}" for i in range(n_channels)]
        self._n_channels = n_channels

        self._buffered_data = env_data
        self._buffered_time = env_time
        self._buffer_t0 = load_t0
        self._buffer_t1 = load_t1

        return env_data, env_time

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

        feature_sel = self.app_state.features_sel

        if feature_sel == "Audio Waveform":
            return self._get_buffered_audio_envelope(t0, t1)

        if feature_sel in getattr(self.app_state, 'ephys_source_map', {}):
            return self._get_buffered_ephys_envelope(t0, t1)

        ds = self.app_state.ds
        time = self.app_state.time.values

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
        """Compute symmetric color range using exclusion percentile from app_state."""
        percentile = self.app_state.get_with_default('heatmap_exclusion_percentile')
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return -1.0, 1.0
        vmax = np.percentile(np.abs(valid), percentile)
        if vmax < 1e-10:
            vmax = 1.0
        return -vmax, vmax

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        if not hasattr(self.app_state, 'features_sel'):
            return

        if t0 is None or t1 is None:
            t0, t1 = self.get_current_xlim()

        self._render_heatmap(t0, t1)

    def _render_heatmap(self, t0: float, t1: float):
        self._rendering = True
        try:
            result = self._get_buffered_data(t0, t1)
            if result[0] is None:
                return

            data, time_vals = result
            norm_mode = self.app_state.get_with_default('heatmap_normalization')
            if norm_mode == "none":
                normalized = data.copy()
            elif norm_mode == "global":
                mu = np.nanmean(data)
                std = np.nanstd(data)
                normalized = (data - mu) / std if std > 0 else data - mu
            else:
                normalized = z_normalize(data)
            np.nan_to_num(normalized, copy=False, nan=0.0)
            vmin, vmax = self._compute_symmetric_levels(normalized)

            self.image_item.setImage(normalized, autoLevels=False)
            self.image_item.setLevels([vmin, vmax])
            self.colorbar.setLevels(values=(vmin, vmax))

            buf_t0 = float(time_vals[0])
            buf_t1 = float(time_vals[-1])
            duration = buf_t1 - buf_t0
            n_channels = self._n_channels

            self.image_item.setRect(pg.QtCore.QRectF(buf_t0, 0, duration, n_channels))
            self.plot_item.setYRange(0, n_channels, padding=0)

            self._update_y_axis_ticks()
        finally:
            self._rendering = False

    def _update_y_axis_ticks(self):
        """Set y-axis tick labels to channel names."""
        left_axis = self.plot_item.getAxis('left')
        ticks = [(i + 0.5, label) for i, label in enumerate(self._channel_labels)]
        left_axis.setTicks([ticks])

    # --- View range handling ---

    def _buffer_covers(self, t0: float, t1: float) -> bool:
        if self._buffered_data is None or self._context_changed():
            return False
        margin = (t1 - t0) * 0.2
        return self._buffer_t0 <= t0 - margin and self._buffer_t1 >= t1 + margin

    def _on_view_range_changed(self):
        if self._rendering:
            return
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return
        if self.app_state.time is None:
            return
        t0, t1 = self.get_current_xlim()
        if self._buffer_covers(t0, t1):
            return
        self._pending_range = (t0, t1)
        self._debounce_timer.start()

    def _debounced_update(self):
        if self._pending_range is None:
            return
        if self.app_state.time is None:
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
