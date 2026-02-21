"""Three-panel audio layout for no-video mode: waveform + spectrogram + feature."""

from typing import Any, Dict

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .app_constants import ENVELOPE_OVERLAY_COLOR, ENVELOPE_OVERLAY_DEBOUNCE_MS, ENVELOPE_OVERLAY_WIDTH

from .label_drawing_mixin import LabelDrawingMixin
from .plots_audiotrace import AudioTracePlot
from .plots_heatmap import HeatmapPlot
from .plots_lineplot import LinePlot
from .plots_overlay import OverlayManager
from .plots_spectrogram import SpectrogramPlot


class TimeSlider(QWidget):
    """Horizontal slider mapped to a time range, emitting time in seconds."""

    time_changed = Signal(float)

    _SLIDER_STEPS = 10000

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._SLIDER_STEPS)
        self._slider.valueChanged.connect(self._on_slider_moved)

        self._label = QLabel("0.00 s")
        self._label.setFixedWidth(80)

        layout.addWidget(self._slider)
        layout.addWidget(self._label)

        self._t_min = 0.0
        self._t_max = 1.0

    def set_time_range(self, t_min: float, t_max: float):
        self._t_min = t_min
        self._t_max = max(t_min + 1e-6, t_max)

    def set_time(self, t: float):
        """Set slider position without emitting time_changed."""
        if self._t_max <= self._t_min:
            return
        frac = (t - self._t_min) / (self._t_max - self._t_min)
        frac = max(0.0, min(1.0, frac))
        self._slider.blockSignals(True)
        self._slider.setValue(int(frac * self._SLIDER_STEPS))
        self._slider.blockSignals(False)
        self._update_label(t)

    def _on_slider_moved(self, value: int):
        frac = value / self._SLIDER_STEPS
        t = self._t_min + frac * (self._t_max - self._t_min)
        self._update_label(t)
        self.time_changed.emit(t)

    def _update_label(self, t: float):
        minutes = int(abs(t) // 60)
        seconds = abs(t) % 60
        sign = "-" if t < 0 else ""
        if minutes:
            self._label.setText(f"{sign}{minutes}:{seconds:05.2f}")
        else:
            self._label.setText(f"{sign}{seconds:.2f} s")


class MultiPanelContainer(LabelDrawingMixin, QWidget):
    """Three-panel layout: audio waveform (top), spectrogram (mid), feature (bottom).

    All three panels share the same x-axis via pyqtgraph linking.
    Labels, changepoints, and time markers are drawn on all visible panels.
    """

    plot_changed = Signal(str)
    labels_redraw_needed = Signal()
    spectrogram_overlay_shown = Signal()

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state

        # --- Plots ---
        self.audio_trace_plot = AudioTracePlot(app_state)
        self.spectrogram_plot = SpectrogramPlot(app_state)
        self.line_plot = LinePlot(None, app_state)
        self.heatmap_plot = HeatmapPlot(app_state)
        self.ephys_trace_plot = None  # not used in no-video mode

        # Bottom panel: line_plot or heatmap_plot
        self._bottom_plot = self.line_plot
        self._bottom_type = "lineplot"

        # current_plot semantics: always the bottom (feature) panel
        self.current_plot = self._bottom_plot
        self.current_plot_type = self._bottom_type

        # --- Mixin state ---
        self.label_mappings: Dict[int, Dict[str, Any]] = {}
        self.audio_overlay_type = None
        self.audio_cp_items: list = []
        self.dataset_cp_items: list = []

        self.overlay_manager = OverlayManager()
        self.line_plot.vb.sigYRangeChanged.connect(
            lambda: self.overlay_manager.rescale_for_plot(self.line_plot)
        )
        self.audio_trace_plot.vb.sigYRangeChanged.connect(
            lambda: self.overlay_manager.rescale_for_plot(self.audio_trace_plot)
        )

        # --- Layout ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.addWidget(self.audio_trace_plot)
        self._splitter.addWidget(self.spectrogram_plot)
        self._splitter.addWidget(self.line_plot)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 1)

        self.heatmap_plot.hide()

        main_layout.addWidget(self._splitter)

        self.time_slider = TimeSlider()
        self.time_slider.time_changed.connect(self._on_slider_time)
        main_layout.addWidget(self.time_slider)

        # --- Audio playback state ---
        self._playback_playing = False
        self._playback_timer = QTimer()
        self._playback_timer.setInterval(33)  # ~30 fps marker update
        self._playback_timer.timeout.connect(self._advance_playback_marker)
        self._playback_start_time = 0.0
        self._playback_start_wall = 0.0

        # --- X-axis linking ---
        self.spectrogram_plot.plotItem.setXLink(self.audio_trace_plot.plotItem)
        self.line_plot.plotItem.setXLink(self.audio_trace_plot.plotItem)
        self.heatmap_plot.plotItem.setXLink(self.audio_trace_plot.plotItem)

        # Hide x-axis labels on upper panels (only bottom shows time)
        self.audio_trace_plot.plotItem.getAxis("bottom").setStyle(showValues=False)
        self.audio_trace_plot.plotItem.setLabel("bottom", "")
        self.spectrogram_plot.plotItem.getAxis("bottom").setStyle(showValues=False)
        self.spectrogram_plot.plotItem.setLabel("bottom", "")

    # --- Public API matching PlotContainer interface ---

    def get_current_plot(self):
        return self._bottom_plot

    def get_current_xlim(self):
        return self.audio_trace_plot.get_current_xlim()

    def set_x_range(self, mode="default", curr_xlim=None, center_on_frame=None):
        return self.audio_trace_plot.set_x_range(
            mode=mode, curr_xlim=curr_xlim, center_on_frame=center_on_frame,
        )

    @property
    def vb(self):
        return self._bottom_plot.vb

    def get_hovered_plot(self):
        """Return the plot panel currently under the mouse cursor."""
        for plot in self._visible_plots():
            if plot.underMouse():
                return plot
        return self._bottom_plot

    def _visible_plots(self):
        """Yield all currently visible plot panels."""
        for plot in (self.audio_trace_plot, self.spectrogram_plot, self._bottom_plot):
            if plot.isVisible():
                yield plot

    def update_time_marker_by_time(self, time_s: float):
        for plot in (self.audio_trace_plot, self.spectrogram_plot, self._bottom_plot):
            plot.update_time_marker(time_s)
        self.time_slider.set_time(time_s)

    def update_time_marker_and_window(self, frame_number):
        fps = self.app_state.effective_fps
        current_time = frame_number / fps
        self.update_time_marker_by_time(current_time)

    def apply_y_range(self, ymin, ymax):
        return self._bottom_plot.apply_y_range(ymin, ymax)

    def toggle_axes_lock(self):
        for plot in (self.audio_trace_plot, self.spectrogram_plot, self._bottom_plot):
            plot.toggle_axes_lock()

    # --- Bottom panel switching ---

    def switch_to_lineplot(self):
        if self._bottom_type == "lineplot":
            return
        self._swap_bottom_panel(self.line_plot, "lineplot")

    def switch_to_heatmap(self):
        if self._bottom_type == "heatmap":
            return
        self._swap_bottom_panel(self.heatmap_plot, "heatmap")

    def _swap_bottom_panel(self, new_plot, new_type):
        prev_xlim = self._bottom_plot.get_current_xlim()
        prev_marker = self._bottom_plot.time_marker.value()
        sizes = self._splitter.sizes()

        idx = self._splitter.indexOf(self._bottom_plot)
        self._bottom_plot.hide()

        self._splitter.insertWidget(idx, new_plot)
        new_plot.show()
        new_plot.plotItem.setXLink(self.audio_trace_plot.plotItem)

        self._bottom_plot = new_plot
        self._bottom_type = new_type
        self.current_plot = new_plot
        self.current_plot_type = new_type

        self._splitter.setSizes(sizes)
        new_plot.set_x_range(mode="preserve", curr_xlim=prev_xlim)
        new_plot.update_time_marker(prev_marker)

        self.plot_changed.emit(new_type)
        self.labels_redraw_needed.emit()

    def is_lineplot(self):
        return self._bottom_type == "lineplot"

    def is_heatmap(self):
        return self._bottom_type == "heatmap"

    def is_spectrogram(self):
        return False  # spectrogram is always visible, not the "current" mode

    def is_audiotrace(self):
        return False  # audio trace is always visible

    def is_ephystrace(self):
        return False

    def has_spectrogram_overlay(self) -> bool:
        return False  # no overlay needed, spectrogram is its own panel

    # --- Audio panel updates ---

    def update_audio_panels(self):
        """Refresh audio-driven panels (waveform + spectrogram) after mic change."""
        from .data_sources import build_audio_source

        source = build_audio_source(self.app_state)
        self.spectrogram_plot.set_source(source)

        # Use data-based range if plots haven't been initialized yet
        t0, t1 = self.audio_trace_plot.get_current_xlim()
        time = self.app_state.time
        if time is not None:
            vals = np.asarray(time)
            data_t0, data_t1 = float(vals[0]), float(vals[-1])
            # Detect uninitialised default range (pyqtgraph starts at -0.5..0.5 or similar)
            if t1 - t0 < 0.01 or t0 < data_t0 - 1000 or t1 > data_t1 + 1000:
                window = self.app_state.get_with_default("window_size")
                t0 = data_t0
                t1 = min(data_t0 + float(window), data_t1)
                self.audio_trace_plot.vb.setXRange(t0, t1, padding=0)

        self.audio_trace_plot.update_plot(t0=t0, t1=t1)
        self.spectrogram_plot.update_plot(t0=t0, t1=t1)

    # --- Time slider ---

    def _on_slider_time(self, time_s: float):
        self.update_time_marker_by_time(time_s)
        xlim = self.get_current_xlim()
        if time_s < xlim[0] or time_s > xlim[1]:
            window_size = self.app_state.get_with_default("window_size")
            half = window_size / 2.0
            self.audio_trace_plot.vb.setXRange(
                time_s - half, time_s + half, padding=0,
            )

    def update_time_range_from_data(self):
        time = self.app_state.time
        if time is not None:
            vals = np.asarray(time)
            self.time_slider.set_time_range(float(vals[0]), float(vals[-1]))

    # --- Audio playback (space key) ---

    def toggle_pause_resume(self):
        """Play/pause audio from the current time marker position."""
        if self._playback_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("sounddevice not installed — cannot play audio.")
            return

        from .plots_spectrogram import SharedAudioCache
        import time as _time

        audio_path = getattr(self.app_state, 'audio_path', None)
        if not audio_path:
            return

        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            return

        fs = loader.rate
        _, channel_idx = self.app_state.get_audio_source()

        # Start from the current time marker
        current_time = self.audio_trace_plot.time_marker.value()
        xlim = self.get_current_xlim()
        end_time = xlim[1]

        start_sample = max(0, int(current_time * fs))
        end_sample = min(len(loader), int(end_time * fs))
        if end_sample <= start_sample:
            return

        audio_data = loader[start_sample:end_sample]
        if audio_data.ndim > 1:
            ch = min(channel_idx, audio_data.shape[1] - 1)
            audio_data = audio_data[:, ch]

        self._playback_start_time = current_time
        self._playback_start_wall = _time.perf_counter()
        self._playback_playing = True

        speed = self.app_state.audio_playback_speed
        sd.stop()
        sd.play(audio_data, samplerate=int(fs * speed))
        self._playback_timer.start()

    def _stop_playback(self):
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass
        self._playback_timer.stop()
        self._playback_playing = False

    def _advance_playback_marker(self):
        import time as _time
        elapsed = _time.perf_counter() - self._playback_start_wall
        speed = self.app_state.audio_playback_speed
        current = self._playback_start_time + elapsed * speed
        xlim = self.get_current_xlim()

        if current > xlim[1]:
            self._stop_playback()
            return

        self.update_time_marker_by_time(current)

    # --- Overlay stubs (not needed in multi-panel but called by widgets) ---

    def show_audio_overlay(self, overlay_type: str):
        pass  # spectrogram/waveform always visible

    def hide_audio_overlay(self):
        pass

    def update_audio_overlay(self):
        pass

    def apply_overlay_levels(self, vmin: float, vmax: float):
        pass

    def apply_overlay_colormap(self, colormap_name: str):
        pass

    def show_confidence_plot(self, confidence_data):
        pass  # TODO: could add to bottom panel

    def hide_confidence_plot(self):
        pass

    def draw_amplitude_envelope(self, *args, **kwargs):
        pass

    def clear_amplitude_envelope(self):
        pass

    # --- Envelope overlay ---

    def _get_envelope_target(self) -> str:
        """Get envelope target: 'audio' or 'feature'."""
        return getattr(self.app_state, '_envelope_target', 'audio')

    def _get_envelope_host_plot(self):
        target = self._get_envelope_target()
        if target == "audio":
            return self.audio_trace_plot
        return self._bottom_plot

    def show_envelope_overlay(self):
        host = self._get_envelope_host_plot()
        if host is None:
            return

        self.hide_envelope_overlay()

        t0, t1 = host.get_current_xlim()
        signal_data, fs, buf_t0 = self._load_envelope_data(host, t0, t1)
        if signal_data is None:
            return

        from .widgets_transform import compute_energy_envelope
        metric = self.app_state.get_with_default('energy_metric')
        env_time, env_data = compute_energy_envelope(signal_data, fs, metric, self.app_state)

        if env_data is None or len(env_data) == 0:
            return

        # Trim to visible window
        mask = (env_time >= t0) & (env_time <= t1)
        env_time = env_time[mask]
        env_data = env_data[mask]

        if len(env_data) == 0:
            return

        item = pg.PlotCurveItem(
            pen=pg.mkPen(color=ENVELOPE_OVERLAY_COLOR, width=ENVELOPE_OVERLAY_WIDTH),
        )
        self.overlay_manager.add_scaled_overlay('envelope', host, item, env_time, env_data)

        self._envelope_debounce = QTimer()
        self._envelope_debounce.setSingleShot(True)
        self._envelope_debounce.setInterval(ENVELOPE_OVERLAY_DEBOUNCE_MS)
        self._envelope_debounce.timeout.connect(self._refresh_envelope_data)

        def on_x_range_changed():
            if self.overlay_manager.has_overlay('envelope'):
                self._envelope_debounce.start()

        host.vb.sigXRangeChanged.connect(on_x_range_changed)
        self._envelope_xrange_updater = on_x_range_changed

    def hide_envelope_overlay(self):
        updater = getattr(self, '_envelope_xrange_updater', None)
        if updater:
            for plot in (self.audio_trace_plot, self._bottom_plot):
                try:
                    plot.vb.sigXRangeChanged.disconnect(updater)
                except (RuntimeError, TypeError):
                    pass
            self._envelope_xrange_updater = None

        debounce = getattr(self, '_envelope_debounce', None)
        if debounce:
            debounce.stop()
            self._envelope_debounce = None

        self.overlay_manager.remove_overlay('envelope')

    def _refresh_envelope_data(self):
        host = self._get_envelope_host_plot()
        if host is None or not self.overlay_manager.has_overlay('envelope'):
            return
        t0, t1 = host.get_current_xlim()
        signal_data, fs, buf_t0 = self._load_envelope_data(host, t0, t1)
        if signal_data is None:
            return

        from .widgets_transform import compute_energy_envelope
        metric = self.app_state.get_with_default('energy_metric')
        env_time, env_data = compute_energy_envelope(signal_data, fs, metric, self.app_state)
        if env_data is None or len(env_data) == 0:
            return
        mask = (env_time >= t0) & (env_time <= t1)
        self.overlay_manager.update_overlay_data('envelope', env_time[mask], env_data[mask])

    def _load_envelope_data(self, host, t0, t1):
        """Load signal data for envelope computation."""
        target = self._get_envelope_target()
        if target == "audio":
            from .plots_spectrogram import SharedAudioCache
            audio_path = getattr(self.app_state, 'audio_path', None)
            if not audio_path:
                return None, None, None
            loader = SharedAudioCache.get_loader(audio_path)
            if loader is None:
                return None, None, None
            fs = loader.rate
            _, channel_idx = self.app_state.get_audio_source()
            start_idx = max(0, int(t0 * fs))
            stop_idx = min(len(loader), int(t1 * fs))
            if stop_idx <= start_idx:
                return None, None, None
            audio_data = np.array(loader[start_idx:stop_idx], dtype=np.float64)
            if audio_data.ndim > 1:
                ch = min(channel_idx, audio_data.shape[1] - 1)
                audio_data = audio_data[:, ch]
            return audio_data, fs, t0
        else:
            from ethograph.utils.data_utils import get_time_coord, sel_valid
            feature_sel = getattr(self.app_state, 'features_sel', None)
            ds = getattr(self.app_state, 'ds', None)
            if not feature_sel or ds is None or feature_sel not in ds:
                return None, None, None
            da = ds[feature_sel]
            time_coord = get_time_coord(da)
            if time_coord is None:
                return None, None, None
            time_vals = time_coord.values
            ds_kwargs = self.app_state.get_ds_kwargs()
            data, _ = sel_valid(da, ds_kwargs)
            if data.ndim > 1:
                data = data[:, 0]
            mask = (time_vals >= t0) & (time_vals <= t1)
            if not np.any(mask):
                return None, None, None
            dt = np.median(np.diff(time_vals[:min(1000, len(time_vals))]))
            fs = 1.0 / dt if dt > 0 else 1.0
            return np.asarray(data[mask], dtype=np.float64), fs, t0

    def clear_audio_cache(self):
        from .plots_spectrogram import SharedAudioCache
        SharedAudioCache.clear_cache()
        if hasattr(self.spectrogram_plot, "buffer"):
            self.spectrogram_plot.buffer._clear_buffer()
        if hasattr(self.audio_trace_plot, "buffer"):
            self.audio_trace_plot.buffer.audio_loader = None
            self.audio_trace_plot.buffer.current_path = None
