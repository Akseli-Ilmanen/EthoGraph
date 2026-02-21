"""Audio waveform trace plot with smart downsampling (inspired by audian's TraceItem)."""

from typing import Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from .plots_base import BasePlot
from .plots_spectrogram import SharedAudioCache


class AudioTracePlot(BasePlot):
    """Audio waveform plot with smart min/max downsampling per pixel."""

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        self.setLabel('left', 'Amplitude')

        self.trace_item = pg.PlotDataItem(
            connect='all',
            antialias=False,
            skipFiniteCheck=True,
        )
        self.trace_item.setPen(pg.mkPen(color='#00aa00', width=1.5))
        self.addItem(self.trace_item)

        self.buffer = AudioTraceBuffer(app_state)
        self.current_range = None

        self.label_items = []

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(50)
        self._debounce_timer.timeout.connect(self._debounced_update)
        self._pending_range = None

        self.vb.sigRangeChanged.connect(self._on_view_range_changed)

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Update the audio trace content with smart downsampling."""
        audio_path = self._get_audio_path()
        if not audio_path:
            return

        if t0 is None or t1 is None:
            xmin, xmax = self.get_current_xlim()
            t0, t1 = xmin, xmax

        result = self.buffer.get_trace_data(audio_path, t0, t1)
        if result is None:
            return

        times, amplitudes, step = result
        if times is not None and amplitudes is not None:
            self.trace_item.setData(times, amplitudes)

            if step > 1:
                self.trace_item.setPen(pg.mkPen(color='#00aa00', width=1.0))
                self.trace_item.setSymbol(None)
            else:
                self.trace_item.setPen(pg.mkPen(color='#00aa00', width=2.0))
                if len(times) < 200:
                    self.trace_item.setSymbol('o')
                    self.trace_item.setSymbolSize(4)
                    self.trace_item.setSymbolBrush('#00aa00')
                else:
                    self.trace_item.setSymbol(None)

        self.current_range = (t0, t1)

    def _get_audio_path(self):
        """Get audio path from app_state."""
        import os
        audio_path = getattr(self.app_state, 'audio_path', None)
        if not audio_path:
            if (hasattr(self.app_state, 'audio_folder') and
                hasattr(self.app_state, 'mics_sel') and
                hasattr(self.app_state, 'ds')):
                try:
                    audio_file = self.app_state.ds.attrs.get(self.app_state.mics_sel)
                    if audio_file:
                        audio_path = os.path.join(self.app_state.audio_folder, audio_file)
                        self.app_state.audio_path = audio_path
                except Exception:
                    pass
        return audio_path

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        """Apply amplitude range."""
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _on_view_range_changed(self):
        """Handle view range changes with debouncing."""
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return

        self._pending_range = self.get_current_xlim()
        self._debounce_timer.start()

    def _debounced_update(self):
        """Execute debounced trace update."""
        if self._pending_range is None:
            return

        t0, t1 = self._pending_range
        self._pending_range = None
        self.update_plot_content(t0, t1)

    def update_time_marker_and_window(self, frame_number: int):
        """Update time marker and refresh content for timeline changes."""
        super().update_time_marker_and_window(frame_number)

        if hasattr(self.app_state, 'ds') and self.app_state.ds is not None:
            t0, t1 = self.get_current_xlim()
            current_time = frame_number / self.app_state.ds.fps

            if self.current_range is None or current_time < self.current_range[0] or current_time > self.current_range[1]:
                self.update_plot_content(t0, t1)


class AudioTraceBuffer:
    """Buffer for audio trace data with smart downsampling (audian algorithm)."""

    def __init__(self, app_state):
        self.app_state = app_state
        self.current_path = None
        self.current_channel = None
        self.fs = None
        self.audio_loader = None

    def get_trace_data(self, audio_path, t0, t1):
        """Get audio trace data with smart min/max downsampling."""
        _, channel_idx = self.app_state.get_audio_source()

        if audio_path != self.current_path or channel_idx != self.current_channel:
            self.current_path = audio_path
            self.current_channel = channel_idx
            self.audio_loader = SharedAudioCache.get_loader(audio_path)
            if self.audio_loader:
                self.fs = self.audio_loader.rate

        if not self.audio_loader or not self.fs:
            return None

        start = max(0, int(t0 * self.fs))
        stop = min(len(self.audio_loader), int(t1 * self.fs) + 1)

        if stop <= start:
            return None

        try:
            screen_width = QApplication.primaryScreen().size().width()
        except Exception:
            screen_width = 1920

        step = max(1, (stop - start) // screen_width)

        audio_data = self.audio_loader[start:stop]
        if audio_data.ndim > 1:
            n_channels = audio_data.shape[1]
            ch = min(channel_idx, n_channels - 1)
            audio_data = audio_data[:, ch]

        if getattr(self.app_state, 'noise_reduce_enabled', False):
            try:
                import noisereduce as nr
                cache = getattr(self.app_state, 'function_params_cache', None) or {}
                nr_params = cache.get('noise_reduction', {})
                audio_data = nr.reduce_noise(y=audio_data, sr=int(self.fs), **nr_params)
            except ImportError:
                pass

        if step > 1:
            aligned_start = (start // step) * step
            aligned_stop = ((stop // step) + 1) * step
            aligned_stop = min(len(self.audio_loader), aligned_stop)

            actual_start = max(0, aligned_start)
            actual_stop = aligned_stop

            audio_data = self.audio_loader[actual_start:actual_stop]
            if audio_data.ndim > 1:
                n_channels = audio_data.shape[1]
                ch = min(channel_idx, n_channels - 1)
                audio_data = audio_data[:, ch]

            if getattr(self.app_state, 'noise_reduce_enabled', False):
                try:
                    import noisereduce as nr
                    cache = getattr(self.app_state, 'function_params_cache', None) or {}
                    nr_params = cache.get('noise_reduction', {})
                    audio_data = nr.reduce_noise(y=audio_data, sr=int(self.fs), **nr_params)
                except ImportError:
                    pass

            n_segments = len(audio_data) // step
            if n_segments == 0:
                return None

            usable_len = n_segments * step
            audio_data = audio_data[:usable_len]

            segments = np.arange(0, usable_len, step)
            plot_data = np.zeros(2 * len(segments))

            np.minimum.reduceat(audio_data, segments, out=plot_data[0::2])
            np.maximum.reduceat(audio_data, segments, out=plot_data[1::2])

            step2 = step / 2
            plot_time = np.arange(actual_start, actual_start + len(plot_data) * step2, step2) / self.fs

            return plot_time, plot_data, step
        else:
            plot_time = np.arange(start, stop) / self.fs
            return plot_time, audio_data, 1
