"""Enhanced spectrogram plot inheriting from BasePlot."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyqtgraph as pg
from audioio import AudioLoader
from qtpy.QtCore import QTimer, Signal
from scipy.signal import spectrogram

from .plots_base import BasePlot
from .app_constants import (
    SPECTROGRAM_DEBOUNCE_MS,
    DEFAULT_BUFFER_MULTIPLIER,
    BUFFER_COVERAGE_MARGIN,
    DEFAULT_FALLBACK_MAX_FREQUENCY,
    Z_INDEX_BACKGROUND,
)

if TYPE_CHECKING:
    from .spectrogram_sources import SpectrogramSource


class SharedAudioCache:
    """Singleton cache for AudioLoader instances.

    Multiple GUI components (waveform plot, spectrogram, heatmap, envelope
    overlay, set_time) all need the same audio loader. Without caching, each
    would open the file independently, wasting file handles and memory.
    """

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_loader(cls, audio_path, buffer_size=10.0):
        if not audio_path:
            return None

        with cls._lock:
            if audio_path not in cls._instances:
                try:
                    cls._instances[audio_path] = AudioLoader(audio_path, buffersize=buffer_size)
                except (OSError, IOError, ValueError) as e:
                    print(f"Failed to load audio file {audio_path}: {e}")
                    return None
            return cls._instances[audio_path]

    @classmethod
    def clear_cache(cls):
        with cls._lock:
            cls._instances.clear()



class SpectrogramPlot(BasePlot):
    """Spectrogram plot with shared sync and marker functionality from BasePlot."""

    sigFilterChanged = Signal(float, float)
    bufferUpdated = Signal()

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        self.setLabel('left', 'Frequency', units='Hz')

        self.spec_item = pg.ImageItem()
        self.spec_item.setZValue(Z_INDEX_BACKGROUND)
        self.addItem(self.spec_item)

        self.init_colorbar()
        self.buffer = SpectrogramBuffer(app_state)
        self.current_range = None
        self.source: SpectrogramSource | None = None

        self._set_frequency_limits()

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(SPECTROGRAM_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._debounced_update)
        self._pending_range = None

        self.vb.sigRangeChanged.connect(self._on_view_range_changed)

    def init_colorbar(self):
        """Initialize colorbar for spectrogram."""
        vmin = self.app_state.get_with_default("vmin_db")
        vmax = self.app_state.get_with_default("vmax_db")
        self.spec_item.setLevels([vmin, vmax])

        colormap = self.app_state.get_with_default("spec_colormap")
        self.spec_item.setColorMap(colormap)

    def update_colormap(self, colormap_name: str):
        """Update colormap for spectrogram."""
        self.spec_item.setColorMap(colormap_name)

    def update_levels(self, vmin=None, vmax=None):
        """Update dB levels for spectrogram display."""
        if vmin is None:
            vmin = self.app_state.get_with_default("vmin_db")
        if vmax is None:
            vmax = self.app_state.get_with_default("vmax_db")
        self.spec_item.setLevels([vmin, vmax])

    def set_source(self, source: SpectrogramSource | None):
        """Set a custom spectrogram source (e.g. XarraySource). Clears buffer."""
        self.source = source
        self.buffer._clear_buffer()
        if source is not None:
            self._set_frequency_limits()

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Update the spectrogram content."""
        source = self.source
        if source is None:
            source = self._build_fallback_audio_source()
        if source is None:
            return

        if t0 is None or t1 is None:
            xmin, xmax = self.get_current_xlim()
            t0, t1 = xmin, xmax

        result = self.buffer.get_spectrogram(source, t0, t1)
        if result is None:
            return

        Sxx_db, spec_rect = result
        if Sxx_db is not None and self.buffer.buffer_changed:
            self.spec_item.setImage(Sxx_db.T, autoLevels=False)
            self.spec_item.setRect(*spec_rect)
            self.buffer.buffer_changed = False
            self.bufferUpdated.emit()

        self.current_range = (t0, t1)

    def _build_fallback_audio_source(self):
        """Build AudioFileSource from app_state as backward-compat fallback."""
        from .spectrogram_sources import build_audio_source

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

        if not audio_path:
            return None

        return build_audio_source(self.app_state)

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        """Apply frequency range for spectrogram."""
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _set_frequency_limits(self):
        """Set frequency limits based on source sampling rate."""
        if self.source is not None:
            nyquist_freq = self.source.rate / 2
            self.vb.setLimits(yMin=0, yMax=nyquist_freq)
            return

        audio_path = getattr(self.app_state, 'audio_path', None)
        if audio_path:
            try:
                audio_loader = SharedAudioCache.get_loader(audio_path)
                if audio_loader:
                    fs = audio_loader.rate
                    nyquist_freq = fs / 2
                    self.vb.setLimits(yMin=0, yMax=nyquist_freq)
                    return
            except (OSError, IOError, AttributeError):
                pass

        self.vb.setLimits(yMin=0, yMax=DEFAULT_FALLBACK_MAX_FREQUENCY)

    def _apply_y_constraints(self):
        """Apply frequency-based y-axis constraints."""
        self._set_frequency_limits()

    def update_buffer_settings(self):
        """Update buffer settings from app state."""
        self.buffer.update_buffer_size()

    def _on_view_range_changed(self):
        """Handle view range changes with debouncing."""
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return

        self._pending_range = self.get_current_xlim()
        self._debounce_timer.start()

    def _debounced_update(self):
        """Execute debounced spectrogram update."""
        if self._pending_range is None:
            return

        t0, t1 = self._pending_range
        self._pending_range = None
        self.update_plot_content(t0, t1)

    def update_time_marker_and_window(self, frame_number: int):
        """Update time marker and refresh spectrogram content for timeline changes."""
        super().update_time_marker_and_window(frame_number)

        if hasattr(self.app_state, 'ds') and self.app_state.ds is not None:
            t0, t1 = self.get_current_xlim()
            current_time = frame_number / self.app_state.ds.fps

            if self.current_range is None or current_time < self.current_range[0] or current_time > self.current_range[1]:
                self.update_plot_content(t0, t1)




class SpectrogramBuffer:
    """Single-buffer spectrogram cache inspired by audian's BufferedData pattern."""

    def __init__(self, app_state):
        self.app_state = app_state
        self.current_identity: str | None = None
        self.buffer_multiplier = self._get_buffer_multiplier()

        self.Sxx_db = None
        self.freqs = None
        self.times = None
        self.buffer_t0 = 0.0
        self.buffer_t1 = 0.0
        self.fs = None
        self.fresolution = 1.0

        self.buffer_changed = False

    def _get_buffer_multiplier(self):
        spec_buffer = getattr(self.app_state, 'spec_buffer', None)
        if spec_buffer is not None and spec_buffer > 0:
            return spec_buffer
        try:
            val = self.app_state.get_with_default("buffer_multiplier")
            return val if val is not None else DEFAULT_BUFFER_MULTIPLIER
        except (KeyError, AttributeError):
            return DEFAULT_BUFFER_MULTIPLIER

    def _covers_range(self, t0, t1):
        """Check if current buffer covers requested range with margin."""
        if self.Sxx_db is None:
            return False
        margin = (t1 - t0) * BUFFER_COVERAGE_MARGIN
        return self.buffer_t0 <= t0 - margin and self.buffer_t1 >= t1 + margin

    def get_spectrogram(self, source: SpectrogramSource, t0: float, t1: float):
        """Get spectrogram data, computing only if necessary."""
        if source.identity != self.current_identity:
            self._clear_buffer()
            self.current_identity = source.identity

        if self._covers_range(t0, t1):
            return self.Sxx_db, self._get_spec_rect()

        self._compute_buffer(source, t0, t1)

        if self.Sxx_db is None:
            return None

        return self.Sxx_db, self._get_spec_rect()

    def _compute_buffer(self, source: SpectrogramSource, t0: float, t1: float):
        """Compute spectrogram for buffered range."""
        self.fs = source.rate

        window_size = t1 - t0
        buffer_size = window_size * self.buffer_multiplier
        self.buffer_t0 = max(0.0, t0 - buffer_size / 2)
        self.buffer_t1 = t1 + buffer_size / 2

        max_time = source.duration
        if self.buffer_t1 > max_time:
            self.buffer_t1 = max_time

        audio_data = source.get_data(self.buffer_t0, self.buffer_t1)

        if len(audio_data) == 0:
            return

        nfft = self.app_state.get_with_default("nfft")
        hop_frac = self.app_state.get_with_default("hop_frac")

        # Clamp nfft to data length (power-of-2) for low sample-rate sources
        max_nfft = 1 << (len(audio_data).bit_length() - 1)  # largest pow2 <= len
        if max_nfft < 4:
            return
        nfft = min(nfft, max_nfft)

        if source.supports_noise_reduction and getattr(self.app_state, 'noise_reduce_enabled', False):
            try:
                from ethograph.features.audio_changepoints import apply_noise_reduction
                prop_decrease = getattr(self.app_state, 'noise_reduce_prop_decrease', 1.0)
                audio_data = apply_noise_reduction(audio_data, int(self.fs), nfft, hop_frac, prop_decrease=prop_decrease)
            except ImportError:
                pass

        hop = int(nfft * hop_frac)

        if len(audio_data) < nfft:
            return

        with np.errstate(under='ignore'):
            freqs, times, Sxx = spectrogram(
                audio_data, fs=self.fs,
                nperseg=nfft, noverlap=nfft - hop
            )

        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        np.nan_to_num(Sxx_db, copy=False, nan=-100.0, posinf=0.0, neginf=-100.0)
        self.Sxx_db = Sxx_db
        self.freqs = freqs
        self.times = times + self.buffer_t0
        self.fresolution = self.fs / nfft if nfft > 0 else 1.0
        self.buffer_changed = True

    def _get_spec_rect(self):
        """Get rectangle [x, y, width, height] for setRect."""
        if self.Sxx_db is None or self.freqs is None:
            return [0, 0, 1, 1]

        t_duration = self.buffer_t1 - self.buffer_t0
        f_max = self.freqs[-1] + self.fresolution if len(self.freqs) > 0 else self.fs / 2

        return [self.buffer_t0, 0, t_duration, f_max]

    def _clear_buffer(self):
        self.Sxx_db = None
        self.freqs = None
        self.times = None
        self.buffer_t0 = 0.0
        self.buffer_t1 = 0.0
        self.buffer_changed = False
        self.current_identity = None

    def update_buffer_size(self):
        self.buffer_multiplier = self._get_buffer_multiplier()
        self._clear_buffer()