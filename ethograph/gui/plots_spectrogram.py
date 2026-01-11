"""Enhanced spectrogram plot inheriting from BasePlot."""

import os
import threading
import pyqtgraph as pg
import numpy as np
from audioio import AudioLoader
from scipy.signal import spectrogram
from qtpy.QtCore import Signal
from typing import Optional
from .plots_base import BasePlot


class SharedAudioCache:
    """Singleton cache for AudioLoader instances.

    This prevents repeatedly opening/closing audio files when computing
    spectrograms or accessing audio data from different parts of the application.
    Thread-safe implementation.
    """

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_loader(cls, audio_path, buffer_size=10.0):
        """Get or create AudioLoader for given audio path."""
        if not audio_path:
            return None

        with cls._lock:
            if audio_path not in cls._instances:
                try:
                    cls._instances[audio_path] = AudioLoader(audio_path, buffersize=buffer_size)
                except Exception as e:
                    print(f"Failed to load audio file {audio_path}: {e}")
                    return None
            return cls._instances[audio_path]

    @classmethod
    def clear_cache(cls):
        """Clear all cached AudioLoader instances."""
        with cls._lock:
            cls._instances.clear()



class SpectrogramPlot(BasePlot):
    """Spectrogram plot with shared sync and marker functionality from BasePlot."""

    sigFilterChanged = Signal(float, float)  # highpass, lowpass

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        # Spectrogram specific setup
        self.setLabel('left', 'Frequency', units='Hz')

        self.spec_item = pg.ImageItem()
        self.spec_item.setZValue(-20)  # Below label rectangles (z=-10) but above background
        self.addItem(self.spec_item)

        self.init_colorbar()
        self.buffer = SpectrogramBuffer(app_state)
        self.current_range = None
        self._last_update_range = None

        # Storage for label rectangles (same as LinePlot)
        self.label_items = []


        # Set frequency limits for spectrogram (0 Hz to Nyquist frequency)
        self._set_frequency_limits()

        # Connect to view changes for buffer updates
        self.vb.sigRangeChanged.connect(self._on_view_range_changed)

    def init_colorbar(self):
        """Initialize colorbar for spectrogram."""
        # Set initial levels for the spectrogram image
        vmin = self.app_state.get_with_default("vmin_db")
        vmax = self.app_state.get_with_default("vmax_db")
        self.spec_item.setLevels([vmin, vmax])

        # Set colormap for the image item
        self.spec_item.setColorMap('viridis')

        # Note: For a full colorbar widget, we would need to create a separate
        # ColorBarItem or use a different approach. For now, we'll use the
        # built-in color mapping of ImageItem.

    
    # May use in the future
    def update_colormap(self, colormap_name='viridis'):
        """Update colormap for spectrogram."""
        if hasattr(self.spec_item, 'setColorMap'):
            self.spec_item.setColorMap(colormap_name)

    def update_levels(self, vmin=None, vmax=None):
        """Update dB levels for spectrogram display."""
        if vmin is None:
            vmin = self.app_state.get_with_default("vmin_db")
        if vmax is None:
            vmax = self.app_state.get_with_default("vmax_db")
        self.spec_item.setLevels([vmin, vmax])

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Update the spectrogram content."""
        # Try to get audio path
        audio_path = getattr(self.app_state, 'audio_path', None)
        if not audio_path:
            # Try to construct audio path from audio folder and mic selection
            if (hasattr(self.app_state, 'audio_folder') and
                hasattr(self.app_state, 'mics_sel') and
                hasattr(self.app_state, 'ds')):
                try:
                    audio_file = self.app_state.ds.attrs.get(self.app_state.mics_sel)
                    if audio_file:
                        audio_path = os.path.join(self.app_state.audio_folder, audio_file)
                        self.app_state.audio_path = audio_path
                except:
                    pass

        if not audio_path:
            return

        # Get current view range if not specified
        if t0 is None or t1 is None:
            xmin, xmax = self.get_current_xlim()
            t0, t1 = xmin, xmax

        # Check if we need to recompute
        if self.buffer.needs_update(t0, t1, audio_path):
            # Compute spectrogram
            Sxx, freqs, times = self.buffer.compute(audio_path, t0, t1)

            if Sxx is not None:
                # Update image
                self.spec_item.setImage(Sxx.T, autoLevels=False)

                # Set transform to match time/frequency axes
                tr = pg.QtGui.QTransform()
                tr.translate(t0, freqs[0])
                tr.scale(
                    (t1-t0)/Sxx.shape[1],
                    (freqs[-1]-freqs[0])/Sxx.shape[0]
                )
                self.spec_item.setTransform(tr)

        self.current_range = (t0, t1)

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        """Apply frequency range for spectrogram."""
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _set_frequency_limits(self):
        """Set frequency limits based on audio sampling rate."""
        audio_path = getattr(self.app_state, 'audio_path', None)
        if audio_path:
            try:
                audio_loader = SharedAudioCache.get_loader(audio_path)
                if audio_loader:
                    fs = audio_loader.rate
                    nyquist_freq = fs / 2
                    # Set Y limits: 0 Hz to Nyquist frequency
                    self.vb.setLimits(yMin=0, yMax=nyquist_freq)
                    return
            except Exception:
                pass

        # Fallback: reasonable default frequency range for audio
        self.vb.setLimits(yMin=0, yMax=25000)  # 0 Hz to 25 kHz

    def _apply_y_constraints(self):
        """Apply frequency-based y-axis constraints."""
        self._set_frequency_limits()

    def update_buffer_settings(self):
        """Update buffer settings from app state."""
        self.buffer.update_buffer_size()

    def _on_view_range_changed(self):
        """Handle view range changes to update spectrogram if needed."""
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return

        # Get current x-range
        current_xlim = self.get_current_xlim()
        t0, t1 = current_xlim

        # Check if the view has changed significantly
        if (self._last_update_range is None or
            abs(t0 - self._last_update_range[0]) > 0.1 or
            abs(t1 - self._last_update_range[1]) > 0.1):

            # Update spectrogram content for new range
            self.update_plot_content(t0, t1)
            self._last_update_range = (t0, t1)

    def update_time_marker_and_window(self, frame_number: int):
        """Update time marker and refresh spectrogram content for timeline changes."""
        # Call parent method to update time marker
        super().update_time_marker_and_window(frame_number)

        # For spectrogram, we also need to update content when timeline changes
        # since the user might have jumped to a completely different time range
        if hasattr(self.app_state, 'ds') and self.app_state.ds is not None:
            current_xlim = self.get_current_xlim()
            t0, t1 = current_xlim

            # Force update if this is a significant timeline jump
            current_time = frame_number / self.app_state.ds.fps
            if (self._last_update_range is None or
                current_time < self._last_update_range[0] or
                current_time > self._last_update_range[1]):

                self.update_plot_content(t0, t1)
                self._last_update_range = (t0, t1)




class SpectrogramBuffer:
    """Smart buffering for spectrogram computation."""

    def __init__(self, app_state):
        self.app_state = app_state
        self.cache = {}
        self.current_path = None
        # Use spec_buffer for spectrogram-specific buffering, fall back to buffer_multiplier
        spec_buffer = getattr(app_state, 'spec_buffer', None)
        if spec_buffer is not None and spec_buffer > 0:
            self.buffer_multiplier = spec_buffer
        else:
            try:
                self.buffer_multiplier = app_state.get_with_default("buffer_multiplier")
            except KeyError:
                self.buffer_multiplier = 5.0  # Final fallback

        # Ensure buffer_multiplier is never None
        if self.buffer_multiplier is None:
            self.buffer_multiplier = 5.0




    def needs_update(self, t0, t1, audio_path):
        """Check if buffer needs update."""
        if audio_path != self.current_path:
            self.cache.clear()
            self.current_path = audio_path
            return True

        # Check if we have any cached data that covers this time range
        for cached_key in self.cache.keys():
            cached_t0, cached_t1 = cached_key
            # Check if cached range fully covers requested range with some tolerance
            if (cached_t0 <= t0 + 0.01 and cached_t1 >= t1 - 0.01):
                return False

        return True

    def compute(self, audio_path, t0, t1):
        """Compute or retrieve spectrogram from cache."""
        # First check if we can find cached data that covers this range
        for cached_key, cached_data in self.cache.items():
            cached_t0, cached_t1 = cached_key
            if (cached_t0 <= t0 + 0.01 and cached_t1 >= t1 - 0.01):
                # Found cached data that covers requested range
                Sxx_db, freqs, times = cached_data

                # Extract the portion we need
                time_mask = (times >= t0 - 0.01) & (times <= t1 + 0.01)
                if np.any(time_mask):
                    return Sxx_db[:, time_mask], freqs, times[time_mask]
                else:
                    return cached_data  # Return full cached data if extraction fails

        key = self._get_cache_key(t0, t1)

        if key in self.cache:
            return self.cache[key]

        # # Check if we're in streaming mode and can use streaming cache
        # if hasattr(self.app_state, 'sync_state') and self.app_state.sync_state == 'pyav_stream_mode':
        #     return self._compute_streaming(audio_path, t0, t1, key)

        # Compute with buffer
        buffer_size = (t1 - t0) * self.buffer_multiplier
        buffer_t0 = max(0, t0 - buffer_size/2)
        buffer_t1 = t1 + buffer_size/2

        # Get audio data
        audio_loader = SharedAudioCache.get_loader(audio_path)
        if not audio_loader:
            return None, None, None

        fs = audio_loader.rate
        i0 = int(buffer_t0 * fs)
        i1 = int(buffer_t1 * fs)

        audio_data = audio_loader[i0:i1]
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Compute spectrogram
        nfft = self.app_state.get_with_default("nfft")
        hop = int(nfft * self.app_state.get_with_default("hop_frac"))

        freqs, times, Sxx = spectrogram(
            audio_data, fs=fs,
            nperseg=nfft, noverlap=nfft-hop
        )

        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Adjust times to absolute
        times += buffer_t0

        # Cache result
        self.cache[key] = (Sxx_db, freqs, times)

        # Trim cache if too large
        if len(self.cache) > 10:
            # Remove oldest entries
            keys = sorted(self.cache.keys())
            for k in keys[:-10]:
                del self.cache[k]

        return Sxx_db, freqs, times

    def _get_cache_key(self, t0, t1):
        """Generate cache key for time range."""
        # Use actual time range with buffer for more precise caching
        buffer_size = (t1 - t0) * self.buffer_multiplier
        buffer_t0 = max(0, t0 - buffer_size/2)
        buffer_t1 = t1 + buffer_size/2
        # Round to nearest 0.01s for finer granularity
        return (round(buffer_t0, 2), round(buffer_t1, 2))

    def update_buffer_size(self):
        """Update buffer multiplier from app state settings."""
        spec_buffer = getattr(self.app_state, 'spec_buffer', None)
        if spec_buffer is not None and spec_buffer > 0:
            self.buffer_multiplier = spec_buffer
        else:
            try:
                self.buffer_multiplier = self.app_state.get_with_default("buffer_multiplier")
            except KeyError:
                self.buffer_multiplier = 5.0  # Final fallback

        # Ensure buffer_multiplier is never None
        if self.buffer_multiplier is None:
            self.buffer_multiplier = 5.0

        # Clear cache when buffer size changes to force recomputation
        self.cache.clear()