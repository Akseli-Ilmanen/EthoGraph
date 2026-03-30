<<<<<<< HEAD
"""Audio waveform trace plot with smart downsampling (inspired by audian's TraceItem)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
=======
"""Audio waveform trace plot with smart downsampling.

Uses the unified ``WindowedBuffer`` + ``FileSource`` from ``modality.py``
for data caching.  Rendering uses audian-style min/max envelope
downsampling (pixel-accurate).
"""

from __future__ import annotations

from typing import Optional
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

import numpy as np
import pyqtgraph as pg

<<<<<<< HEAD
from ethograph.gui.plots_timeseriessource import RegularTimeseriesSource, TimeseriesSource

from .app_constants import AUDIOTRACE_DEBOUNCE_MS, BUFFER_COVERAGE_MARGIN, DEFAULT_BUFFER_MULTIPLIER_AUDIO
from .plots_base import BasePlot, ThrottleDebounce
from .plots_spectrogram import SharedAudioCache

if TYPE_CHECKING:
    pass

=======
from .app_constants import AUDIOTRACE_DEBOUNCE_MS, DEFAULT_BUFFER_MULTIPLIER_AUDIO
from .modality import FileSource, ModalitySource, SourceData, WindowedBuffer
from .plots_base import BasePlot, ThrottleDebounce
from .plots_spectrogram import SharedAudioCache

>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

class AudioTracePlot(BasePlot):
    """Audio waveform plot with smart min/max downsampling per pixel.

<<<<<<< HEAD
    Accepts a ``TimeseriesSource`` via :meth:`set_source` for data access.
=======
    Accepts a ``ModalitySource`` via :meth:`set_source` for data access.
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    Falls back to constructing one from ``app_state.audio_path`` if no
    source is explicitly set.
    """

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

<<<<<<< HEAD
        self.buffer = AudioTraceBuffer(app_state)
=======
        self._buffer = WindowedBuffer(
            buffer_multiplier=DEFAULT_BUFFER_MULTIPLIER_AUDIO,
        )
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

        self.label_items = []

        self._td = ThrottleDebounce(
            debounce_ms=AUDIOTRACE_DEBOUNCE_MS,
            throttle_cb=self._do_range_update,
            debounce_cb=self._do_range_update,
        )

        self.vb.sigRangeChanged.connect(self._on_view_range_changed)

<<<<<<< HEAD
    def set_source(self, source: TimeseriesSource | None):
        self.buffer.set_source(source)
=======
    @property
    def source(self) -> ModalitySource | None:
        return self._buffer.source

    def set_source(self, source: ModalitySource | None):
        self._buffer.set_source(source)
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        source = self._resolve_source()
        if source is None:
            return

        if t0 is None or t1 is None:
            xmin, xmax = self.get_current_xlim()
            t0, t1 = xmin, xmax

<<<<<<< HEAD
        pixel_width = self.vb.screenGeometry().width() or 400
        result = self.buffer.get_trace_data(t0, t1, pixel_width)
=======
        cached = self._buffer.get(t0, t1)
        if cached is None or cached.empty():
            return

        pixel_width = self.vb.screenGeometry().width() or 400
        result = _downsample_minmax(cached, t0, t1, pixel_width)
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
        if result is None:
            return

        times, amplitudes, step = result
<<<<<<< HEAD
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

    def _resolve_source(self) -> TimeseriesSource | None:
        if self.buffer.source is not None:
            return self.buffer.source
        audio_path, _ = self.app_state.get_audio_source()
        if not audio_path:
            return None
        self.buffer.set_source_from_path(audio_path)
        return self.buffer.source


=======
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

    def _resolve_source(self) -> ModalitySource | None:
        if self._buffer.source is not None:
            return self._buffer.source
        audio_path, _ = self.app_state.get_audio_source()
        if not audio_path:
            return None
        self._set_source_from_path(audio_path)
        return self._buffer.source

    def _set_source_from_path(self, audio_path: str):
        _, channel_idx = self.app_state.get_audio_source()
        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            self._buffer.set_source(None)
            return
        source = FileSource("audio", loader, channel=channel_idx)
        self._buffer.set_source(source)
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

    def _apply_y_constraints(self):
        y_min, y_max = self._get_data_y_range()
        y_range = y_max - y_min
        y_buffer = y_range * 0.1

        self.vb.setLimits(
            yMin=y_min - y_buffer,
            yMax=y_max + y_buffer,
            minYRange=y_range * 0.05,
            maxYRange=y_range + 2 * y_buffer,
        )

        auto_y = self.vb.autoRangeEnabled()[1]
        if auto_y:
            self.vb.enableAutoRange(y=True)
        else:
            self.plot_item.setYRange(y_min, y_max, padding=0)

    def _get_data_y_range(self):
        data = self.trace_item.yData
        if data is not None and len(data) > 0:
            lo = float(np.min(data))
            hi = float(np.max(data))
            if lo < hi:
                return lo, hi
        return -1.0, 1.0

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _on_view_range_changed(self):
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return
        self._td.trigger()

    def _do_range_update(self):
        t0, t1 = self.get_current_xlim()
        self.update_plot_content(t0, t1)


<<<<<<< HEAD
class AudioTraceBuffer:
    """Buffer for audio trace data with smart min/max downsampling.

    Accepts a ``TimeseriesSource`` via :meth:`set_source`.  The source
    provides ``sampling_rate`` and ``get_data(t0, t1)`` — all time-to-sample
    math is handled by the source.

    Falls back to constructing a ``RegularTimeseriesSource`` from
    ``SharedAudioCache`` via :meth:`set_source_from_path`.
    """

    def __init__(self, app_state):
        self.app_state = app_state
        self.source: TimeseriesSource | None = None
        self._current_identity: str | None = None
        self._raw_cache: np.ndarray | None = None
        self._cache_start: int = 0
        self._cache_stop: int = 0

    def set_source(self, source: TimeseriesSource | None):
        new_identity = source.identity if source else None
        if new_identity != self._current_identity:
            self.source = source
            self._current_identity = new_identity
            self._clear_cache()

    def set_source_from_path(self, audio_path: str):
        _, channel_idx = self.app_state.get_audio_source()
        identity = f"regular:audio:{audio_path}:{channel_idx}"
        if identity == self._current_identity:
            return
        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            self.source = None
            self._current_identity = None
            self._clear_cache()
            return
        self.source = RegularTimeseriesSource(
            "audio", loader, channel=channel_idx,
        )
        self._current_identity = self.source.identity
        self._clear_cache()

    def _clear_cache(self):
        self._raw_cache = None
        self._cache_start = 0
        self._cache_stop = 0

    def _covers_range(self, start: int, stop: int) -> bool:
        if self._raw_cache is None:
            return False
        n_view = stop - start
        margin = int(n_view * BUFFER_COVERAGE_MARGIN)
        return self._cache_start <= start - margin and self._cache_stop >= stop + margin

    def _build_raw_cache(self, view_start: int, view_stop: int):
        fs = self.source.sampling_rate
        n_total = self.source.n_samples
        start_s = self.source.time_range.start_s

        n_view = view_stop - view_start
        buffer_extra = int(n_view * DEFAULT_BUFFER_MULTIPLIER_AUDIO / 2)
        cache_start = max(0, view_start - buffer_extra)
        cache_stop = min(n_total, view_stop + buffer_extra)

        if cache_stop <= cache_start:
            self._raw_cache = None
            return

        t0 = start_s + cache_start / fs
        t1 = start_s + cache_stop / fs
        _, audio_data = self.source.get_data(t0, t1)

        if len(audio_data) == 0:
            self._raw_cache = None
            return

        self._raw_cache = audio_data
        self._cache_start = cache_start
        self._cache_stop = cache_start + len(audio_data)

    def get_trace_data(
        self, t0: float, t1: float, pixel_width: int = 400,
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        if self.source is None:
            return None

        fs = self.source.sampling_rate
        n_total = self.source.n_samples
        start_s = self.source.time_range.start_s

        start = max(0, int((t0 - start_s) * fs))
        stop = min(n_total, int((t1 - start_s) * fs) + 1)

        if stop <= start:
            return None

        step = max(1, (stop - start) // max(pixel_width, 400))

        if step > 1:
            actual_start = max(0, (start // step) * step)
            actual_stop = min(n_total, ((stop // step) + 1) * step)
        else:
            actual_start = start
            actual_stop = stop

        if not self._covers_range(actual_start, actual_stop):
            self._build_raw_cache(actual_start, actual_stop)

        if self._raw_cache is None:
            return None

        local_start = max(0, actual_start - self._cache_start)
        local_stop = min(len(self._raw_cache), actual_stop - self._cache_start)
        if local_stop <= local_start:
            return None

        audio_data = self._raw_cache[local_start:local_stop]
        if audio_data.ndim == 2:
            _, channel_idx = self.app_state.get_audio_source()
            channel_idx = min(channel_idx, audio_data.shape[1] - 1)
            audio_data = audio_data[:, channel_idx]

        if step > 1:
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
            plot_time = np.arange(
                actual_start, actual_start + len(plot_data) * step2, step2,
            ) / fs + start_s

            return plot_time, plot_data, step
        else:
            plot_time = np.arange(start, stop) / fs + start_s
            return plot_time, audio_data, 1
=======
def _downsample_minmax(
    data: SourceData,
    t0: float,
    t1: float,
    pixel_width: int,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Min/max envelope downsampling for waveform display.

    Returns ``(times, amplitudes, step)`` where ``step > 1`` means the
    data has been downsampled (min/max interleaved).
    """
    ts = data.timestamps
    vals = data.values

    if len(ts) == 0:
        return None

    i0 = int(np.searchsorted(ts, t0, side='left'))
    i1 = int(np.searchsorted(ts, t1, side='right'))
    if i1 <= i0:
        return None

    n_visible = i1 - i0
    step = max(1, n_visible // max(pixel_width, 400))

    if step > 1:
        aligned_start = (i0 // step) * step
        aligned_stop = min(len(vals), ((i1 // step) + 1) * step)
        chunk = vals[aligned_start:aligned_stop]

        n_segments = len(chunk) // step
        if n_segments == 0:
            return None

        usable = n_segments * step
        chunk = chunk[:usable]

        segments = np.arange(0, usable, step)
        plot_data = np.zeros(2 * n_segments)
        np.minimum.reduceat(chunk, segments, out=plot_data[0::2])
        np.maximum.reduceat(chunk, segments, out=plot_data[1::2])

        seg_starts = segments + aligned_start
        plot_time = np.empty(2 * n_segments)
        plot_time[0::2] = ts[seg_starts]
        plot_time[1::2] = ts[np.minimum(seg_starts + step - 1, len(ts) - 1)]

        return plot_time, plot_data, step
    else:
        return ts[i0:i1], vals[i0:i1], 1
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
