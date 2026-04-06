"""Audio waveform trace plot with smart downsampling.

Uses the unified ``WindowedBuffer`` + ``FileSource`` from ``modality.py``
for data caching.  Rendering uses audian-style min/max envelope
downsampling (pixel-accurate).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg

from .app_constants import AUDIOTRACE_DEBOUNCE_MS, DEFAULT_BUFFER_MULTIPLIER_AUDIO
from .modality import FileSource, ModalitySource, SourceData, WindowedBuffer
from .plots_base import BasePlot, ThrottleDebounce
from .plots_spectrogram import SharedAudioCache


class AudioTracePlot(BasePlot):
    """Audio waveform plot with smart min/max downsampling per pixel.

    Accepts a ``ModalitySource`` via :meth:`set_source` for data access.
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

        self._buffer = WindowedBuffer(
            buffer_multiplier=DEFAULT_BUFFER_MULTIPLIER_AUDIO,
        )

        self.label_items = []

        self._td = ThrottleDebounce(
            debounce_ms=AUDIOTRACE_DEBOUNCE_MS,
            throttle_cb=self._do_range_update,
            debounce_cb=self._do_range_update,
        )

        self.vb.sigRangeChanged.connect(self._on_view_range_changed)

    @property
    def source(self) -> ModalitySource | None:
        return self._buffer.source

    def set_source(self, source: ModalitySource | None):
        self._buffer.set_source(source)

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        source = self._resolve_source()
        if source is None:
            return

        if t0 is None or t1 is None:
            xmin, xmax = self.get_current_xlim()
            t0, t1 = xmin, xmax

        cached = self._buffer.get(t0, t1)
        if cached is None or cached.empty():
            return

        pixel_width = self.vb.screenGeometry().width() or 400
        result = _downsample_minmax(cached, t0, t1, pixel_width)
        if result is None:
            return

        times, amplitudes, step = result
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
        if not self.isVisible():
            return
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return
        self._td.trigger()

    def _do_range_update(self):
        if not self.isVisible():
            return
        t0, t1 = self.get_current_xlim()
        self.update_plot_content(t0, t1)


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
