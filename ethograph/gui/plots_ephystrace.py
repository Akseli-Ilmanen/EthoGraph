"""Extracellular ephys waveform trace plot with smart downsampling.

Mirrors the AudioTracePlot / AudioTraceBuffer pattern but loads raw .dat
binary files (e.g. Intan, Open Ephys, SpikeGLX) via numpy.memmap or Neo's
RawBinarySignalIO.  The rendering uses the same audian-style min/max envelope
downsampling that AudioTraceBuffer uses.

Supports two loader backends:
  1. numpy.memmap  – zero-overhead, OS-managed paging (default for .dat/.bin)
  2. neo.io.RawBinarySignalIO – if you want Neo's object model / unit handling

Both expose the same minimal interface consumed by EphysTraceBuffer:
    loader[start:stop]  →  ndarray (samples × channels)
    len(loader)         →  total sample count
    loader.rate         →  sampling rate (Hz)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Loader protocol – anything that satisfies this can drive the plot
# ---------------------------------------------------------------------------

@runtime_checkable
class EphysLoader(Protocol):
    rate: float

    def __len__(self) -> int: ...
    def __getitem__(self, key) -> NDArray: ...


# ---------------------------------------------------------------------------
# Backend 1: numpy.memmap (preferred for raw .dat / .bin)
# ---------------------------------------------------------------------------

class MemmapLoader:
    """Wraps a flat binary file as a (samples × channels) memmap array.

    Parameters
    ----------
    path
        Path to the raw binary file.
    n_channels
        Number of interleaved channels.
    sampling_rate
        Sampling rate in Hz.
    dtype
        Sample dtype on disk (typically int16 for most acquisition systems).
    gain
        Scaling factor to convert raw integers → volts (µV/bit × 1e-6).
        Set to 1.0 if data is already in float / physical units.
    offset
        Byte offset to skip file header, if any.
    """

    def __init__(
        self,
        path: str | Path,
        n_channels: int,
        sampling_rate: float,
        dtype: np.dtype | str = np.int16,
        gain: float = 1.0,
        offset: int = 0,
    ):
        self.rate = sampling_rate
        self._gain = gain
        self._raw = np.memmap(
            str(path), dtype=np.dtype(dtype), mode="r", offset=offset,
        )
        self._n_channels = n_channels
        total_samples = len(self._raw) // n_channels
        self._raw = self._raw[: total_samples * n_channels].reshape(
            total_samples, n_channels
        )

    def __len__(self) -> int:
        return self._raw.shape[0]

    @property
    def n_channels(self) -> int:
        return self._n_channels

    def __getitem__(self, key) -> NDArray[np.float64]:
        chunk = self._raw[key]
        if self._gain != 1.0:
            return chunk.astype(np.float64) * self._gain
        return chunk.astype(np.float64) if chunk.dtype != np.float64 else chunk


# ---------------------------------------------------------------------------
# Backend 2: Neo RawBinarySignalIO (lazy, with units)
# ---------------------------------------------------------------------------

class NeoRawLoader:
    """Wraps Neo's RawBinarySignalIO behind the same slice interface.

    Uses neo.rawio for direct chunk access without building the full
    Neo Block/Segment hierarchy on every slice.
    """

    def __init__(
        self,
        path: str | Path,
        n_channels: int,
        sampling_rate: float,
        dtype: str = "int16",
        signal_gain: float = 1.0,
        signal_offset: float = 0.0,
        bytes_offset: int = 0,
    ):
        from neo.rawio import RawBinarySignalRawIO

        self._reader = RawBinarySignalRawIO(
            filename=str(path),
            dtype=dtype,
            nb_channel=n_channels,
            sampling_rate=sampling_rate,
            signal_gain=signal_gain,
            signal_offset=signal_offset,
            bytesoffset=bytes_offset,
        )
        self._reader.parse_header()
        self.rate = sampling_rate
        self._n_channels = n_channels
        self._n_samples = self._reader.get_signal_size(
            block_index=0, seg_index=0,
        )

    def __len__(self) -> int:
        return self._n_samples

    @property
    def n_channels(self) -> int:
        return self._n_channels

    def __getitem__(self, key) -> NDArray[np.float64]:
        if isinstance(key, slice):
            start, stop, _ = key.indices(self._n_samples)
        else:
            start, stop = key, key + 1

        raw = self._reader.get_analogsignal_chunk(
            block_index=0,
            seg_index=0,
            i_start=start,
            i_stop=stop,
        )
        return self._reader.rescale_signal_raw_to_float(
            raw, dtype="float64",
        )


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def create_ephys_loader(
    path: str | Path,
    n_channels: int,
    sampling_rate: float,
    dtype: str = "int16",
    gain: float = 1.0,
    offset: int = 0,
    backend: str = "memmap",
) -> EphysLoader:
    if backend == "neo":
        return NeoRawLoader(
            path, n_channels, sampling_rate, dtype, signal_gain=gain, bytes_offset=offset,
        )
    return MemmapLoader(
        path, n_channels, sampling_rate, np.dtype(dtype), gain=gain, offset=offset,
    )


# ---------------------------------------------------------------------------
# EphysTraceBuffer – drop-in analogue of AudioTraceBuffer
# ---------------------------------------------------------------------------

class EphysTraceBuffer:
    """Buffer for ephys trace data with audian-style min/max downsampling.

    This mirrors AudioTraceBuffer exactly but takes an EphysLoader instead
    of an audioio.AudioLoader.  The downsampling algorithm (reduceat min/max
    envelope) is identical.
    """

    def __init__(self, loader: EphysLoader, channel: int = 0):
        self.loader = loader
        self.channel = channel
        self.fs = loader.rate

    def get_trace_data(
        self,
        t0: float,
        t1: float,
        screen_width: int = 1920,
    ) -> tuple[NDArray, NDArray, int] | None:
        start = max(0, int(t0 * self.fs))
        stop = min(len(self.loader), int(t1 * self.fs) + 1)
        if stop <= start:
            return None

        step = max(1, (stop - start) // screen_width)

        if step > 1:
            aligned_start = (start // step) * step
            aligned_stop = min(len(self.loader), ((stop // step) + 1) * step)

            audio_data = self._load_channel(aligned_start, aligned_stop)
            n_segments = len(audio_data) // step
            if n_segments == 0:
                return None

            usable_len = n_segments * step
            audio_data = audio_data[:usable_len]

            segments = np.arange(0, usable_len, step)
            plot_data = np.empty(2 * len(segments))
            np.minimum.reduceat(audio_data, segments, out=plot_data[0::2])
            np.maximum.reduceat(audio_data, segments, out=plot_data[1::2])

            half_step = step / 2
            plot_time = np.arange(
                aligned_start,
                aligned_start + len(plot_data) * half_step,
                half_step,
            ) / self.fs

            return plot_time, plot_data, step

        audio_data = self._load_channel(start, stop)
        plot_time = np.arange(start, start + len(audio_data)) / self.fs
        return plot_time, audio_data, 1

    def _load_channel(self, start: int, stop: int) -> NDArray[np.float64]:
        chunk = self.loader[start:stop]
        if chunk.ndim > 1:
            ch = min(self.channel, chunk.shape[1] - 1)
            return chunk[:, ch]
        return chunk


# ---------------------------------------------------------------------------
# PyQtGraph plot widget (mirrors AudioTracePlot)
# ---------------------------------------------------------------------------

try:
    import pyqtgraph as pg
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    HAS_QT = True
except ImportError:
    HAS_QT = False


if HAS_QT:

    class EphysTracePlot(pg.PlotWidget):
        """Extracellular waveform viewer – drop-in replacement for AudioTracePlot.

        Usage
        -----
        loader = MemmapLoader("recording.dat", n_channels=32, sampling_rate=30000)
        plot = EphysTracePlot(loader, channel=5)
        plot.show()
        plot.set_xrange(10.0, 12.0)  # view 2 seconds starting at t=10s
        """

        def __init__(
            self,
            loader: EphysLoader,
            channel: int = 0,
            parent=None,
        ):
            super().__init__(parent=parent)

            self.buffer = EphysTraceBuffer(loader, channel)
            self.current_range: tuple[float, float] | None = None

            self.setLabel("left", "Amplitude")
            self.setLabel("bottom", "Time", units="s")

            self.trace_item = pg.PlotDataItem(
                connect="all", antialias=False, skipFiniteCheck=True,
            )
            self.trace_item.setPen(pg.mkPen(color="#2196F3", width=1.5))
            self.addItem(self.trace_item)

            self._debounce_timer = QTimer()
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.setInterval(50)
            self._debounce_timer.timeout.connect(self._debounced_update)
            self._pending_range: tuple[float, float] | None = None

            self.sigXRangeChanged.connect(self._on_range_changed)

        def set_xrange(self, t0: float, t1: float):
            self.setXRange(t0, t1, padding=0)
            self.update_trace(t0, t1)

        def set_channel(self, channel: int):
            self.buffer.channel = channel
            if self.current_range:
                self.update_trace(*self.current_range)

        def update_trace(self, t0: float, t1: float):
            try:
                screen_width = QApplication.primaryScreen().size().width()
            except Exception:
                screen_width = 1920

            result = self.buffer.get_trace_data(t0, t1, screen_width)
            if result is None:
                return

            times, amplitudes, step = result
            self.trace_item.setData(times, amplitudes)

            if step > 1:
                self.trace_item.setPen(pg.mkPen(color="#2196F3", width=1.0))
                self.trace_item.setSymbol(None)
            else:
                self.trace_item.setPen(pg.mkPen(color="#2196F3", width=2.0))
                if len(times) < 200:
                    self.trace_item.setSymbol("o")
                    self.trace_item.setSymbolSize(4)
                    self.trace_item.setSymbolBrush("#2196F3")
                else:
                    self.trace_item.setSymbol(None)

            self.current_range = (t0, t1)

        def _on_range_changed(self):
            vr = self.viewRange()
            self._pending_range = (vr[0][0], vr[0][1])
            self._debounce_timer.start()

        def _debounced_update(self):
            if self._pending_range is None:
                return
            t0, t1 = self._pending_range
            self._pending_range = None
            self.update_trace(t0, t1)
