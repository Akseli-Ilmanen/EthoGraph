"""Extracellular ephys waveform trace plot with smart downsampling.

Mirrors the AudioTracePlot / AudioTraceBuffer pattern for raw ephys data.
Rendering uses audian-style min/max envelope downsampling.

Two loading paths:
  - Known formats (.rhd, .rhs, .oebin, .edf, ...): Neo auto-detects
    dtype, gain, rate, and channel count from file headers.
  - Raw binary (.dat, .bin, .raw): user provides n_channels and
    sampling_rate; dtype defaults to int16.

All loaders expose the same interface consumed by EphysTraceBuffer:
    loader[start:stop]  ->  ndarray (samples x channels)
    len(loader)         ->  total sample count
    loader.rate         ->  sampling rate (Hz)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Loader protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EphysLoader(Protocol):
    rate: float

    def __len__(self) -> int: ...
    def __getitem__(self, key) -> NDArray: ...


# ---------------------------------------------------------------------------
# MemmapLoader – raw binary backend
# ---------------------------------------------------------------------------

class MemmapLoader:
    """Zero-copy loader for flat interleaved binary files.

    Parameters
    ----------
    path
        Path to the raw binary file.
    n_channels
        Number of interleaved channels.
    sampling_rate
        Sampling rate in Hz.
    dtype
        Sample dtype on disk (typically int16).
    gain
        Scaling factor applied on read (e.g. 0.195e-6 for Intan uV/bit).
    offset
        Byte offset to skip file header.
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
        raw = np.memmap(str(path), dtype=np.dtype(dtype), mode="r", offset=offset)
        total_samples = len(raw) // n_channels
        self._raw = raw[: total_samples * n_channels].reshape(total_samples, n_channels)
        self._n_channels = n_channels

    def __len__(self) -> int:
        return self._raw.shape[0]

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channel_names(self) -> list[str]:
        return [f"Ch {i}" for i in range(self._n_channels)]

    def __getitem__(self, key) -> NDArray[np.float64]:
        chunk = self._raw[key]
        if self._gain != 1.0:
            return chunk.astype(np.float64) * self._gain
        return chunk.astype(np.float64) if chunk.dtype != np.float64 else chunk


# ---------------------------------------------------------------------------
# GenericEphysLoader – auto-detecting unified loader
# ---------------------------------------------------------------------------

class GenericEphysLoader:
    """Auto-detecting ephys loader.

    For known formats Neo extracts all metadata from file headers.
    For raw binary the user must provide n_channels and sampling_rate.

    Parameters
    ----------
    path
        Path to ephys file (.rhd, .edf, .dat, etc.).
    n_channels
        Required only for raw binary formats.
    sampling_rate
        Required only for raw binary formats.
    dtype
        Raw binary dtype, ignored for known formats.
    gain
        Raw binary gain factor, ignored for known formats.
    stream_id
        Which Neo stream to load (e.g. "0" for amplifier, "1" for aux).
    """

    KNOWN_EXTENSIONS = {
        ".rhd": "IntanRawIO",
        ".rhs": "IntanRawIO",
        ".oebin": "OpenEphysBinaryRawIO",
        ".nwb": "NWBIO",
        ".ns5": "BlackrockRawIO",
        ".ns6": "BlackrockRawIO",
        ".nev": "BlackrockRawIO",
        ".abf": "AxonRawIO",
        ".edf": "EdfRawIO",
        ".bdf": "EdfRawIO",
        ".vhdr": "BrainVisionRawIO",
    }

    def __init__(
        self,
        path: str | Path,
        n_channels: int | None = None,
        sampling_rate: float | None = None,
        dtype: str = "int16",
        gain: float = 1.0,
        stream_id: str = "0",
    ):
        self.path = Path(path)
        self._reader = None
        self._loader: MemmapLoader | None = None

        ext = self.path.suffix.lower()
        rawio_name = self.KNOWN_EXTENSIONS.get(ext)

        if rawio_name:
            self._init_neo(rawio_name, stream_id)
        elif ext in (".dat", ".bin", ".raw"):
            if n_channels is None or sampling_rate is None:
                raise ValueError(
                    f"Raw binary '{ext}' requires n_channels and sampling_rate. "
                    "Use a format with headers (.rhd, .oebin, .edf, ...) "
                    "for automatic detection."
                )
            self._init_memmap(n_channels, sampling_rate, dtype, gain)
        else:
            supported = ", ".join(sorted(self.KNOWN_EXTENSIONS.keys()))
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {supported}, .dat, .bin, .raw"
            )

    # -- Neo backend --------------------------------------------------------

    def _init_neo(self, rawio_name: str, stream_id: str):
        import neo.rawio

        rawio_cls = getattr(neo.rawio, rawio_name)

        if rawio_name == "OpenEphysBinaryRawIO":
            self._reader = rawio_cls(dirname=str(self.path.parent))
        else:
            self._reader = rawio_cls(filename=str(self.path))

        self._reader.parse_header()
        self._resolve_stream(stream_id)

    def _resolve_stream(self, stream_id: str):
        streams = self._reader.header["signal_streams"]
        stream_ids = list(streams["id"])

        if stream_id not in stream_ids:
            stream_id = stream_ids[0]

        self._stream_idx = stream_ids.index(stream_id)
        self._n_samples = self._reader.get_signal_size(
            block_index=0, seg_index=0, stream_index=self._stream_idx,
        )

        channels = self._reader.header["signal_channels"]
        mask = channels["stream_id"] == stream_id
        self._n_channels = int(np.sum(mask))
        self.rate = float(channels[mask]["sampling_rate"][0])

    # -- Memmap backend -----------------------------------------------------

    def _init_memmap(
        self, n_channels: int, sampling_rate: float, dtype: str, gain: float,
    ):
        self._loader = MemmapLoader(
            self.path, n_channels, sampling_rate, np.dtype(dtype), gain=gain,
        )
        self._n_channels = n_channels
        self._n_samples = len(self._loader)
        self.rate = sampling_rate

    # -- Public interface ---------------------------------------------------

    def __len__(self) -> int:
        return self._n_samples

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channel_names(self) -> list[str]:
        if self._loader is not None:
            return self._loader.channel_names
        channels = self._reader.header["signal_channels"]
        streams = self._reader.header["signal_streams"]
        stream_id = streams["id"][self._stream_idx]
        mask = channels["stream_id"] == stream_id
        return list(channels[mask]["name"])

    @property
    def streams(self) -> dict | None:
        if self._reader is None:
            return None
        streams = self._reader.header["signal_streams"]
        channels = self._reader.header["signal_channels"]
        info = {}
        for sid, name in zip(streams["id"], streams["name"]):
            mask = channels["stream_id"] == sid
            info[sid] = {
                "name": name,
                "n_channels": int(np.sum(mask)),
                "rate": float(channels[mask]["sampling_rate"][0]),
            }
        return info

    def __getitem__(self, key) -> NDArray[np.float64]:
        if isinstance(key, slice):
            start, stop, _ = key.indices(self._n_samples)
        else:
            start, stop = key, key + 1

        if self._loader is not None:
            return self._loader[start:stop]

        raw = self._reader.get_analogsignal_chunk(
            block_index=0, seg_index=0,
            i_start=start, i_stop=stop,
            stream_index=self._stream_idx,
        )
        return self._reader.rescale_signal_raw_to_float(
            raw, dtype="float64", stream_index=self._stream_idx,
        )


# ---------------------------------------------------------------------------
# SharedEphysCache – thread-safe singleton for GenericEphysLoader instances
# ---------------------------------------------------------------------------

class SharedEphysCache:
    """Singleton cache for GenericEphysLoader instances.

    Multiple GUI components (waveform plot, spectrogram, heatmap, set_time,
    stream discovery) all need the same ephys loader. Without caching, each
    would re-parse file headers and allocate buffers independently.
    Cache key is (path, stream_id) so the same file can yield different
    loaders per stream.
    """

    _instances: dict[tuple[str, str], GenericEphysLoader] = {}
    _lock = threading.Lock()

    @classmethod
    def get_loader(
        cls,
        path: str | Path,
        stream_id: str = "0",
        n_channels: int | None = None,
        sampling_rate: float | None = None,
    ) -> GenericEphysLoader | None:
        path_str = str(path)
        key = (path_str, stream_id)

        with cls._lock:
            if key not in cls._instances:
                try:
                    cls._instances[key] = GenericEphysLoader(
                        path_str,
                        n_channels=n_channels,
                        sampling_rate=sampling_rate,
                        stream_id=stream_id,
                    )
                except (OSError, IOError, ValueError) as e:
                    print(f"Failed to load ephys file {path_str}: {e}")
                    return None
            return cls._instances[key]

    @classmethod
    def clear_cache(cls):
        with cls._lock:
            cls._instances.clear()


# ---------------------------------------------------------------------------
# EphysTraceBuffer – min/max envelope downsampling
# ---------------------------------------------------------------------------

class EphysTraceBuffer:
    """Audian-style min/max envelope downsampling for ephys traces."""

    def __init__(self, loader: EphysLoader | None = None, channel: int = 0):
        self.loader = loader
        self.channel = channel
        self.fs = loader.rate if loader else 1.0

    def set_loader(self, loader: EphysLoader, channel: int = 0):
        self.loader = loader
        self.channel = channel
        self.fs = loader.rate

    def get_trace_data(
        self,
        t0: float,
        t1: float,
        screen_width: int = 1920,
    ) -> tuple[NDArray, NDArray, int] | None:
        if self.loader is None:
            return None

        start = max(0, int(t0 * self.fs))
        stop = min(len(self.loader), int(t1 * self.fs) + 1)
        if stop <= start:
            return None

        step = max(1, (stop - start) // screen_width)

        if step > 1:
            aligned_start = (start // step) * step
            aligned_stop = min(len(self.loader), ((stop // step) + 1) * step)

            data = self._load_channel(aligned_start, aligned_stop)
            n_segments = len(data) // step
            if n_segments == 0:
                return None

            usable = n_segments * step
            data = data[:usable]

            segments = np.arange(0, usable, step)
            envelope = np.empty(2 * len(segments))
            np.minimum.reduceat(data, segments, out=envelope[0::2])
            np.maximum.reduceat(data, segments, out=envelope[1::2])

            half_step = step / 2
            times = np.arange(
                aligned_start,
                aligned_start + len(envelope) * half_step,
                half_step,
            ) / self.fs

            return times, envelope, step

        data = self._load_channel(start, stop)
        times = np.arange(start, start + len(data)) / self.fs
        return times, data, 1

    channel_spacing = 3.0

    def get_multichannel_trace_data(
        self,
        t0: float,
        t1: float,
        screen_width: int = 1920,
        channel_range: tuple[int, int] | None = None,
    ) -> tuple[NDArray, NDArray, int, int] | None:
        """Load channels with min/max envelope, z-normalize, and offset-stack.

        Parameters
        ----------
        channel_range
            ``(start, end)`` inclusive channel indices. ``None`` means all.

        Returns (times, data_2d, step, n_channels) where data_2d is
        shape (n_points, n_channels), already z-normalized and offset-stacked.
        Channel 0 at top (highest offset), last channel at bottom (offset 0).
        """
        if self.loader is None:
            return None

        total_ch = self.loader.n_channels if hasattr(self.loader, 'n_channels') else 1
        if total_ch <= 1:
            return None

        if channel_range is not None:
            ch_start = max(0, channel_range[0])
            ch_end = min(total_ch - 1, channel_range[1])
        else:
            ch_start, ch_end = 0, total_ch - 1
        n_ch = ch_end - ch_start + 1
        if n_ch <= 0:
            return None

        start = max(0, int(t0 * self.fs))
        stop = min(len(self.loader), int(t1 * self.fs) + 1)
        if stop <= start:
            return None

        step = max(1, (stop - start) // screen_width)

        if step > 1:
            aligned_start = (start // step) * step
            aligned_stop = min(len(self.loader), ((stop // step) + 1) * step)

            raw_all = self.loader[aligned_start:aligned_stop]
            if raw_all.ndim == 1:
                return None
            raw_all = raw_all[:, ch_start:ch_end + 1]

            n_segments = len(raw_all) // step
            if n_segments == 0:
                return None
            usable = n_segments * step
            raw_all = raw_all[:usable]

            segments = np.arange(0, usable, step)
            n_env = 2 * len(segments)
            envelope = np.empty((n_env, n_ch), dtype=np.float64)

            for ch in range(n_ch):
                col = raw_all[:, ch]
                np.minimum.reduceat(col, segments, out=envelope[0::2, ch])
                np.maximum.reduceat(col, segments, out=envelope[1::2, ch])

            half_step = step / 2
            times = np.arange(
                aligned_start, aligned_start + n_env * half_step, half_step,
            ) / self.fs
        else:
            raw_all = self.loader[start:stop]
            if raw_all.ndim == 1:
                return None
            raw_all = raw_all[:, ch_start:ch_end + 1]
            envelope = raw_all.astype(np.float64)
            times = np.arange(start, start + len(envelope)) / self.fs

        # z-normalize each channel
        for ch in range(n_ch):
            col = envelope[:, ch]
            mu = np.mean(col)
            std = np.std(col)
            if std > 0:
                envelope[:, ch] = (col - mu) / std
            else:
                envelope[:, ch] = 0.0

        # offset-stack: channel 0 at top
        for ch in range(n_ch):
            envelope[:, ch] += (n_ch - 1 - ch) * self.channel_spacing

        return times, envelope, step, n_ch

    def _load_channel(self, start: int, stop: int) -> NDArray[np.float64]:
        chunk = self.loader[start:stop]
        if chunk.ndim > 1:
            return chunk[:, min(self.channel, chunk.shape[1] - 1)]
        return chunk


# ---------------------------------------------------------------------------
# EphysTracePlot – BasePlot-based ephys waveform viewer
# ---------------------------------------------------------------------------

try:
    import pyqtgraph as pg
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    _HAS_QT = True
except ImportError:
    _HAS_QT = False


if _HAS_QT:
    from .plots_base import BasePlot

    # Default color cycle for multi-channel traces
    _MULTI_CH_COLORS = [
        '#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
        '#00BCD4', '#CDDC39', '#FF5722', '#607D8B', '#795548',
        '#3F51B5', '#009688', '#FFC107', '#F44336', '#8BC34A',
        '#673AB7',
    ]

    class EphysTracePlot(BasePlot):
        """Extracellular waveform viewer inheriting BasePlot for full GUI integration."""

        def __init__(self, app_state, parent=None):
            super().__init__(app_state, parent)

            self.setLabel('left', 'Amplitude')

            self.trace_item = pg.PlotDataItem(
                connect='all', antialias=False, skipFiniteCheck=True,
            )
            self.trace_item.setPen(pg.mkPen(color='#2196F3', width=1.5))
            self.addItem(self.trace_item)

            self.buffer = EphysTraceBuffer()
            self.current_range: tuple[float, float] | None = None

            self.label_items = []

            self._debounce_timer = QTimer()
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.setInterval(50)
            self._debounce_timer.timeout.connect(self._debounced_update)
            self._pending_range: tuple[float, float] | None = None

            self.vb.sigRangeChanged.connect(self._on_view_range_changed)

            # Multi-channel state
            self._multichannel = False
            self._multi_trace_items: list[pg.PlotDataItem] = []
            self._channel_range: tuple[int, int] | None = None

            self._ephys_offset: float = 0.0
            self._trial_duration: float | None = None

        def set_ephys_offset(self, offset: float, trial_duration: float | None = None):
            self._ephys_offset = offset
            self._trial_duration = trial_duration

        def set_loader(self, loader: EphysLoader, channel: int = 0):
            self.buffer.set_loader(loader, channel)

        def set_channel(self, channel: int):
            self.buffer.channel = channel
            if self.current_range:
                self.update_plot_content(*self.current_range)

        def set_channel_range(self, ch_start: int, ch_end: int):
            self._channel_range = (ch_start, ch_end)
            if self._multichannel and self.current_range:
                self.update_plot_content(*self.current_range)

        def set_multichannel(self, enabled: bool):
            if enabled == self._multichannel:
                return
            self._multichannel = enabled
            if enabled:
                self.trace_item.setData([], [])
                self.trace_item.hide()
            else:
                self._clear_multi_traces()
                self.trace_item.show()
                self._reset_y_axis_ticks()
            if self.current_range:
                self.update_plot_content(*self.current_range)

        def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
            if self.buffer.loader is None:
                return

            if t0 is None or t1 is None:
                xmin, xmax = self.get_current_xlim()
                t0, t1 = xmin, xmax

            # Clamp to trial boundaries
            t0 = max(0.0, t0)
            if self._trial_duration is not None:
                t1 = min(self._trial_duration, t1)

            if self._multichannel:
                self._update_multichannel(t0, t1)
            else:
                self._update_singlechannel(t0, t1)

            self.current_range = (t0, t1)

        def _update_singlechannel(self, t0: float, t1: float):
            try:
                screen_width = QApplication.primaryScreen().size().width()
            except Exception:
                screen_width = 1920

            # Query buffer in absolute ephys time, display in trial-local time
            result = self.buffer.get_trace_data(
                t0 + self._ephys_offset, t1 + self._ephys_offset, screen_width,
            )
            if result is None:
                return

            times, amplitudes, step = result
            times = times - self._ephys_offset
            self.trace_item.setData(times, amplitudes)

            if step > 1:
                self.trace_item.setPen(pg.mkPen(color='#2196F3', width=1.0))
                self.trace_item.setSymbol(None)
            else:
                self.trace_item.setPen(pg.mkPen(color='#2196F3', width=2.0))
                if len(times) < 200:
                    self.trace_item.setSymbol('o')
                    self.trace_item.setSymbolSize(4)
                    self.trace_item.setSymbolBrush('#2196F3')
                else:
                    self.trace_item.setSymbol(None)

        def _update_multichannel(self, t0: float, t1: float):
            try:
                screen_width = QApplication.primaryScreen().size().width()
            except Exception:
                screen_width = 1920

            # Query buffer in absolute ephys time, display in trial-local time
            result = self.buffer.get_multichannel_trace_data(
                t0 + self._ephys_offset, t1 + self._ephys_offset,
                screen_width, channel_range=self._channel_range,
            )
            if result is None:
                return

            times, data_2d, step, n_ch = result
            times = times - self._ephys_offset

            # Grow or shrink trace item pool
            while len(self._multi_trace_items) < n_ch:
                item = pg.PlotDataItem(connect='all', antialias=False, skipFiniteCheck=True)
                self.addItem(item)
                self._multi_trace_items.append(item)
            while len(self._multi_trace_items) > n_ch:
                item = self._multi_trace_items.pop()
                self.removeItem(item)

            pen_width = 1.0 if step > 1 else 1.5
            for ch in range(n_ch):
                color = _MULTI_CH_COLORS[ch % len(_MULTI_CH_COLORS)]
                self._multi_trace_items[ch].setData(times, data_2d[:, ch])
                self._multi_trace_items[ch].setPen(pg.mkPen(color=color, width=pen_width))
                self._multi_trace_items[ch].setSymbol(None)

            # Set y-axis ticks at channel offsets
            ch_start = self._channel_range[0] if self._channel_range else 0
            channel_names = self._get_channel_names(n_ch, ch_start)
            spacing = self.buffer.channel_spacing
            ticks = [((n_ch - 1 - i) * spacing, channel_names[i]) for i in range(n_ch)]

            left_axis = self.plot_item.getAxis('left')
            left_axis.setTicks([ticks])
            self.setLabel('left', '')

            margin = spacing * 0.5
            self.plot_item.setYRange(-margin, (n_ch - 1) * spacing + margin, padding=0)

        def _get_channel_names(self, n_ch: int, ch_start: int = 0) -> list[str]:
            loader = self.buffer.loader
            if hasattr(loader, 'channel_names'):
                names = loader.channel_names
                if len(names) >= ch_start + n_ch:
                    return names[ch_start:ch_start + n_ch]
            return [f"Ch {ch_start + i}" for i in range(n_ch)]

        def _clear_multi_traces(self):
            for item in self._multi_trace_items:
                self.removeItem(item)
            self._multi_trace_items.clear()

        def _reset_y_axis_ticks(self):
            left_axis = self.plot_item.getAxis('left')
            left_axis.setTicks(None)
            self.setLabel('left', 'Amplitude')

        def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
            if self._multichannel:
                return
            if ymin is not None and ymax is not None:
                self.plot_item.setYRange(ymin, ymax)

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
            self.update_plot_content(t0, t1)

        def update_time_marker_and_window(self, frame_number: int):
            super().update_time_marker_and_window(frame_number)

            if hasattr(self.app_state, 'ds') and self.app_state.ds is not None:
                t0, t1 = self.get_current_xlim()
                current_time = frame_number / self.app_state.ds.fps

                if self.current_range is None or current_time < self.current_range[0] or current_time > self.current_range[1]:
                    self.update_plot_content(t0, t1)
