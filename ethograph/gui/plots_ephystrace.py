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


import numpy as np
import threading
import pyqtgraph as pg
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from numpy.typing import NDArray
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication


from .plots_base import BasePlot


def _nice_round(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** int(np.floor(np.log10(value)))
    normalized = value / magnitude
    if normalized < 1.5:
        return magnitude
    elif normalized < 3.5:
        return 2 * magnitude
    elif normalized < 7.5:
        return 5 * magnitude
    return 10 * magnitude


_UNIT_TO_VOLTS: dict[str, float] = {
    "V": 1.0,
    "mV": 1e-3,
    "uV": 1e-6,
    "\u00b5V": 1e-6,
}


def _format_voltage_bar(raw_value: float, loader_units: str) -> tuple[float, str]:
    factor = _UNIT_TO_VOLTS.get(loader_units)
    if factor is None:
        bar = _nice_round(raw_value)
        return bar, f"{bar:.4g} {loader_units}"

    value_in_v = raw_value * factor
    abs_v = abs(value_in_v)
    if abs_v < 1e-4:
        display_val = value_in_v * 1e6
        display_unit = "\u00b5V"
    elif abs_v < 0.1:
        display_val = value_in_v * 1e3
        display_unit = "mV"
    else:
        display_val = value_in_v
        display_unit = "V"

    nice_display = _nice_round(abs(display_val))
    if display_val < 0:
        nice_display = -nice_display

    if display_unit == "\u00b5V":
        bar_in_v = nice_display * 1e-6
    elif display_unit == "mV":
        bar_in_v = nice_display * 1e-3
    else:
        bar_in_v = nice_display

    bar_in_loader = bar_in_v / factor
    return bar_in_loader, f"{nice_display:g} {display_unit}"


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

    @property
    def units(self) -> str:
        return "a.u."

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
    def units(self) -> str:
        if self._loader is not None:
            return self._loader.units
        channels = self._reader.header["signal_channels"]
        streams = self._reader.header["signal_streams"]
        stream_id = streams["id"][self._stream_idx]
        mask = channels["stream_id"] == stream_id
        unit_str = str(channels[mask]["units"][0])
        return unit_str if unit_str else "a.u."

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

    def __init__(self, loader: EphysLoader | None = None, channel: int = 0):
        self.loader = loader
        self.channel = channel
        self.ephys_sr: float | None = None
        self._preproc_flags: dict = {}
        self._segment_offset: float = 0.0
        self._segment_duration: float | None = None
        self._cache: NDArray | None = None
        self._cache_start: int = 0
        self._cache_stop: int = 0
        self._cache_mean: NDArray | None = None
        self._cache_std: NDArray | None = None
        self.channel_spacing = 3.0
        self.display_gain: int = 0
        self.autocenter: bool = False

    def set_loader(self, loader: EphysLoader, channel: int = 0):
        self.loader = loader
        self.channel = channel
        self.ephys_sr = loader.rate
        self._invalidate_cache()

    def set_trial_segment(self, offset: float, duration: float | None):
        self._segment_offset = offset
        self._segment_duration = duration
        self._invalidate_cache()

    def set_preprocessing(self, flags: dict):
        self._preproc_flags = flags
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._cache = None
        self._cache_mean = None
        self._cache_std = None

    def _segment_bounds(self) -> tuple[int, int]:
        start = max(0, int(self._segment_offset * self.ephys_sr))
        if self._segment_duration is not None:
            stop = min(len(self.loader),
                       int((self._segment_offset + self._segment_duration) * self.ephys_sr) + 1)
        else:
            stop = len(self.loader)
        return start, stop

    def _build_cache(self):
        if self._cache is not None or self.loader is None:
            return
        seg_start, seg_stop = self._segment_bounds()
        if seg_stop <= seg_start:
            return

        raw = self.loader[seg_start:seg_stop]
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]

        if self._any_preprocessing():
            data = _apply_preprocessing(raw, self.ephys_sr, self._preproc_flags)
        else:
            data = raw.astype(np.float64) if raw.dtype != np.float64 else raw

        self._cache = data
        self._cache_start = seg_start
        self._cache_stop = seg_stop
        self._cache_mean = data.mean(axis=0)
        per_ch_std = data.std(axis=0)
        per_ch_std[per_ch_std == 0] = 1.0
        median_std = np.median(per_ch_std)
        self._cache_std = np.full_like(per_ch_std, median_std)

    def _get_data(self, start: int, stop: int) -> NDArray | None:
        self._build_cache()
        if self._cache is None:
            return None
        local_start = max(0, start - self._cache_start)
        local_stop = min(self._cache.shape[0], stop - self._cache_start)
        if local_stop <= local_start:
            return None
        return self._cache[local_start:local_stop]

    def _any_preprocessing(self) -> bool:
        return any(v for k, v in self._preproc_flags.items() if isinstance(v, bool) and v)

    # -- single channel -----------------------------------------------------

    def get_trace_data(
        self, t0: float, t1: float, screen_width: int = 1920,
    ) -> tuple[NDArray, NDArray, int] | None:
        if self.loader is None:
            return None

        start = max(0, int(t0 * self.ephys_sr))
        stop = min(len(self.loader), int(t1 * self.ephys_sr) + 1)
        if stop <= start:
            return None

        data_all = self._get_data(start, stop)
        if data_all is None:
            return None

        ch = min(self.channel, data_all.shape[1] - 1)
        data = data_all[:, ch].copy()

        if self.display_gain != 0:
            data *= 0.75 ** (-self.display_gain)

        step = max(1, (stop - start) // screen_width)
        if step > 1:
            return _envelope_downsample(data, start, step, self.ephys_sr)

        times = np.arange(start, start + len(data)) / self.ephys_sr
        return times, data, 1

    # -- multi channel ------------------------------------------------------

    def get_multichannel_trace_data(
        self, t0: float, t1: float, screen_width: int = 1920,
        channel_range: tuple[int, int] | None = None,
    ) -> tuple[NDArray, NDArray, int, int] | None:
        if self.loader is None:
            return None

        total_ch = self.loader.n_channels if hasattr(self.loader, 'n_channels') else 1
        if total_ch <= 1:
            return None

        ch_start, ch_end = (channel_range or (0, total_ch - 1))
        ch_start = max(0, ch_start)
        ch_end = min(total_ch - 1, ch_end)
        n_ch = ch_end - ch_start + 1
        if n_ch <= 0:
            return None

        start = max(0, int(t0 * self.ephys_sr))
        stop = min(len(self.loader), int(t1 * self.ephys_sr) + 1)
        if stop <= start:
            return None

        data_all = self._get_data(start, stop)
        if data_all is None or data_all.ndim < 2:
            return None
        data_all = data_all[:, ch_start:ch_end + 1]

        step = max(1, (stop - start) // screen_width)

        if step > 1:
            n_segments = len(data_all) // step
            if n_segments == 0:
                return None
            usable = n_segments * step
            data_all = data_all[:usable]

            segments = np.arange(0, usable, step)
            n_env = 2 * len(segments)
            display = np.empty((n_env, n_ch))
            for ch in range(n_ch):
                col = data_all[:, ch]
                np.minimum.reduceat(col, segments, out=display[0::2, ch])
                np.maximum.reduceat(col, segments, out=display[1::2, ch])

            aligned_start = (start // step) * step
            half_step = step / 2
            times = np.arange(aligned_start, aligned_start + n_env * half_step, half_step) / self.ephys_sr
        else:
            display = data_all.copy()
            times = np.arange(start, start + len(display)) / self.ephys_sr

        ch_std = self._cache_std[ch_start:ch_end + 1]
        if self.autocenter:
            display = (display - display.mean(axis=0)) / ch_std
        else:
            display = (display - self._cache_mean[ch_start:ch_end + 1]) / ch_std

        gain_factor = 0.75 ** (-self.display_gain)
        display *= gain_factor

        for ch in range(n_ch):
            display[:, ch] += (n_ch - 1 - ch) * self.channel_spacing

        return times, display, step, n_ch


# -- preprocessing (pure function, no state) --------------------------------

def _apply_preprocessing(data: NDArray, sr: float, flags: dict) -> NDArray:
    from ethograph.features.filter import sosfilter

    data = data.astype(np.float64, copy=True)

    # if flags.get("subtract_mean"):
    #     data -= data.mean(axis=0, keepdims=True)

    if flags.get("car") and data.shape[1] > 8:
        data -= np.median(data, axis=1, keepdims=True)

    if flags.get("temporal_filter"):
        cutoff = flags.get("hp_cutoff", 300.0)
        data = sosfilter(data, sr, cutoff, mode='hp', order=3)

    if flags.get("whitening"):
        CC = (data.T @ data) / data.shape[0]
        Wrot = _whitening_from_covariance(CC)
        data = data @ Wrot.T

    return data


def _whitening_from_covariance(CC: np.ndarray) -> np.ndarray:
    E, D, Vt = np.linalg.svd(CC)
    return (E / np.sqrt(D + 1e-6)) @ E.T


def _envelope_downsample(
    data: NDArray, start: int, step: int, sr: float
) -> tuple[NDArray, NDArray, int]:
    aligned_start = (start // step) * step
    n_segments = len(data) // step
    usable = n_segments * step
    data = data[:usable]

    segments = np.arange(0, usable, step)
    envelope = np.empty(2 * len(segments))
    np.minimum.reduceat(data, segments, out=envelope[0::2])
    np.maximum.reduceat(data, segments, out=envelope[1::2])

    half_step = step / 2
    times = np.arange(aligned_start, aligned_start + len(envelope) * half_step, half_step) / sr
    return times, envelope, step

# ---------------------------------------------------------------------------
# EphysTracePlot – BasePlot-based ephys waveform viewer
# ---------------------------------------------------------------------------


_NEUROSCOPE_BG = '#000000'
_NEUROSCOPE_TRACE = '#3399FF'
_NEUROSCOPE_AXIS = '#AAAAAA'

class EphysTracePlot(BasePlot):
    """Extracellular waveform viewer inheriting BasePlot for full GUI integration."""

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        self.setBackground(_NEUROSCOPE_BG)
        for axis_name in ('left', 'bottom'):
            axis = self.plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen(_NEUROSCOPE_AXIS))
            axis.setTextPen(pg.mkPen(_NEUROSCOPE_AXIS))
        self.time_marker.setPen(pg.mkPen('#FF4444', width=2))

        self.setLabel('left', 'Amplitude')

        self.trace_item = pg.PlotDataItem(
            connect='all', antialias=False, skipFiniteCheck=True,
        )
        self.trace_item.setPen(pg.mkPen(color=_NEUROSCOPE_TRACE, width=1.5))
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

        # Calibration scale bars
        self._scale_v_line: pg.PlotDataItem | None = None
        self._scale_h_line: pg.PlotDataItem | None = None
        self._scale_v_text: pg.TextItem | None = None
        self._scale_h_text: pg.TextItem | None = None

        self.setToolTip("Double-click or Ctrl+A to autoscale")
        self.scene().sigMouseClicked.connect(self._handle_double_click)

    def set_ephys_offset(self, offset: float, trial_duration: float | None = None):
        self._ephys_offset = offset
        self._trial_duration = trial_duration
        self.buffer.set_trial_segment(offset, trial_duration)

    def set_loader(self, loader: EphysLoader, channel: int = 0):
        self.buffer.set_loader(loader, channel)
        self._update_amplitude_label()

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
            self.trace_item.setPen(pg.mkPen(color=_NEUROSCOPE_TRACE, width=1.0))
            self.trace_item.setSymbol(None)
        else:
            self.trace_item.setPen(pg.mkPen(color=_NEUROSCOPE_TRACE, width=2.0))
            if len(times) < 200:
                self.trace_item.setSymbol('o')
                self.trace_item.setSymbolSize(4)
                self.trace_item.setSymbolBrush(_NEUROSCOPE_TRACE)
            else:
                self.trace_item.setSymbol(None)
                
        self._update_scale_bars(times)

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
        trace_pen = pg.mkPen(color=_NEUROSCOPE_TRACE, width=pen_width)
        for ch in range(n_ch):
            self._multi_trace_items[ch].setData(times, data_2d[:, ch])
            self._multi_trace_items[ch].setPen(trace_pen)
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

        self._update_scale_bars(times)

    def _update_scale_bars(self, times: NDArray):
        self._clear_scale_bars()

        if len(times) < 2 or self.buffer.ephys_sr is None:
            return

        t0, t1 = float(times[0]), float(times[-1])
        time_window = t1 - t0
        if time_window <= 0:
            return

        # -- Time bar: ~1/20 of window, rounded to a nice value --
        raw_time_ms = (time_window / 20.0) * 1000.0
        time_bar_ms = _nice_round(raw_time_ms)
        time_bar_s = time_bar_ms / 1000.0

        # -- Voltage bar: fixed 0.2 mV (NeuroScope convention) --
        # Convert 0.2 mV to loader units, then to display units
        scale_voltage = 0.2  # mV
        loader_units = "a.u."
        if self.buffer.loader is not None and hasattr(self.buffer.loader, 'units'):
            loader_units = self.buffer.loader.units

        factor = _UNIT_TO_VOLTS.get(loader_units)
        if factor is not None:
            voltage_in_loader = (scale_voltage * 1e-3) / factor  # 0.2 mV -> loader units
        else:
            voltage_in_loader = scale_voltage

        gain_factor = 0.75 ** (-self.buffer.display_gain)
        if self._multichannel:
            # Multichannel: display is z-normalized (divided by median_std), then scaled by gain
            median_std = self.buffer._cache_std[0] if self.buffer._cache_std is not None else 1.0
            voltage_per_display_unit = median_std / gain_factor
        else:
            # Single channel: display is raw loader units, only gain applied
            voltage_per_display_unit = 1.0 / gain_factor
        voltage_bar_display = voltage_in_loader / voltage_per_display_unit

        v_label = "0.2 mV"

        # Position: bottom-right corner
        y_range = self.plot_item.getViewBox().viewRange()[1]
        y_span = y_range[1] - y_range[0]
        x_anchor = t1 - time_window * 0.03
        y_anchor = y_range[0] + y_span * 0.05

        # Vertical bar (voltage)
        v_x = [x_anchor, x_anchor]
        v_y = [y_anchor, y_anchor + voltage_bar_display]
        self._scale_v_line = pg.PlotDataItem(
            v_x, v_y, pen=pg.mkPen('#FFFFFF', width=2),
        )
        self._scale_v_line.setZValue(900)
        self.plot_item.addItem(self._scale_v_line)

        self._scale_v_text = pg.TextItem(v_label, color='#FFFFFF', anchor=(1.0, 0.5))
        self._scale_v_text.setPos(x_anchor - time_window * 0.005, y_anchor + voltage_bar_display / 2)
        self._scale_v_text.setZValue(900)
        self.plot_item.addItem(self._scale_v_text)

        # Horizontal bar (time)
        h_x = [x_anchor - time_bar_s, x_anchor]
        h_y = [y_anchor, y_anchor]
        self._scale_h_line = pg.PlotDataItem(
            h_x, h_y, pen=pg.mkPen('#FFFFFF', width=2),
        )
        self._scale_h_line.setZValue(900)
        self.plot_item.addItem(self._scale_h_line)

        if time_bar_ms >= 1000:
            h_label = f"{time_bar_ms / 1000:.1f} s"
        else:
            h_label = f"{time_bar_ms:.0f} ms"
        self._scale_h_text = pg.TextItem(h_label, color='#FFFFFF', anchor=(0.5, 1.0))
        self._scale_h_text.setPos(x_anchor - time_bar_s / 2, y_anchor - (y_range[1] - y_range[0]) * 0.01)
        self._scale_h_text.setZValue(900)
        self.plot_item.addItem(self._scale_h_text)

    def _clear_scale_bars(self):
        for attr in ('_scale_v_line', '_scale_h_line', '_scale_v_text', '_scale_h_text'):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    self.plot_item.removeItem(item)
                except (RuntimeError, ValueError):
                    pass
                setattr(self, attr, None)

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
        self._update_amplitude_label()

    def _update_amplitude_label(self):
        units = ''
        if self.buffer.loader is not None and hasattr(self.buffer.loader, 'units'):
            units = self.buffer.loader.units
        if units and units != 'a.u.':
            self.setLabel('left', 'Amplitude', units=units)
        else:
            self.setLabel('left', 'Amplitude')

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        if self._multichannel:
            return
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _handle_double_click(self, event):
        from qtpy.QtCore import Qt
        if event.double() and event.button() == Qt.LeftButton:
            self.autoscale()

    def autoscale(self):
        if self._multichannel:
            n_ch = len(self._multi_trace_items)
            if n_ch > 0:
                spacing = self.buffer.channel_spacing
                margin = spacing * 0.5
                self.plot_item.setYRange(-margin, (n_ch - 1) * spacing + margin, padding=0)
        else:
            self.plot_item.enableAutoRange(axis='y')

        if self._trial_duration is not None:
            self.plot_item.setXRange(0, self._trial_duration, padding=0.02)
        else:
            self.plot_item.enableAutoRange(axis='x')

        if self.current_range:
            self.update_plot_content(*self.get_current_xlim())

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
