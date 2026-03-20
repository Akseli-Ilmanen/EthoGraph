"""Extracellular ephys waveform trace plot with smart downsampling.

Mirrors the AudioTracePlot / AudioTraceBuffer pattern for raw ephys data.
Rendering uses audian-style min/max envelope downsampling.

Three loading paths:
  - Known formats (.rhd, .rhs, .oebin, .edf, ...): Neo auto-detects
    dtype, gain, rate, and channel count from file headers.
  - NWB files (.nwb): pynwb with lazy HDF5 access (Neo lacks NWBRawIO).
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
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .plots_timeseriessource import TimeseriesSource
from numpy.typing import NDArray
from qtpy.QtCore import QEvent, Qt, Signal

import warnings
from phylib.io.traces import get_ephys_reader

from ethograph.utils.validation import EPHYS_EXNTENSIONS_RAW

from .app_constants import BUFFER_COVERAGE_MARGIN, DEFAULT_BUFFER_MULTIPLIER_EPHYS, EPHYSTRACE_DEBOUNCE_MS
from .plots_base import BasePlot, ThrottleDebounce
from .video_manager import is_url


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


def select_traces(traces, interval, sample_rate):
    """Load traces in an interval (in seconds) with median subtraction.

    Mirrors phy's ``select_traces`` for simple time-based indexing.
    """
    start, end = interval
    i, j = int(round(sample_rate * start)), int(round(sample_rate * end))
    i = max(0, i)
    if hasattr(traces, '__len__'):
        j = min(j, len(traces))
    data = traces[i:j]
    if hasattr(data, 'astype'):
        data = np.asarray(data, dtype=np.float32)
    data = data - np.median(data, axis=0)
    return data


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





class RemoteNWBLoader:
    def __init__(self, url: str, electrical_series_name: str | None = None):
        import h5py
        import remfile
        import pynwb

        self._rf = remfile.File(url)
        self._h5 = h5py.File(self._rf, "r")
        self._io = pynwb.NWBHDF5IO(file=self._h5, load_namespaces=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*manufacturer.*deprecated", category=DeprecationWarning)
            nwb = self._io.read()

        es = self._resolve_electrical_series(nwb, electrical_series_name)
        self._data = es.data
        self._conversion = float(es.conversion) if es.conversion else 1.0
        self.rate = float(es.rate)
        self.starting_time = float(es.starting_time) if es.starting_time else 0.0
        self._n_channels = self._data.shape[1] if self._data.ndim > 1 else 1

        electrodes = es.electrodes
        if electrodes is not None and hasattr(electrodes, "table"):
            table = electrodes.table
            indices = electrodes.data[:]
            if "label" in table.colnames:
                self._channel_names = [str(table["label"][i]) for i in indices]
            else:
                self._channel_names = [f"Ch {i}" for i in indices]
        else:
            self._channel_names = [f"Ch {i}" for i in range(self._n_channels)]
        self._units = "V"

    def _resolve_electrical_series(self, nwb, name: str | None):
        import pynwb

        if name:
            return nwb.acquisition[name]
        es = next(
            (v for v in nwb.acquisition.values()
             if isinstance(v, pynwb.ecephys.ElectricalSeries)),
            None,
        )
        if es is None:
            raise ValueError(
                f"No ElectricalSeries found. "
                f"Available acquisition keys: {list(nwb.acquisition.keys())}"
            )
        return es

    def __len__(self) -> int:
        return self._data.shape[0]

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channel_names(self) -> list[str]:
        return self._channel_names

    @property
    def units(self) -> str:
        return self._units

    def __getitem__(self, key) -> NDArray[np.float64]:
        chunk = self._data[key]
        if chunk.ndim == 1 and self._n_channels == 1:
            chunk = chunk[:, np.newaxis]
        return chunk.astype(np.float64) * self._conversion

    def __del__(self):
        for resource in (self._io, self._h5, self._rf):
            try:
                resource.close()
            except Exception:
                pass



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

    KNOWN_EXTENSIONS: dict[str, str] = {
        ".nwb": "NWBIO",
        ".rhd": "IntanRawIO",
        ".rhs": "IntanRawIO",
        ".oebin": "OpenEphysBinaryRawIO",
        ".openephys": "OpenEphysRawIO",
        ".continuous": "OpenEphysRawIO",
        ".spikes": "OpenEphysRawIO",
        ".events": "OpenEphysRawIO",
        ".ns1": "BlackrockRawIO", ".ns2": "BlackrockRawIO", ".ns3": "BlackrockRawIO",
        ".ns4": "BlackrockRawIO", ".ns5": "BlackrockRawIO", ".ns6": "BlackrockRawIO",
        ".nev": "BlackrockRawIO", ".sif": "BlackrockRawIO", ".ccf": "BlackrockRawIO",
        ".abf": "AxonRawIO",
        ".axgx": "AxographRawIO", ".axgd": "AxographRawIO",
        ".edf": "EDFRawIO", ".bdf": "EDFRawIO",
        ".vhdr": "BrainVisionRawIO",
        ".smr": "Spike2RawIO", ".smrx": "Spike2RawIO",
        ".ncs": "NeuralynxRawIO", ".nse": "NeuralynxRawIO", ".ntt": "NeuralynxRawIO",
        ".nvt": "NeuralynxRawIO", ".nrd": "NeuralynxRawIO",
        ".trc": "MicromedRawIO",
        ".plx": "PlexonRawIO", ".pl2": "Plexon2RawIO",
        ".rec": "SpikeGadgetsRawIO",
        ".meta": "SpikeGLXRawIO",
        ".medd": "MedRawIO", ".rdat": "MedRawIO", ".ridx": "MedRawIO",
        ".edr": "WinEdrRawIO", ".wcp": "WinWcpRawIO",
        ".xdat": "NeuroNexusRawIO",
        ".tbk": "TdtRawIO", ".tdx": "TdtRawIO", ".tev": "TdtRawIO",
        ".tin": "TdtRawIO", ".tnt": "TdtRawIO", ".tsq": "TdtRawIO", ".sev": "TdtRawIO",
    }

    _DIR_BASED_RAWIO: frozenset[str] = frozenset({
        "OpenEphysBinaryRawIO", "OpenEphysRawIO", "SpikeGLXRawIO", "TdtRawIO",
    })

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
        self._loader:  None 

        self.rate: float = 0.0
        self.dtype: str = dtype
        self.starting_time: float = 0.0
        self._n_channels: int = 0
        self._n_samples: int = 0

        ext = self.path.suffix.lower()

        if ext == ".nwb" and is_url(str(self.path)):
            self._init_remote_nwb()
        elif rawio_name := self.KNOWN_EXTENSIONS.get(ext):
            self._init_neo(rawio_name, stream_id)
        elif ext in EPHYS_EXNTENSIONS_RAW:
            if n_channels is None or sampling_rate is None:
                raise ValueError(f"Raw binary '{ext}' requires n_channels and sampling_rate.")
            self._phylib_memmap(n_channels, sampling_rate, dtype, gain)
        else:
            supported = ", ".join(sorted(self.KNOWN_EXTENSIONS))
            raise ValueError(f"Unsupported format '{ext}'. Supported: {supported}, {', '.join(EPHYS_EXNTENSIONS_RAW)}")

    # -- backends -----------------------------------------------------------

    def _init_remote_nwb(self):
        loader = RemoteNWBLoader(str(self.path))
        self._loader = loader
        self._n_channels = loader.n_channels
        self._n_samples = len(loader)
        self.rate = loader.rate
        self.starting_time = loader.starting_time

    def _init_neo(self, rawio_name: str, stream_id: str):
        import neo.rawio
        rawio_cls = getattr(neo.rawio, rawio_name, None)
        if rawio_cls is None:
            raise ValueError(f"Neo rawio class '{rawio_name}' not available in this Neo installation.")

        path_kwarg = "dirname" if rawio_name in self._DIR_BASED_RAWIO else "filename"
        self._reader = rawio_cls(**{path_kwarg: str(self.path if path_kwarg == "filename" else self.path.parent)})
        self._reader.parse_header()
        self._init_neo_stream(stream_id)

    def _init_neo_stream(self, stream_id: str):
        stream_ids = list(self._reader.header["signal_streams"]["id"])
        if stream_id not in stream_ids:
            stream_id = stream_ids[0]
        self._stream_idx = stream_ids.index(stream_id)

        ch = self._stream_channels
        self._n_channels = len(ch)
        self.rate = float(ch["sampling_rate"][0])
        self.dtype = str(ch["dtype"][0])
        self._n_samples = self._reader.get_signal_size(
            block_index=0, seg_index=0, stream_index=self._stream_idx,
        )
        self.starting_time = float(
            self._reader.get_signal_t_start(
                block_index=0, seg_index=0, stream_index=self._stream_idx,
            )
        )
        
    def _phylib_memmap(self, n_channels: int, sampling_rate: float, dtype: str, gain: float):
        memmap = get_ephys_reader(
            self.path, n_channels=n_channels, sample_rate=sampling_rate, dtype=dtype, gain=gain
        )
        if memmap.ndim == 1:
            memmap = memmap[:, np.newaxis]
        self._loader = memmap
        self._n_channels = memmap.shape[1]
        self._n_samples = memmap.shape[0]
        self.rate = sampling_rate
        self.dtype = str(memmap.dtype) if memmap.dtype != np.dtype('int16') else "int16"



    # -- Neo helpers --------------------------------------------------------

    @property
    def _stream_channels(self) -> np.ndarray:
        channels = self._reader.header["signal_channels"]
        stream_id = self._reader.header["signal_streams"]["id"][self._stream_idx]
        return channels[channels["stream_id"] == stream_id]

    # -- public interface ---------------------------------------------------

    def __len__(self) -> int:
        return self._n_samples

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channel_names(self) -> list[str]:
        if self._loader is not None:
            return self._loader.channel_names
        return list(self._stream_channels["name"])

    @property
    def units(self) -> str:
        if self._loader is not None:
            return self._loader.units
        unit_str = str(self._stream_channels["units"][0])
        return unit_str or "a.u."

    @property
    def streams(self) -> dict | None:
        if self._reader is None:
            return None
        all_channels = self._reader.header["signal_channels"]
        return {
            sid: {
                "name": str(name),
                "n_channels": int(np.sum(mask := all_channels["stream_id"] == sid)),
                "rate": float(all_channels[mask]["sampling_rate"][0]),
                "dtype": str(all_channels[mask]["dtype"][0]),
            }
            for sid, name in zip(
                self._reader.header["signal_streams"]["id"],
                self._reader.header["signal_streams"]["name"],
            )
        }
        
    def __getitem__(self, key) -> NDArray[np.float64]:
        if isinstance(key, slice):
            start, stop, _ = key.indices(self._n_samples)
        else:
            start, stop = key, key + 1

        if getattr(self, '_loader', None) is not None:  # ← safe fallback
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
# Module-level loader cache (replaces SharedEphysCache)
# ---------------------------------------------------------------------------

_loader_cache: dict[tuple[str, str], GenericEphysLoader] = {}
_loader_lock = threading.Lock()


def get_loader(
    path: str | Path,
    stream_id: str = "0",
    n_channels: int | None = None,
    sampling_rate: float | None = None,
) -> GenericEphysLoader | None:
    """Get or create a GenericEphysLoader for the given path/stream."""
    key = (str(path), stream_id)
    with _loader_lock:
        if key not in _loader_cache:
            try:
                _loader_cache[key] = GenericEphysLoader(
                    path,
                    n_channels=n_channels,
                    sampling_rate=sampling_rate,
                    stream_id=stream_id,
                )
            except Exception as e:
                print(f"Failed to load ephys file {path}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return None
        return _loader_cache[key]


def clear_loader_cache():
    with _loader_lock:
        _loader_cache.clear()


# Backward-compat alias
class SharedEphysCache:
    get_loader = staticmethod(get_loader)
    clear_cache = staticmethod(clear_loader_cache)


# ---------------------------------------------------------------------------
# EphysTraceBuffer – min/max envelope downsampling
# ---------------------------------------------------------------------------
class EphysTraceBuffer:

    def __init__(self, loader: EphysLoader | None = None, channel: int = 0):
        self.loader = loader
        self.channel = channel
        self.ephys_sr: float | None = None
        self._starting_time: float = 0.0
        self._preproc_flags: dict = {}
        self._cache: NDArray | None = None
        self._cache_start: int = 0
        self._cache_stop: int = 0
        self._cache_mean: NDArray | None = None
        self._cache_std: NDArray | None = None
        self.channel_spacing = 3.0
        self.display_gain: float = 0.0
        self.autocenter: bool = False

    def set_loader(self, loader: EphysLoader, channel: int = 0):
        self.loader = loader
        self.channel = channel
        self.ephys_sr = loader.rate
        self._starting_time = float(getattr(loader, "starting_time", 0.0))
        self._invalidate_cache()

    def set_preprocessing(self, flags: dict):
        self._preproc_flags = flags
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._cache = None
        self._cache_mean = None
        self._cache_std = None
        self._pyramid = {}

    def _covers_range(self, start: int, stop: int) -> bool:
        if self._cache is None:
            return False
        n_view = stop - start
        margin = int(n_view * BUFFER_COVERAGE_MARGIN)
        return self._cache_start <= start - margin and self._cache_stop >= stop + margin

    def _build_cache(self, view_start: int | None = None, view_stop: int | None = None, scroll_direction: str = "right"):
        if self.loader is None:
            return
        seg_start = 0
        seg_stop = len(self.loader)
        if seg_stop <= seg_start:
            return

        if view_start is None or view_stop is None:
            default_window = int(10.0 * self.ephys_sr) if self.ephys_sr else seg_stop - seg_start
            view_start = seg_start
            view_stop = min(seg_stop, seg_start + default_window)

        n_view = view_stop - view_start
        buffer_extra = int(n_view * DEFAULT_BUFFER_MULTIPLIER_EPHYS / 2)
        cache_start = max(seg_start, view_start - buffer_extra)
        cache_stop = min(seg_stop, view_stop + buffer_extra)

        if cache_stop <= cache_start:
            return

        raw = self.loader[cache_start:cache_stop]
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]

        data = raw.astype(np.float32) if raw.dtype != np.float32 else raw

        self._cache = data
        self._cache_start = cache_start
        self._cache_stop = cache_start + len(data)

        # Precompute multi-resolution pyramid
        self._pyramid = {1: data}
        from .app_constants import PYRAMID_LEVELS
        for level in PYRAMID_LEVELS:
            n = data.shape[0] // level
            if n == 0:
                continue
            reshaped = data[:n * level].reshape(n, level, data.shape[1])
            minv = reshaped.min(axis=1)
            maxv = reshaped.max(axis=1)
            out = np.empty((2 * n, data.shape[1]), dtype=np.float32)
            out[0::2] = minv
            out[1::2] = maxv
            self._pyramid[level] = out

        if self._cache_mean is None:
            self._cache_mean = data.mean(axis=0)
            per_ch_std = data.std(axis=0)
            per_ch_std[per_ch_std == 0] = 1.0
            median_std = np.median(per_ch_std)
            self._cache_std = np.full_like(per_ch_std, median_std)

    def ensure_cache(self, t0_s: float, t1_s: float, scroll_direction: str = "right"):
        if self.loader is None or self.ephys_sr is None:
            return
        start = max(0, int((t0_s - self._starting_time) * self.ephys_sr))
        stop = min(len(self.loader), int((t1_s - self._starting_time) * self.ephys_sr) + 1)
        if not self._covers_range(start, stop):
            self._build_cache(start, stop, scroll_direction)

    def _get_data(self, start: int, stop: int) -> NDArray | None:
        if self._cache is None:
            return None
        local_start = max(0, start - self._cache_start)
        local_stop = min(self._cache.shape[0], stop - self._cache_start)
        if local_stop <= local_start:
            return None
        return self._cache[local_start:local_stop]



    # -- multi channel ------------------------------------------------------

    def get_multichannel_trace_data(
        self, t0: float, t1: float, screen_width: int = 1920,
        channel_range: tuple[int, int] | None = None,
        channel_indices: NDArray | None = None,
        y_positions: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, int, int] | None:
        if self.loader is None:
            return None

        total_ch = self.loader.n_channels if hasattr(self.loader, 'n_channels') else 1
        if total_ch <= 1:
            return None

        start = max(0, int((t0 - self._starting_time) * self.ephys_sr))
        stop = min(len(self.loader), int((t1 - self._starting_time) * self.ephys_sr) + 1)
        if stop <= start:
            return None

        if not self._covers_range(start, stop):
            self._build_cache(start, stop)

        data_all = self._get_data(start, stop)
        if data_all is None or data_all.ndim < 2:
            return None

        if channel_indices is not None:
            valid = channel_indices[channel_indices < data_all.shape[1]]
            if len(valid) == 0:
                return None
            data_all = data_all[:, valid]
            ch_mean = self._cache_mean[valid]
            ch_std_arr = self._cache_std[valid]
            n_ch = len(valid)
        else:
            ch_start, ch_end = (channel_range or (0, total_ch - 1))
            ch_start = max(0, ch_start)
            ch_end = min(total_ch - 1, ch_end)
            n_ch = ch_end - ch_start + 1
            if n_ch <= 0:
                return None
            data_all = data_all[:, ch_start:ch_end + 1]
            ch_mean = self._cache_mean[ch_start:ch_end + 1]
            ch_std_arr = self._cache_std[ch_start:ch_end + 1]

        step = max(1, (stop - start) // screen_width)

        # Pick best precomputed pyramid level (Phy-style multiresolution)
        local_start = max(0, start - self._cache_start)
        local_stop = min(self._cache.shape[0], stop - self._cache_start)
        level = 1
        from .app_constants import PYRAMID_LEVELS
        if step > 1 and hasattr(self, '_pyramid'):
            for l in PYRAMID_LEVELS:
                if step >= l and l in self._pyramid:
                    level = l
                    break

        if level > 1:
            # Use precomputed pyramid — already has min/max interleaved
            pyr_data = self._pyramid[level]
            pyr_i0 = local_start // level
            pyr_i1 = local_stop // level
            if pyr_i1 <= pyr_i0:
                return None
            # Pyramid has all cache channels; slice to selected channels
            pyr_slice = pyr_data[pyr_i0 * 2:pyr_i1 * 2]
            if channel_indices is not None:
                valid = channel_indices[channel_indices < pyr_slice.shape[1]]
                display = pyr_slice[:, valid]
            else:
                display = pyr_slice[:, ch_start:ch_end + 1]
            n_display = pyr_i1 - pyr_i0
            aligned_start = self._cache_start + pyr_i0 * level
            half_level = level / 2
            times = self._starting_time + np.arange(
                aligned_start, aligned_start + 2 * n_display * half_level, half_level
            ) / self.ephys_sr
            if len(times) > display.shape[0]:
                times = times[:display.shape[0]]
        elif step > 1:
            # Inline min/max fallback (step doesn't match any pyramid level)
            n_segments = len(data_all) // step
            if n_segments == 0:
                return None
            usable = n_segments * step
            data_reshaped = data_all[:usable].reshape(n_segments, step, n_ch)
            min_vals = data_reshaped.min(axis=1)
            max_vals = data_reshaped.max(axis=1)
            display = np.empty((2 * n_segments, n_ch), dtype=np.float32)
            display[0::2, :] = min_vals
            display[1::2, :] = max_vals
            aligned_start = start
            half_step = step / 2
            times = self._starting_time + np.arange(
                aligned_start, aligned_start + 2 * n_segments * half_step, half_step
            ) / self.ephys_sr
        else:
            display = data_all.copy()
            times = self._starting_time + np.arange(start, start + len(display)) / self.ephys_sr

        if self.autocenter:
            display = (display - display.mean(axis=0)) / ch_std_arr
        else:
            display = (display - ch_mean) / ch_std_arr

        gain_factor = 0.75 ** (-self.display_gain)
        display *= gain_factor

        # Vectorized y offset (no Python loop)
        if y_positions is not None:
            display += y_positions[np.newaxis, :]
        else:
            offsets = np.arange(n_ch - 1, -1, -1, dtype=np.float32) * self.channel_spacing
            display += offsets[np.newaxis, :]

        return times, display, step, n_ch


# ---------------------------------------------------------------------------
# EphysTracePlot – BasePlot-based ephys waveform viewer
# ---------------------------------------------------------------------------


# Phy-inspired color scheme
# https://github.com/cortex-lab/phy/blob/master/phy/cluster/views/trace.py
# Phy uses a purple→teal channel gradient: (.353,.161,.443) to (.133,.404,.396)
_PHY_BG = '#000000'
_PHY_TRACE_COLOR_0 = (90, 41, 113)   # deep purple — top channel
_PHY_TRACE_COLOR_1 = (34, 103, 101)  # teal-green  — bottom channel
_PHY_TRACE_SINGLE = '#808080'        # neutral gray for single-channel mode
_PHY_AXIS = '#AAAAAA'

class EphysTracePlot(BasePlot):
    _initializing = False
    """Extracellular waveform viewer inheriting BasePlot for full GUI integration."""

    gain_scroll_requested = Signal(int)      # delta: +1 = increase, -1 = decrease
    visible_channels_changed = Signal(int, int)  # (first_visible_index, last_visible_index)
    y_space_changed = Signal()  # emitted when global y-coordinate space is rebuilt

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        self.setBackground(_PHY_BG)
        for axis_name in ('left', 'bottom'):
            axis = self.plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen(_PHY_AXIS))
            axis.setTextPen(pg.mkPen(_PHY_AXIS))
        self.time_marker.setPen(pg.mkPen('#FF4444', width=2, style=pg.QtCore.Qt.DotLine))

        self.setLabel('left', 'Amplitude')

                    
        self.trace_item = pg.PlotDataItem(
            connect='finite', antialias=False, skipFiniteCheck=True,
        )
        self.trace_item.setPen(pg.mkPen(color=_PHY_TRACE_COLOR_0, width=1.5))
        self.addItem(self.trace_item)

        # Second trace item for alternating channel colors (teal, odd channels)
        self.trace_item2 = pg.PlotDataItem(
            connect='finite', antialias=False, skipFiniteCheck=True,
        )
        self.trace_item2.setPen(pg.mkPen(color=_PHY_TRACE_COLOR_1, width=1.0))
        self.addItem(self.trace_item2)

        self.buffer = EphysTraceBuffer()

        self.label_items = []

        # Debounce-only — renders are expensive (blocking); throttle would queue renders
        self._td = ThrottleDebounce(
            debounce_ms=EPHYSTRACE_DEBOUNCE_MS,
            debounce_cb=self._do_range_update,
        )

        self.vb.sigRangeChanged.connect(self._on_view_range_changed)
        self.vb.installEventFilter(self)

        # Right-click drag state (Phy-style vertical scaling)
        # https://github.com/cortex-lab/phy/blob/master/phy/plot/interact.py
        self._orig_mouseDragEvent = self.vb.mouseDragEvent
        self.vb.mouseDragEvent = self._vb_mouse_drag_event
        self._drag_last_y: float | None = None
        self._drag_last_x: float | None = None
        self._drag_gain_accum: float = 0.0
        self._drag_x_accum: float = 0.0

        # Multi-channel state (always enabled)
        self.vb.setMouseEnabled(x=True, y=True)
        self._channel_range: tuple[int, int] | None = None
        self._custom_channel_set: NDArray | None = None

        # Global Y coordinate space (Phy-style viewport zoom)
        self._total_ordered_channels: NDArray = np.array([], dtype=int)
        self._hw_to_global_y: dict[int, float] = {}
        self._hw_to_order_idx: dict[int, int] = {}
        self._last_visible_hw: set[int] = set()

        self._ephys_offset: float = 0.0
        self._trial_duration: float | None = None
        self._source: TimeseriesSource | None = None

        # Calibration scale bars
        self._scale_v_line: pg.PlotDataItem | None = None
        self._scale_h_line: pg.PlotDataItem | None = None
        self._scale_v_text: pg.TextItem | None = None
        self._scale_h_text: pg.TextItem | None = None


        # Hardware (hw) channels vs Kilosort (KS) channels:
        #   hw = physical channel index in the recording file (e.g. Intan "A-009").
        #        The loader's __getitem__ and channel_names are indexed by hw.
        #   KS  = Kilosort's 0-based reindex after dropping dead/reference channels
        #        via channel_map.npy. KS channel k reads hw channel channel_map[k].
        # _probe_channel_order contains hw indices sorted by probe y-position
        # (depth), derived as channel_map[argsort(channel_positions[:, 1])].
        self._probe_channel_order: NDArray | None = None

        # Spike waveform overlays (from Kilosort)
        self._spike_times_local: NDArray | None = None
        self._spike_samples_abs: NDArray | None = None
        self._spike_channels: list[int] = []  # neighbor channels sorted by proximity
        self._spike_waveform_items: list[pg.PlotDataItem] = []
        self._spike_snippet_ms = 0.5  # ms before and after spike peak (1.0ms total, matches Phy)
        self._pen_spike = pg.mkPen(color=(255, 50, 50, 220), width=2.0)
        self._pen_spike_dim = pg.mkPen(color=(255, 80, 80, 100), width=1.5)

        # Multi-cluster spike overlays (Show all good neurons)
        self._multi_spike_data: list[tuple[NDArray, NDArray, list[int], tuple]] = []

        # Pre-cached pens for multichannel (avoid per-frame allocation)
        self._pen_multi = [
            pg.mkPen(color=_PHY_TRACE_COLOR_0, width=1.0),
            pg.mkPen(color=_PHY_TRACE_COLOR_1, width=1.0),
        ]
        self._pen_multi_thick = [
            pg.mkPen(color=_PHY_TRACE_COLOR_0, width=1.5),
            pg.mkPen(color=_PHY_TRACE_COLOR_1, width=1.5),
        ]
        self._last_n_ch: int = 0

        self.setToolTip("Double-click or Ctrl+A to autoscale")

    def set_source(self, source: TimeseriesSource | None):
        self._source = source

    def set_ephys_offset(self, offset: float, trial_duration: float | None = None):
        self._ephys_offset = offset
        self._trial_duration = trial_duration
        self.buffer._invalidate_cache()

    def set_loader(self, loader: EphysLoader, channel: int = 0):
        type(self)._initializing = True
        self.buffer.set_loader(loader, channel)
        self._setup_global_y_space()
        self._update_amplitude_label()
        type(self)._initializing = False
        if self.current_range:
            self.update_plot_content(*self.current_range)

    def set_channel(self, channel: int):
        self.buffer.channel = channel
        self._update_amplitude_label()
        if self.current_range and not type(self)._initializing:
            self.update_plot_content(*self.current_range)

        


    def set_channel_range(self, ch_start: int, ch_end: int):
        self._channel_range = (ch_start, ch_end)
        if len(self._total_ordered_channels) == 0:
            return
        spacing = self.buffer.channel_spacing
        total = len(self._total_ordered_channels)
        y_lo = (total - 1 - ch_end) * spacing - spacing * 0.5
        y_hi = (total - 1 - ch_start) * spacing + spacing * 0.5
        self.plot_item.setYRange(y_lo, y_hi, padding=0)

    def set_custom_channel_set(self, hw_indices: NDArray | None):
        was_custom = self._custom_channel_set is not None
        self._custom_channel_set = hw_indices
        self._last_n_ch = 0
        needs_rebuild = hw_indices is not None or was_custom
        if needs_rebuild:
            self._setup_global_y_space()
            if self.current_range and not type(self)._initializing:
                self.update_plot_content(*self.current_range)

    def set_probe_channel_order(self, order: NDArray | None):
        self._probe_channel_order = order
        self._last_n_ch = 0
        self._setup_global_y_space()
        if self.current_range and not type(self)._initializing:
            self.update_plot_content(*self.current_range)

    def eventFilter(self, obj, event):
        if obj is self.vb and event.type() == QEvent.GraphicsSceneWheel:
            modifiers = event.modifiers()
            delta = 1 if event.delta() > 0 else -1
            if modifiers & Qt.ControlModifier:
                self.gain_scroll_requested.emit(delta)
                event.accept()
                return True
            if modifiers & Qt.AltModifier:
                spacing = self.buffer.channel_spacing
                y_lo, y_hi = self.vb.viewRange()[1]
                shift = spacing * 2 * (-delta)
                self.plot_item.setYRange(y_lo + shift, y_hi + shift, padding=0)
                event.accept()
                return True
            # Plain wheel: fall through to pyqtgraph native X+Y zoom
        return super().eventFilter(obj, event)

    def _vb_mouse_drag_event(self, ev):
        """Right-click drag: Phy-style interactive control.

        Vertical drag   -> Y viewport zoom (show more/fewer channels)
        Horizontal drag -> X-axis zoom (time window size)

        Adapted from Phy's right-drag box scaling:
        https://github.com/cortex-lab/phy/blob/master/phy/plot/interact.py
        """
        if ev.button() == Qt.RightButton:
            ev.accept()
            pos = ev.pos()
            if ev.isStart():
                self._drag_last_y = pos.y()
                self._drag_last_x = pos.x()
                self._drag_gain_accum = 0.0
                self._drag_x_accum = 0.0
            elif ev.isFinish():
                self._drag_last_y = None
                self._drag_last_x = None
            else:
                step = 8.0
                if self._drag_last_y is not None:
                    dy = pos.y() - self._drag_last_y
                    self._drag_last_y = pos.y()
                    self._drag_gain_accum += dy
                    while abs(self._drag_gain_accum) >= step:
                        direction = 1 if self._drag_gain_accum > 0 else -1
                        y_lo, y_hi = self.vb.viewRange()[1]
                        y_center = (y_lo + y_hi) / 2
                        y_span = y_hi - y_lo
                        factor = 1.1 if direction > 0 else 1.0 / 1.1
                        new_span = y_span * factor
                        self.plot_item.setYRange(
                            y_center - new_span / 2, y_center + new_span / 2,
                            padding=0,
                        )
                        self._drag_gain_accum -= direction * step
                if self._drag_last_x is not None:
                    dx = pos.x() - self._drag_last_x
                    self._drag_last_x = pos.x()
                    self._drag_x_accum += dx
                    while abs(self._drag_x_accum) >= step:
                        direction = 1 if self._drag_x_accum > 0 else -1
                        xmin, xmax = self.get_current_xlim()
                        span = xmax - xmin
                        factor = 0.9 if direction > 0 else 1.0 / 0.9
                        new_span = span * factor
                        center = (xmin + xmax) / 2
                        self.plot_item.setXRange(
                            center - new_span / 2, center + new_span / 2,
                            padding=0,
                        )
                        self._drag_x_accum -= direction * step
        else:
            self._orig_mouseDragEvent(ev)

    def set_multichannel(self, enabled: bool):

        self.trace_item.setData([], [])
        self.trace_item2.setData([], [])
        if enabled:
            self._setup_global_y_space()
            self.vb.setMouseEnabled(x=True, y=True)
        else:
            self._reset_y_axis_ticks()
            self.vb.setLimits(yMin=None, yMax=None)
            self.vb.setMouseEnabled(x=True, y=False)
            self._last_visible_hw = set()
        if self.current_range and not type(self)._initializing:
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

        self._update_multichannel(t0, t1)


        self.current_range = (t0, t1)

    _pen_single_thin = pg.mkPen(color=_PHY_TRACE_SINGLE, width=1.0)
    _pen_single_thick = pg.mkPen(color=_PHY_TRACE_SINGLE, width=2.0)


    def _update_multichannel(self, t0: float, t1: float):
        visible_t0, visible_t1 = t0, t1

        ch_indices, y_positions = self._channels_in_viewport()
        if len(ch_indices) == 0:
            print(f"[EphysTracePlot] _update_multichannel: No channels in viewport for t0={t0}, t1={t1}")
            return

        # Expand draw range beyond the viewport so the trace is pre-rendered
        # for upcoming positions (eliminates black blinking during playback).
        window = visible_t1 - visible_t0
        buf_s = window * DEFAULT_BUFFER_MULTIPLIER_EPHYS / 2
        t0_draw = max(0.0, visible_t0 - buf_s)
        t1_draw = visible_t1 + buf_s
        if self._trial_duration is not None:
            t1_draw = min(self._trial_duration, t1_draw)
        draw_window = t1_draw - t0_draw

        # Scale pixel_width so sample density matches the visible range.
        pixel_width_base = max(self.width(), 400)
        pixel_width_draw = int(pixel_width_base * draw_window / window) if window > 0 else pixel_width_base

        # Buffer returns data with y_positions already baked in
        result = self.buffer.get_multichannel_trace_data(
            t0_draw + self._ephys_offset, t1_draw + self._ephys_offset,
            pixel_width_draw, channel_indices=ch_indices,
            y_positions=y_positions,
        )

        if result is None:
            print(f"[EphysTracePlot] _update_multichannel: No data for t0={t0_draw}, t1={t1_draw}, channels={ch_indices}")
            return

        times, data_2d, step, n_ch = result
        times = times - self._ephys_offset
        n_t = data_2d.shape[0]


        # --- Two batched draw calls (alternating purple/teal per channel) ---
        # Split even channels -> trace_item (purple), odd -> trace_item2 (teal)
        even_chs = [i for i in range(n_ch) if i % 2 == 0]
        odd_chs  = [i for i in range(n_ch) if i % 2 == 1]

        def _pack(ch_list):
            if not ch_list:
                return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            pts = len(ch_list) * (n_t + 1)
            xb = np.empty(pts, dtype=np.float32)
            yb = np.empty(pts, dtype=np.float32)
            for k, ch in enumerate(ch_list):
                s = k * (n_t + 1)
                xb[s:s + n_t] = times
                yb[s:s + n_t] = data_2d[:, ch]
                xb[s + n_t] = np.nan
                yb[s + n_t] = np.nan
            return xb, yb

        xe, ye = _pack(even_chs)
        xo, yo = _pack(odd_chs)
        self.trace_item.setData(xe, ye)
        self.trace_item2.setData(xo, yo)

        # Rebuild ticks when visible channel set changes
        current_visible = set(int(c) for c in ch_indices)
        if current_visible != self._last_visible_hw:
            self._last_visible_hw = current_visible
            self._last_n_ch = n_ch
            channel_names = self._get_channel_names(ch_indices)
            ticks = [
                (float(y_positions[i]), channel_names[i])
                for i in range(n_ch)
            ]
            left_axis = self.plot_item.getAxis('left')
            left_axis.setTicks([ticks])
            self.setLabel('left', '')

            # Emit visible channel range for range slider sync
            if self._hw_to_order_idx:
                indices_in_order = [self._hw_to_order_idx[hw] for hw in ch_indices if hw in self._hw_to_order_idx]
                if indices_in_order:
                    self.visible_channels_changed.emit(min(indices_in_order), max(indices_in_order))

        self._update_scale_bars(visible_t0, visible_t1)
        self._update_spike_waveforms(visible_t0, visible_t1)

    def _update_scale_bars(self, t0: float, t1: float):
        self._clear_scale_bars()

        if self.buffer.ephys_sr is None:
            return

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

        # Multichannel: display is z-normalized (divided by median_std), then scaled by gain
        median_std = self.buffer._cache_std[0] if self.buffer._cache_std is not None else 1.0
        voltage_per_display_unit = median_std / gain_factor

        voltage_bar_display = voltage_in_loader / voltage_per_display_unit

        v_label = "0.2 mV"

        # Position: bottom-right corner
        y_range = self.plot_item.getViewBox().viewRange()[1]
        y_span = y_range[1] - y_range[0]
        x_anchor = t1 - time_window * 0.03
        y_anchor = y_range[0] + y_span * 0.05

        vb = self.plot_item.getViewBox()

        # Vertical bar (voltage)
        v_x = [x_anchor, x_anchor]
        v_y = [y_anchor, y_anchor + voltage_bar_display]
        self._scale_v_line = pg.PlotDataItem(
            v_x, v_y, pen=pg.mkPen('#FFFFFF', width=2),
        )
        self._scale_v_line.setZValue(900)
        vb.addItem(self._scale_v_line, ignoreBounds=True)

        self._scale_v_text = pg.TextItem(v_label, color='#FFFFFF', anchor=(1.0, 0.5))
        self._scale_v_text.setPos(x_anchor - time_window * 0.005, y_anchor + voltage_bar_display / 2)
        self._scale_v_text.setZValue(900)
        vb.addItem(self._scale_v_text, ignoreBounds=True)

        # Horizontal bar (time)
        h_x = [x_anchor - time_bar_s, x_anchor]
        h_y = [y_anchor, y_anchor]
        self._scale_h_line = pg.PlotDataItem(
            h_x, h_y, pen=pg.mkPen('#FFFFFF', width=2),
        )
        self._scale_h_line.setZValue(900)
        vb.addItem(self._scale_h_line, ignoreBounds=True)

        if time_bar_ms >= 1000:
            h_label = f"{time_bar_ms / 1000:.1f} s"
        else:
            h_label = f"{time_bar_ms:.0f} ms"
        self._scale_h_text = pg.TextItem(h_label, color='#FFFFFF', anchor=(0.5, 1.0))
        self._scale_h_text.setPos(x_anchor - time_bar_s / 2, y_anchor - (y_range[1] - y_range[0]) * 0.01)
        self._scale_h_text.setZValue(900)
        vb.addItem(self._scale_h_text, ignoreBounds=True)

    def _clear_scale_bars(self):
        vb = self.plot_item.getViewBox()
        for attr in ('_scale_v_line', '_scale_h_line', '_scale_v_text', '_scale_h_text'):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    vb.removeItem(item)
                except (RuntimeError, ValueError):
                    pass
                setattr(self, attr, None)

    def _visible_hw_channels(self) -> NDArray:
        if self._custom_channel_set is not None:
            return self._custom_channel_set
        if self._probe_channel_order is not None:
            all_ch = self._probe_channel_order
        else:
            total = self.buffer.loader.n_channels if hasattr(self.buffer.loader, 'n_channels') else 1
            all_ch = np.arange(total)
        if self._channel_range:
            lo = max(0, self._channel_range[0])
            hi = min(len(all_ch) - 1, self._channel_range[1])
            return all_ch[lo:hi + 1]
        return all_ch

    def _all_ordered_channels(self) -> NDArray:
        if self._custom_channel_set is not None:
            return self._custom_channel_set
        if self._probe_channel_order is not None:
            return self._probe_channel_order
        if self.buffer.loader is None:
            return np.array([], dtype=int)
        total = self.buffer.loader.n_channels if hasattr(self.buffer.loader, 'n_channels') else 1
        return np.arange(total)

    def _setup_global_y_space(self):
        self._total_ordered_channels = self._all_ordered_channels()
        total = len(self._total_ordered_channels)
        if total == 0:
            return
        spacing = self.buffer.channel_spacing
        self._hw_to_global_y = {
            int(hw): (total - 1 - i) * spacing
            for i, hw in enumerate(self._total_ordered_channels)
        }
        self._hw_to_order_idx = {
            int(hw): i for i, hw in enumerate(self._total_ordered_channels)
        }
        margin = spacing * 0.5
        y_max = (total - 1) * spacing + margin
        self.vb.setLimits(yMin=-margin, yMax=y_max)
        self.plot_item.setYRange(-margin, y_max, padding=0)
        self.y_space_changed.emit()

    def _channels_in_viewport(self) -> tuple[NDArray, NDArray]:
        y_lo, y_hi = self.vb.viewRange()[1]
        spacing = self.buffer.channel_spacing
        half = spacing * 0.5
        hw_indices = []
        offsets = []
        for i, hw in enumerate(self._total_ordered_channels):
            y_pos = self._hw_to_global_y.get(int(hw))
            if y_pos is None:
                continue
            if (y_pos + half) >= y_lo and (y_pos - half) <= y_hi:
                hw_indices.append(int(hw))
                offsets.append(y_pos)
        return np.array(hw_indices, dtype=int), np.array(offsets, dtype=float)

    def _get_channel_names(self, hw_indices: NDArray | None = None) -> list[str]:
        if hw_indices is None:
            hw_indices = self._visible_hw_channels()
        return [f"Ch {i}" for i in hw_indices]

    def _reset_y_axis_ticks(self):
        left_axis = self.plot_item.getAxis('left')
        left_axis.setTicks(None)
        self._update_amplitude_label()

    def _update_amplitude_label(self):
        loader = self.buffer.loader
        units = ''
        if loader is not None and hasattr(loader, 'units'):
            units = loader.units

        channel_name = None
        if loader is not None and hasattr(loader, 'channel_names'):
            ch = self.buffer.channel

        if channel_name:
            label = f"Amplitude Ch{ch}"
        else:
            label = "Amplitude"

        if units and units != 'a.u.':
            self.setLabel('left', label, units=units)
        else:
            self.setLabel('left', label)

    def set_spike_data(
        self,
        spike_times_local: NDArray,
        spike_samples_abs: NDArray,
        channels: list[int] | None = None,
    ):
        self.clear_spike_overlays()
        if len(spike_times_local) == 0:
            return
        order = np.argsort(spike_times_local)
        self._spike_times_local = spike_times_local[order]
        self._spike_samples_abs = spike_samples_abs[order]
        self._spike_channels = channels or [self.buffer.channel]
        t0, t1 = self.get_current_xlim()
        self._update_spike_waveforms(t0, t1)

    def clear_spike_overlays(self):
        self._spike_times_local = None
        self._spike_samples_abs = None
        self._spike_channels = []
        self._multi_spike_data.clear()
        vb = self.plot_item.getViewBox()
        for item in self._spike_waveform_items:
            try:
                vb.removeItem(item)
            except (RuntimeError, ValueError):
                pass
        self._spike_waveform_items.clear()

    def set_multi_cluster_spike_data(
        self,
        cluster_entries: list[tuple[NDArray, NDArray, list[int], tuple]],
    ):
        self.clear_spike_overlays()
        self._multi_spike_data = cluster_entries
        t0, t1 = self.get_current_xlim()
        self._update_spike_waveforms(t0, t1)

    def _update_spike_waveforms(self, t0: float | None = None, t1: float | None = None):
        vb = self.plot_item.getViewBox()
        for item in self._spike_waveform_items:
            try:
                vb.removeItem(item)
            except (RuntimeError, ValueError):
                pass
        self._spike_waveform_items.clear()

        if self.buffer.ephys_sr is None or self.buffer._cache is None:
            return

        if t0 is None or t1 is None:
            t0, t1 = self.get_current_xlim()

        if self._multi_spike_data:
            self._draw_multi_cluster_waveforms(t0, t1)
            return

        if self._spike_times_local is None or len(self._spike_times_local) == 0:
            return

        self._draw_spike_waveforms_multi(t0, t1)


    def _draw_spike_waveforms_single(self, t0: float, t1: float):
        sr = self.buffer.ephys_sr
        half_w = int(self._spike_snippet_ms * 0.001 * sr)
        ch = self._spike_channels[0] if self._spike_channels else self.buffer.channel
        ch = min(ch, self.buffer._cache.shape[1] - 1)
        cache_start = self.buffer._cache_start
        cache_n = self.buffer._cache.shape[0]
        gain_factor = 0.75 ** (-self.buffer.display_gain) if self.buffer.display_gain != 0 else 1.0

        i_start = np.searchsorted(self._spike_times_local, t0, side='left')
        i_end = np.searchsorted(self._spike_times_local, t1, side='right')
        spike_samples = self._spike_samples_abs[i_start:i_end]
        if len(spike_samples) == 0:
            return

        cache_data = self.buffer._cache
        ephys_offset = self._ephys_offset
        max_snippet = 2 * half_w
        all_t = np.empty(len(spike_samples) * (max_snippet + 1))
        all_y = np.empty(len(spike_samples) * (max_snippet + 1))
        pos = 0

        for spike_s in spike_samples:
            local_idx = int(spike_s) - cache_start
            s0 = max(0, local_idx - half_w)
            s1 = min(cache_n, local_idx + half_w)
            if s1 <= s0:
                continue
            n = s1 - s0
            all_y[pos:pos + n] = cache_data[s0:s1, ch]
            all_t[pos:pos + n] = np.arange(s0 + cache_start, s1 + cache_start, dtype=np.float64) / sr - ephys_offset
            pos += n
            all_t[pos] = np.nan
            all_y[pos] = np.nan
            pos += 1

        if pos == 0:
            return
        all_t = all_t[:pos]
        all_y = all_y[:pos]
        if gain_factor != 1.0:
            all_y *= gain_factor

        item = pg.PlotDataItem(all_t, all_y, pen=self._pen_spike, connect='finite', antialias=False)
        item.setZValue(800)
        self.plot_item.getViewBox().addItem(item, ignoreBounds=True)
        self._spike_waveform_items.append(item)

    def _draw_spike_waveforms_multi(self, t0: float, t1: float):
        sr = self.buffer.ephys_sr
        half_w = int(self._spike_snippet_ms * 0.001 * sr)
        cache_start = self.buffer._cache_start
        cache_n = self.buffer._cache.shape[0]
        gain_factor = 0.75 ** (-self.buffer.display_gain)

        hw_to_y = self._hw_to_global_y
        y_lo, y_hi = self.vb.viewRange()[1]
        draw_channels = [ch for ch in self._spike_channels if ch in hw_to_y and y_lo <= hw_to_y[ch] <= y_hi]
        if not draw_channels:
            return

        cache_mean = self.buffer._cache_mean
        cache_std = self.buffer._cache_std
        if cache_mean is None or cache_std is None:
            return

        i_start = np.searchsorted(self._spike_times_local, t0, side='left')
        i_end = np.searchsorted(self._spike_times_local, t1, side='right')
        spike_samples = self._spike_samples_abs[i_start:i_end]
        if len(spike_samples) == 0:
            return

        cache_data = self.buffer._cache
        ephys_offset = self._ephys_offset
        max_snippet = 2 * half_w

        # Collect snippets into two pen groups: bright (top 3 channels) and dim
        groups = [
            (self._pen_spike, []),
            (self._pen_spike_dim, []),
        ]
        for rank, ch in enumerate(draw_channels):
            y_off = hw_to_y[ch]
            ch_m = float(cache_mean[ch])
            ch_s = float(cache_std[ch])
            bucket = groups[0][1] if rank < 3 else groups[1][1]
            for spike_s in spike_samples:
                local_idx = int(spike_s) - cache_start
                s0 = max(0, local_idx - half_w)
                s1 = min(cache_n, local_idx + half_w)
                if s1 > s0:
                    bucket.append((s0, s1, ch, ch_m, ch_s, y_off))

        vb = self.plot_item.getViewBox()
        for pen, entries in groups:
            if not entries:
                continue
            all_t = np.empty(len(entries) * (max_snippet + 1))
            all_y = np.empty(len(entries) * (max_snippet + 1))
            pos = 0
            for s0, s1, ch, ch_m, ch_s, y_off in entries:
                n = s1 - s0
                snippet = (cache_data[s0:s1, ch] - ch_m) / ch_s
                snippet = snippet * gain_factor + y_off
                all_y[pos:pos + n] = snippet
                all_t[pos:pos + n] = np.arange(s0 + cache_start, s1 + cache_start, dtype=np.float64) / sr - ephys_offset
                pos += n
                all_t[pos] = np.nan
                all_y[pos] = np.nan
                pos += 1
            if pos == 0:
                continue
            item = pg.PlotDataItem(all_t[:pos], all_y[:pos], pen=pen, connect='finite', antialias=False)
            item.setZValue(800)
            vb.addItem(item, ignoreBounds=True)
            self._spike_waveform_items.append(item)

    def _draw_multi_cluster_waveforms(self, t0: float, t1: float):
        sr = self.buffer.ephys_sr
        half_w = int(self._spike_snippet_ms * 0.001 * sr)
        cache_start = self.buffer._cache_start
        cache_n = self.buffer._cache.shape[0]
        cache_n_ch = self.buffer._cache.shape[1]
        cache_data = self.buffer._cache


        hw_to_y = self._hw_to_global_y
        y_lo, y_hi = self.vb.viewRange()[1] 
        gain_factor = 0.75 ** (-self.buffer.display_gain)
        cache_mean = self.buffer._cache_mean
        cache_std = self.buffer._cache_std
        ephys_offset = self._ephys_offset
        max_snippet = 2 * half_w

        vb = self.plot_item.getViewBox()

        for spike_times_local, spike_samples_abs, channels, color in self._multi_spike_data:
            if len(spike_times_local) == 0:
                continue

            i_start = np.searchsorted(spike_times_local, t0, side='left')
            i_end = np.searchsorted(spike_times_local, t1, side='right')
            spike_samples = spike_samples_abs[i_start:i_end]
            if len(spike_samples) == 0:
                continue

            draw_channels = [ch for ch in channels if ch in hw_to_y and ch < cache_n_ch and y_lo <= hw_to_y[ch] <= y_hi]
            if not draw_channels:
                continue
            total_alloc = len(spike_samples) * len(draw_channels) * (max_snippet + 1)
            all_t = np.empty(total_alloc)
            all_y = np.empty(total_alloc)
            pos = 0
            for ch in draw_channels:
                y_off = hw_to_y[ch]
                ch_m = float(cache_mean[ch])
                ch_s = float(cache_std[ch])
                for spike_s in spike_samples:
                    local_idx = int(spike_s) - cache_start
                    s0 = max(0, local_idx - half_w)
                    s1 = min(cache_n, local_idx + half_w)
                    if s1 <= s0:
                        continue
                    n = s1 - s0
                    snippet = (cache_data[s0:s1, ch] - ch_m) / ch_s
                    snippet = snippet * gain_factor + y_off
                    all_y[pos:pos + n] = snippet
                    all_t[pos:pos + n] = np.arange(s0 + cache_start, s1 + cache_start, dtype=np.float64) / sr - ephys_offset
                    pos += n
                    all_t[pos] = np.nan
                    all_y[pos] = np.nan
                    pos += 1

            if pos == 0:
                continue
            all_t = all_t[:pos]
            all_y = all_y[:pos]


            pen = pg.mkPen(color=color, width=2.0)
            item = pg.PlotDataItem(all_t, all_y, pen=pen, connect='finite', antialias=False)
            item.setZValue(800)
            vb.addItem(item, ignoreBounds=True)
            self._spike_waveform_items.append(item)

    def get_spike_target_time(self, delta: int = +1) -> float | None:
        if self._spike_times_local is None or len(self._spike_times_local) == 0:
            return None
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return None

        video = getattr(self.app_state, 'video', None)
        current_time = video.frame_to_time(self.app_state.current_frame) if video else self.app_state.current_frame / self.app_state.video_fps
        idx = np.searchsorted(self._spike_times_local, current_time)
        n = len(self._spike_times_local)
        target_idx = (idx + delta) % n
        return float(self._spike_times_local[target_idx])

    def jump_to_spike(self, delta: int = +1):
        target_time = self.get_spike_target_time(delta)
        if target_time is None:
            return

        xmin, xmax = self.get_current_xlim()
        half = (xmax - xmin) / 2
        new_xmin = target_time - half
        new_xmax = target_time + half

        self.plot_item.setXRange(new_xmin, new_xmax, padding=0)
        self.update_plot_content(new_xmin, new_xmax)

    def _get_time_bounds(self):
        if self._source is not None:
            tr = self._source.time_range
            if tr.duration > 0:
                return tr.start_s, tr.end_s
        if self._trial_duration is not None and self._trial_duration > 0:
            return 0.0, self._trial_duration
        return super()._get_time_bounds()


    def auto_channel_spacing(self):
        cache = self.buffer._cache
        if cache is None or cache.shape[1] < 2:
            return
        ch_indices = self._visible_hw_channels()
        if len(ch_indices) < 2:
            return
        valid = [ch for ch in ch_indices if ch < cache.shape[1]]
        if len(valid) < 2:
            return
        data = cache[:, valid].astype(np.float64)
        means = data.mean(axis=0)
        stds = data.std(axis=0)
        stds[stds == 0] = 1.0
        normed = (data - means) / stds
        p_low = np.percentile(normed, 1, axis=0)
        p_high = np.percentile(normed, 99, axis=0)
        spans = p_high - p_low
        max_span = float(np.max(spans))
        self.buffer.channel_spacing = max_span * 0.80

    def auto_gain(self) -> float:
        """Compute optimal display_gain using Phy's quantile-based approach.

        Phy (cortex-lab) normalises traces by computing quantile-based data
        bounds after per-channel median subtraction, then mapping those bounds
        into the available display range.

        Algorithm (adapted from phy/cluster/views/trace.py, ``plot()``):
          1. Subtract per-channel median  (DC removal, ``select_traces()``)
          2. ymin = quantile(data, q)     where q = 0.01  (``trace_quantile``)
             ymax = quantile(data, 1 - q)
          3. Map [ymin, ymax] into the channel lane height

        Reference:
          https://github.com/cortex-lab/phy/blob/master/phy/cluster/views/trace.py

        Returns the computed gain value.
        """
        if self.buffer._cache is None:
            if self.current_range:
                t0, t1 = self.current_range
                self.buffer.ensure_cache(
                    t0 + self._ephys_offset, t1 + self._ephys_offset,
                )
            else:
                self.buffer._build_cache()
        cache = self.buffer._cache
        if cache is None:
            return self.buffer.display_gain

        trace_quantile = 0.01
        ch_indices = self._visible_hw_channels()
        valid = ch_indices[ch_indices < cache.shape[1]]
        if len(valid) == 0:
            return self.buffer.display_gain

        data = cache[:, valid].astype(np.float64)
        data = data - np.median(data, axis=0)

        ymin = np.quantile(data, trace_quantile)
        ymax = np.quantile(data, 1.0 - trace_quantile)
        data_span = ymax - ymin
        if data_span == 0:
            return self.buffer.display_gain

        cache_std = self.buffer._cache_std[valid]
        median_std = float(np.median(cache_std))
        if median_std == 0:
            return self.buffer.display_gain

        normed_span = data_span / median_std
        target_span = 0.9 * self.buffer.channel_spacing
        optimal_factor = target_span / normed_span

        optimal_gain = -np.log(optimal_factor) / np.log(0.75)
        self.buffer.display_gain = round(float(optimal_gain), 1)

        if self.current_range:
            self.update_plot_content(*self.current_range)

        return self.buffer.display_gain

    def autoscale(self):
        total = len(self._total_ordered_channels)
        if total > 0:
            spacing = self.buffer.channel_spacing
            margin = spacing * 0.5
            self.plot_item.setYRange(-margin, (total - 1) * spacing + margin, padding=0)


        if self.current_range:
            self.update_plot_content(*self.get_current_xlim())

    def _on_view_range_changed(self):
        if not hasattr(self.app_state, 'ds') or self.app_state.ds is None:
            return
        self._td.trigger()

    def _do_range_update(self):
        t0, t1 = self.get_current_xlim()
        self.update_plot_content(t0, t1)

