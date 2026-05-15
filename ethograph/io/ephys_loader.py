"""Ephys loading module — unified interface for multiple formats.
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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ethograph.io.validation import EPHYS_EXTENSIONS_RAW

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EphysData:
    """Immutable ephys metadata and lazy data access.

    Single source of truth for:
    - Sample rate (rate)
    - Number of channels (n_channels)
    - Session start time (starting_time) — NO RECONCILIATION
    - Channel names and units

    The _loader_func handles all backend-specific loading.
    """

    rate: float  # sampling rate (Hz)
    n_channels: int
    starting_time: float = 0.0  # ← single source of truth
    channel_names: list[str] | None = None
    units: str = "V"
    stream_info: dict[str, dict] | None = None  # for multi-stream formats
    _loader_func: Any = None  # internal: backend-specific loader

    def __getitem__(self, key) -> NDArray:
        """Fetch data samples. Returns (n_samples, n_channels) or (n_samples,)."""
        if self._loader_func is None:
            raise RuntimeError("No loader attached to EphysData")
        return self._loader_func[key]

    def __len__(self) -> int:
        """Total number of samples."""
        if self._loader_func is None:
            raise RuntimeError("No loader attached to EphysData")
        return len(self._loader_func)


# ---------------------------------------------------------------------------
# Internal backend wrappers (minimal, focused)
# ---------------------------------------------------------------------------


class _NWBWrapper:
    """Lazy NWB ElectricalSeries loader."""

    def __init__(self, path: str, stream_id: str | None = None):
        import warnings

        import h5py
        import pynwb

        self._h5 = h5py.File(path, "r")
        self._io = pynwb.NWBHDF5IO(file=self._h5, load_namespaces=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*manufacturer.*deprecated", category=DeprecationWarning)
            self._nwb = self._io.read()

        # Discover all ElectricalSeries
        all_series = {
            name: es for name, es in self._nwb.acquisition.items() if isinstance(es, pynwb.ecephys.ElectricalSeries)
        }
        if not all_series:
            raise ValueError("No ElectricalSeries found in NWB acquisition")

        if stream_id is None or stream_id not in all_series:
            stream_id = next(iter(all_series))

        self._es = all_series[stream_id]
        self._data = self._es.data
        self._conversion = float(self._es.conversion) if self._es.conversion else 1.0
        self._n_channels = self._data.shape[1] if self._data.ndim > 1 else 1
        self.rate, self.starting_time = self._resolve_timing()
        self.units = str(self._es.unit) if hasattr(self._es, "unit") and self._es.unit else "V"
        self.channel_names = [f"Ch {i}" for i in range(self._n_channels)]
        self.stream_info = {
            sid: {
                "name": sid,
                "n_channels": (es.data.shape[1] if es.data.ndim > 1 else 1),
                "rate": self._resolve_timing(es)[0],
            }
            for sid, es in all_series.items()
        }

    def _resolve_timing(self, es=None):
        """Extract rate and starting_time from ElectricalSeries."""
        from ethograph.utils.nwb import resolve_timeseries_timing

        es = es or self._es
        return resolve_timeseries_timing(es)

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, key) -> NDArray[np.float64]:
        chunk = self._data[key]
        if chunk.ndim == 1 and self._n_channels == 1:
            chunk = chunk[:, np.newaxis]
        return chunk.astype(np.float64) * self._conversion

    def __del__(self):
        if hasattr(self, "_io") and self._io:
            try:
                self._io.close()
            except Exception:
                pass
        if hasattr(self, "_h5") and self._h5:
            try:
                self._h5.close()
            except Exception:
                pass


class _NeoWrapper:
    """Neo-based format loader (RHD, OpenEphys, Intan, etc)."""

    KNOWN_EXTENSIONS = {
        ".rhd": "IntanRawIO",
        ".rhs": "IntanRawIO",
        ".oebin": "OpenEphysBinaryRawIO",
        ".openephys": "OpenEphysRawIO",
        ".continuous": "OpenEphysRawIO",
        ".spikes": "OpenEphysRawIO",
        ".events": "OpenEphysRawIO",
        ".ns1": "BlackrockRawIO",
        ".ns2": "BlackrockRawIO",
        ".ns3": "BlackrockRawIO",
        ".ns4": "BlackrockRawIO",
        ".ns5": "BlackrockRawIO",
        ".ns6": "BlackrockRawIO",
        ".nev": "BlackrockRawIO",
        ".sif": "BlackrockRawIO",
        ".ccf": "BlackrockRawIO",
        ".abf": "AxonRawIO",
        ".axgx": "AxographRawIO",
        ".axgd": "AxographRawIO",
        ".edf": "EDFRawIO",
        ".bdf": "EDFRawIO",
        ".vhdr": "BrainVisionRawIO",
        ".smr": "Spike2RawIO",
        ".smrx": "Spike2RawIO",
        ".ncs": "NeuralynxRawIO",
        ".nse": "NeuralynxRawIO",
        ".ntt": "NeuralynxRawIO",
        ".nvt": "NeuralynxRawIO",
        ".nrd": "NeuralynxRawIO",
        ".trc": "MicromedRawIO",
        ".plx": "PlexonRawIO",
        ".pl2": "Plexon2RawIO",
        ".rec": "SpikeGadgetsRawIO",
        ".meta": "SpikeGLXRawIO",
        ".medd": "MedRawIO",
        ".rdat": "MedRawIO",
        ".ridx": "MedRawIO",
        ".edr": "WinEdrRawIO",
        ".wcp": "WinWcpRawIO",
        ".xdat": "NeuroNexusRawIO",
        ".tbk": "TdtRawIO",
        ".tdx": "TdtRawIO",
        ".tev": "TdtRawIO",
        ".tin": "TdtRawIO",
        ".tnt": "TdtRawIO",
        ".tsq": "TdtRawIO",
        ".sev": "TdtRawIO",
    }
    _DIR_BASED = frozenset({"OpenEphysBinaryRawIO", "OpenEphysRawIO", "SpikeGLXRawIO", "TdtRawIO"})

    def __init__(self, path: str, stream_id: str = "0"):
        import neo.rawio

        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        rawio_name = self.KNOWN_EXTENSIONS.get(ext)
        if not rawio_name:
            raise ValueError(f"Unknown ephys format: {ext}")

        rawio_cls = getattr(neo.rawio, rawio_name, None)
        if not rawio_cls:
            raise ValueError(f"Neo rawio '{rawio_name}' not available")

        is_dir = rawio_name in self._DIR_BASED
        key = "dirname" if is_dir else "filename"
        val = str(path_obj.parent if is_dir else path_obj)
        self._reader = rawio_cls(**{key: val})
        self._reader.parse_header()

        # Select stream
        stream_ids = list(self._reader.header["signal_streams"]["id"])
        if stream_id not in stream_ids:
            stream_id = stream_ids[0]
        self._stream_idx = stream_ids.index(stream_id)

        # Extract metadata
        ch = self._stream_channels
        self._n_channels = len(ch)
        self.rate = float(ch["sampling_rate"][0])
        self.starting_time = float(self._reader.get_signal_t_start(0, 0, self._stream_idx))
        self._n_samples = self._reader.get_signal_size(0, 0, self._stream_idx)
        self.units = str(ch["units"][0]) or "V"
        self.channel_names = list(ch["name"])

    @property
    def _stream_channels(self):
        ch = self._reader.header["signal_channels"]
        sid = self._reader.header["signal_streams"]["id"][self._stream_idx]
        return ch[ch["stream_id"] == sid]

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, key) -> NDArray[np.float64]:
        if isinstance(key, slice):
            start, stop, _ = key.indices(self._n_samples)
        else:
            start, stop = key, key + 1

        raw = self._reader.get_analogsignal_chunk(0, 0, start, stop, self._stream_idx)
        return self._reader.rescale_signal_raw_to_float(raw, dtype="float64", stream_index=self._stream_idx)

    @property
    def stream_info(self) -> dict:
        """Metadata for all streams."""
        all_ch = self._reader.header["signal_channels"]
        return {
            sid: {
                "name": str(name),
                "n_channels": int(np.sum(mask := all_ch["stream_id"] == sid)),
                "rate": float(all_ch[mask]["sampling_rate"][0]),
            }
            for sid, name in zip(
                self._reader.header["signal_streams"]["id"],
                self._reader.header["signal_streams"]["name"],
            )
        }


class _RawBinaryWrapper:
    """Raw binary format loader (phylib-based)."""

    def __init__(self, path: str, n_channels: int, sr: float, dtype: str, gain: float):
        from phylib.io.traces import get_ephys_reader

        if not n_channels or not sr:
            raise ValueError("Raw binary requires n_channels and sampling_rate")

        memmap = get_ephys_reader(path, n_channels=n_channels, sample_rate=sr, dtype=dtype, gain=gain)
        if memmap.ndim == 1:
            memmap = memmap[:, np.newaxis]

        self._memmap = memmap
        self.rate = sr
        self.starting_time = 0.0
        self._n_channels = memmap.shape[1]
        self.units = "V"
        self.channel_names = [f"Ch {i}" for i in range(self._n_channels)]

    def __len__(self) -> int:
        return self._memmap.shape[0]

    def __getitem__(self, key) -> NDArray[np.float64]:
        return self._memmap[key].astype(np.float64)

    @property
    def stream_info(self):
        return None


# ---------------------------------------------------------------------------
# Main entry point — unified loader
# ---------------------------------------------------------------------------


def load_ephys(
    path: str | Path,
    stream_id: str = "0",
    n_channels: int | None = None,
    sampling_rate: float | None = None,
    dtype: str = "int16",
    gain: float = 1.0,
) -> EphysData:
    """Load ephys file and return unified EphysData.

    Auto-detects format: NWB, Neo-supported (.rhd, .oebin, etc), or raw binary.
    No module-level cache — caller manages lifecycle.

    Parameters
    ----------
    path : str | Path
        Path to ephys file
    stream_id : str
        For multi-stream formats, which stream to load
    n_channels : int | None
        Required only for raw binary
    sampling_rate : float | None
        Required only for raw binary
    dtype : str
        Raw binary dtype (default "int16")
    gain : float
        Raw binary gain (default 1.0)

    Returns
    -------
    EphysData
        Immutable configuration + lazy data access
    """
    path = Path(path)
    ext = path.suffix.lower()

    try:
        if ext == ".nwb":
            loader = _NWBWrapper(str(path), stream_id=stream_id)
        elif ext in EPHYS_EXTENSIONS_RAW:
            loader = _RawBinaryWrapper(str(path), n_channels, sampling_rate, dtype, gain)
        else:
            loader = _NeoWrapper(str(path), stream_id=stream_id)

        return EphysData(
            rate=loader.rate,
            n_channels=loader._n_channels if hasattr(loader, "_n_channels") else len(loader.channel_names),
            starting_time=loader.starting_time,
            channel_names=loader.channel_names,
            units=loader.units,
            stream_info=getattr(loader, "stream_info", None),
            _loader_func=loader,
        )
    except Exception as e:
        logger.error(f"Failed to load ephys {path}: {type(e).__name__}: {e}", exc_info=True)
        raise
