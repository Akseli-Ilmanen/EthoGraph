"""Source-agnostic data providers for the spectrogram and plot pipelines.

Bridges the GUI-specific SpectrogramSource protocol with the general-purpose
TimeseriesSource abstractions in ethograph.utils.timeseries_source.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import xarray as xr

from ethograph.gui.plots_timeseriessource import (
    ArrayTimeseriesSource,
    RegularTimeseriesSource,
    TimeseriesSource,
)

from .plots_spectrogram import SharedAudioCache

if TYPE_CHECKING:
    pass


@runtime_checkable
class SpectrogramSource(Protocol):
    rate: float
    duration: float
    supports_noise_reduction: bool

    def get_data(self, t0: float, t1: float) -> np.ndarray: ...

    @property
    def identity(self) -> str: ...


# ---------------------------------------------------------------------------
# Concrete SpectrogramSource implementations (delegate to TimeseriesSource)
# ---------------------------------------------------------------------------


class AudioFileSource:
    """Audio file source for spectrogram and waveform plots.

    Internally delegates to ``RegularTimeseriesSource`` for data access.
    """

    supports_noise_reduction = True

    def __init__(self, audio_path: str, channel_idx: int = 0):
        self._audio_path = audio_path
        self._channel_idx = channel_idx
        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            raise ValueError(f"Failed to load audio: {audio_path}")
        self._ts = RegularTimeseriesSource(
            "audio", loader, channel=channel_idx,
        )

    @property
    def rate(self) -> float:
        return self._ts.sampling_rate

    @property
    def duration(self) -> float:
        return self._ts.time_range.duration

    @property
    def timeseries_source(self) -> RegularTimeseriesSource:
        return self._ts

    def get_data(self, t0: float, t1: float) -> np.ndarray:
        _, data = self._ts.get_data(t0, t1)
        return data

    @property
    def identity(self) -> str:
        return f"{self._audio_path}:{self._channel_idx}"


class XarraySource:
    """Wraps an xarray DataArray (already 1-D after selection) for spectrogram.

    Internally delegates to ``ArrayTimeseriesSource`` for data access.
    """

    supports_noise_reduction = False

    def __init__(
        self,
        da: xr.DataArray,
        time_coords: np.ndarray,
        variable_name: str,
        ds_kwargs_hash: str,
    ):
        self._variable_name = variable_name
        self._ds_kwargs_hash = ds_kwargs_hash
        data = np.asarray(da, dtype=np.float64)
        np.nan_to_num(data, copy=False, nan=0.0)
        self._ts = ArrayTimeseriesSource(
            variable_name,
            np.asarray(time_coords, dtype=np.float64),
            data,
            identity_hint=f"{variable_name}:{ds_kwargs_hash}",
        )

    @property
    def rate(self) -> float:
        return self._ts.sampling_rate

    @property
    def duration(self) -> float:
        return self._ts.time_range.duration

    @property
    def timeseries_source(self) -> ArrayTimeseriesSource:
        return self._ts

    def get_data(self, t0: float, t1: float) -> np.ndarray:
        _, data = self._ts.get_data(t0, t1)
        return data

    @property
    def identity(self) -> str:
        return f"xarray:{self._variable_name}:{self._ds_kwargs_hash}"


class EphysFileSource:
    """Ephys file source for spectrogram consumption.

    Internally delegates to ``RegularTimeseriesSource`` for data access.
    """

    supports_noise_reduction = False

    def __init__(self, path: str, stream_id: str = "0", channel_idx: int = 0):
        from .plots_ephystrace import SharedEphysCache

        self._path = path
        self._stream_id = stream_id
        self._channel_idx = channel_idx
        loader = SharedEphysCache.get_loader(path, stream_id)
        if loader is None:
            raise ValueError(f"Failed to load ephys: {path}")
        self._ts = RegularTimeseriesSource(
            "ephys", loader, channel=channel_idx,
        )

    @property
    def rate(self) -> float:
        return self._ts.sampling_rate

    @property
    def duration(self) -> float:
        return self._ts.time_range.duration

    @property
    def timeseries_source(self) -> RegularTimeseriesSource:
        return self._ts

    def get_data(self, t0: float, t1: float) -> np.ndarray:
        _, data = self._ts.get_data(t0, t1)
        return data

    @property
    def identity(self) -> str:
        return f"ephys:{self._path}:{self._stream_id}:{self._channel_idx}"


# ---------------------------------------------------------------------------
# Adapter: use any TimeseriesSource as a SpectrogramSource
# ---------------------------------------------------------------------------


class SpectrogramSourceAdapter:
    """Wraps any ``TimeseriesSource`` for use as a ``SpectrogramSource``.

    Extracts a single channel if the source is multi-channel.
    """

    supports_noise_reduction = False

    def __init__(self, source: TimeseriesSource, *, channel: int = 0):
        self._source = source
        self._channel = channel

    @property
    def rate(self) -> float:
        return self._source.sampling_rate

    @property
    def duration(self) -> float:
        return self._source.time_range.duration

    @property
    def timeseries_source(self) -> TimeseriesSource:
        return self._source

    def get_data(self, t0: float, t1: float) -> np.ndarray:
        _, data = self._source.get_data(t0, t1)
        if data.ndim > 1:
            ch = min(self._channel, data.shape[1] - 1)
            data = data[:, ch]
        return data

    @property
    def identity(self) -> str:
        return self._source.identity


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_audio_source(app_state) -> AudioFileSource | None:
    """Build an AudioFileSource from the current app_state."""
    audio_path = getattr(app_state, 'audio_path', None)
    if not audio_path:
        return None
    _, channel_idx = app_state.get_audio_source()
    try:
        return AudioFileSource(audio_path, channel_idx)
    except ValueError:
        return None


def build_audio_source_from_alignment(app_state) -> AudioFileSource | None:
    """Build an AudioFileSource using the trial alignment if available."""
    alignment = getattr(app_state, 'trial_alignment', None)
    if alignment and "audio" in alignment.continuous:
        ts = alignment.continuous["audio"]
        return SpectrogramSourceAdapter(ts)
    return build_audio_source(app_state)


def build_xarray_source(
    da: xr.DataArray,
    time_coords: np.ndarray,
    variable_name: str,
    ds_kwargs: dict,
) -> XarraySource:
    """Build an XarraySource from a 1-D DataArray."""
    kwargs_str = str(sorted(ds_kwargs.items()))
    ds_kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:12]
    return XarraySource(da, time_coords, variable_name, ds_kwargs_hash)
