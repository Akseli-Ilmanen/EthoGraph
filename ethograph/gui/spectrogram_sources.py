"""Source-agnostic data providers for the spectrogram pipeline."""

from __future__ import annotations

import hashlib
import os
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr

from .plots_spectrogram import SharedAudioCache


@runtime_checkable
class SpectrogramSource(Protocol):
    rate: float
    duration: float
    supports_noise_reduction: bool

    def get_data(self, t0: float, t1: float) -> np.ndarray: ...

    @property
    def identity(self) -> str: ...


class AudioFileSource:
    """Wraps SharedAudioCache audio loader for spectrogram consumption."""

    supports_noise_reduction = True

    def __init__(self, audio_path: str, channel_idx: int = 0):
        self._audio_path = audio_path
        self._channel_idx = channel_idx
        self._loader = SharedAudioCache.get_loader(audio_path)
        if self._loader is None:
            raise ValueError(f"Failed to load audio: {audio_path}")

    @property
    def rate(self) -> float:
        return self._loader.rate

    @property
    def duration(self) -> float:
        return len(self._loader) / self._loader.rate

    def get_data(self, t0: float, t1: float) -> np.ndarray:
        i0 = int(t0 * self.rate)
        i1 = int(t1 * self.rate)
        i0 = max(0, i0)
        i1 = min(len(self._loader), i1)
        if i1 <= i0:
            return np.array([], dtype=np.float64)
        data = self._loader[i0:i1]
        if data.ndim > 1:
            ch = min(self._channel_idx, data.shape[1] - 1)
            data = data[:, ch]
        return np.asarray(data, dtype=np.float64)

    @property
    def identity(self) -> str:
        return f"{self._audio_path}:{self._channel_idx}"


class XarraySource:
    """Wraps an xarray DataArray (already 1-D after dimension selection) for spectrogram."""

    supports_noise_reduction = False

    def __init__(
        self,
        da: xr.DataArray,
        time_coords: np.ndarray,
        variable_name: str,
        ds_kwargs_hash: str,
    ):
        self._da = da
        self._time = np.asarray(time_coords, dtype=np.float64)
        self._variable_name = variable_name
        self._ds_kwargs_hash = ds_kwargs_hash

        dt = np.median(np.diff(self._time))
        self._rate = 1.0 / dt
        self._duration = float(self._time[-1] - self._time[0])

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def duration(self) -> float:
        return self._duration

    def get_data(self, t0: float, t1: float) -> np.ndarray:
        mask = (self._time >= t0) & (self._time <= t1)
        values = np.asarray(self._da, dtype=np.float64)
        return values[mask]

    @property
    def identity(self) -> str:
        return f"xarray:{self._variable_name}:{self._ds_kwargs_hash}"


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
