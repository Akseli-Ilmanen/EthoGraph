"""Source-agnostic data providers for the spectrogram and plot pipelines.

Bridges the GUI-specific SpectrogramSource protocol with the general-purpose
TimeseriesSource abstractions in plots_timeseriessource.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr

from ethograph.gui.plots_timeseriessource import (
    ArrayTimeseriesSource,
    RegularTimeseriesSource,
    TimeseriesSource,
)

from .plots_spectrogram import SharedAudioCache


@runtime_checkable
class SpectrogramSource(Protocol):
    rate: float
    duration: float

    def get_data(self, t0: float, t1: float) -> np.ndarray: ...

    @property
    def identity(self) -> str: ...


class SpectrogramSourceAdapter:
    """Wraps any ``TimeseriesSource`` for use as a ``SpectrogramSource``.

    Extracts a single channel if the source is multi-channel.
    """

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


def build_audio_source(app_state) -> SpectrogramSourceAdapter | None:
    """Build a SpectrogramSourceAdapter for audio from the current app_state."""
    audio_path = getattr(app_state, 'audio_path', None)
    if not audio_path:
        return None
    _, channel_idx = app_state.get_audio_source()
    loader = SharedAudioCache.get_loader(audio_path)
    if loader is None:
        return None
    ts = RegularTimeseriesSource("audio", loader, channel=channel_idx)
    return SpectrogramSourceAdapter(ts, channel=channel_idx)


def build_audio_source_from_alignment(app_state) -> SpectrogramSourceAdapter | None:
    """Build a SpectrogramSourceAdapter using the trial alignment if available."""
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
) -> SpectrogramSourceAdapter:
    """Build a SpectrogramSourceAdapter from a 1-D DataArray."""
    data = np.asarray(da, dtype=np.float64)
    np.nan_to_num(data, copy=False, nan=0.0)
    ts = ArrayTimeseriesSource(
        variable_name,
        np.asarray(time_coords, dtype=np.float64),
        data,
        identity_hint=f"{variable_name}:{str(sorted(ds_kwargs.items()))}",
    )
    return SpectrogramSourceAdapter(ts)
