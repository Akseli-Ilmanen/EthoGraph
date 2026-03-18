"""Neurosift-inspired data source abstractions for time-aligned multi-modal data.

Provides a uniform interface for accessing data from different sources
(xarray variables, audio files, ephys recordings, spike data) that may
have different sampling rates but share a common time axis in seconds.

Key types:
    TimeRange                -- Immutable time interval with set operations
    TimeseriesSource         -- Protocol for continuous sampled data
    EventSource              -- Protocol for discrete event data
    RegularTimeseriesSource  -- Uniform sampling (audio/ephys file loaders)
    ArrayTimeseriesSource    -- Explicit timestamps (xarray variables)
    SpikeEventSource         -- Spike times with optional cluster/channel info
    TrialAlignment           -- All sources for a trial + global time range
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Core value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeRange:
    """Immutable time interval in seconds."""

    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s

    def overlaps(self, other: TimeRange) -> bool:
        return self.start_s < other.end_s and other.start_s < self.end_s

    def union(self, other: TimeRange) -> TimeRange:
        return TimeRange(min(self.start_s, other.start_s), max(self.end_s, other.end_s))

    def intersect(self, other: TimeRange) -> TimeRange | None:
        lo = max(self.start_s, other.start_s)
        hi = min(self.end_s, other.end_s)
        return TimeRange(lo, hi) if lo < hi else None

    def contains(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def __repr__(self) -> str:
        return f"TimeRange({self.start_s:.3f}s .. {self.end_s:.3f}s, dur={self.duration:.3f}s)"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class TimeseriesSource(Protocol):
    """Uniform interface for continuous time-aligned data.

    Inspired by Neurosift's TimeseriesTimestampsClient + TimeseriesClient.
    Each source keeps its native sampling rate -- no resampling happens.
    """

    @property
    def name(self) -> str: ...

    @property
    def time_range(self) -> TimeRange: ...

    @property
    def sampling_rate(self) -> float: ...

    @property
    def n_channels(self) -> int: ...

    @property
    def n_samples(self) -> int: ...

    @property
    def identity(self) -> str: ...

    def index_for_time(self, t: float) -> int: ...

    def time_for_index(self, i: int) -> float: ...

    def get_data(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(timestamps, data)`` for the window ``[t0, t1]``.

        Returns
        -------
        timestamps : 1-D float64 array of times in seconds
        data       : 1-D (single channel) or 2-D (samples x channels) array
        """
        ...


@runtime_checkable
class EventSource(Protocol):
    """Interface for discrete event data (spikes, changepoints, labels)."""

    @property
    def name(self) -> str: ...

    @property
    def time_range(self) -> TimeRange: ...

    @property
    def identity(self) -> str: ...

    def get_events(self, t0: float, t1: float) -> np.ndarray:
        """Return event times in ``[t0, t1]`` as a 1-D float64 array."""
        ...


# ---------------------------------------------------------------------------
# Continuous source: uniform sampling (file-based loaders)
# ---------------------------------------------------------------------------


class RegularTimeseriesSource:
    """Wraps a file-based loader with uniform sampling rate.

    Compatible with ``AudioLoader`` (audioio), ``EphysLoader``,
    ``MemmapLoader``, ``NWBLoader``, or any object exposing
    ``rate``, ``__len__``, and ``__getitem__``.

    Parameters
    ----------
    name
        Human-readable label (e.g. ``"audio"``, ``"ephys"``).
    loader
        Object with ``rate: float``, ``__len__() -> int``,
        ``__getitem__(slice) -> ndarray``.
    start_time
        Time in seconds corresponding to sample index 0.
    channel
        If set, extract a single channel from multi-channel data.
    """

    def __init__(
        self,
        name: str,
        loader,
        *,
        start_time: float = 0.0,
        channel: int | None = None,
    ):
        self._name = name
        self._loader = loader
        self._start_time = start_time
        self._channel = channel
        self._rate = float(loader.rate)
        self._n_samples = len(loader)

        if channel is not None:
            self._n_channels = 1
        elif hasattr(loader, "n_channels"):
            self._n_channels = loader.n_channels
        else:
            probe = loader[0 : min(2, self._n_samples)]
            self._n_channels = probe.shape[1] if probe.ndim > 1 else 1

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_range(self) -> TimeRange:
        end = self._start_time + self._n_samples / self._rate
        return TimeRange(self._start_time, end)

    @property
    def sampling_rate(self) -> float:
        return self._rate

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def identity(self) -> str:
        loader_id = getattr(self._loader, "filepath", id(self._loader))
        ch = f":{self._channel}" if self._channel is not None else ""
        return f"regular:{self._name}:{loader_id}{ch}"

    def index_for_time(self, t: float) -> int:
        idx = round((t - self._start_time) * self._rate)
        return max(0, min(idx, self._n_samples - 1))

    def time_for_index(self, i: int) -> float:
        return self._start_time + i / self._rate

    def get_data(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        i0 = max(0, int((t0 - self._start_time) * self._rate))
        i1 = min(self._n_samples, int((t1 - self._start_time) * self._rate) + 1)
        if i1 <= i0:
            empty = np.array([], dtype=np.float64)
            return empty, empty

        data = self._loader[i0:i1]
        if self._channel is not None and data.ndim > 1:
            ch = min(self._channel, data.shape[1] - 1)
            data = data[:, ch]

        timestamps = self._start_time + np.arange(i0, i1) / self._rate
        return timestamps.astype(np.float64), np.asarray(data, dtype=np.float64)


# ---------------------------------------------------------------------------
# Continuous source: explicit timestamps (xarray / in-memory arrays)
# ---------------------------------------------------------------------------


class ArrayTimeseriesSource:
    """Wraps explicit timestamp + data arrays.

    Handles both regular and irregular sampling transparently via
    ``np.searchsorted``.  Use the :meth:`from_xarray` factory to build
    directly from an ``xr.DataArray``.

    Parameters
    ----------
    name
        Human-readable label.
    timestamps
        1-D array of sample times in seconds.
    data
        1-D ``(n_samples,)`` or 2-D ``(n_samples, n_channels)`` array.
    identity_hint
        Optional string for cache-invalidation identity.
    """

    def __init__(
        self,
        name: str,
        timestamps: np.ndarray,
        data: np.ndarray,
        *,
        identity_hint: str = "",
    ):
        self._name = name
        self._timestamps = np.asarray(timestamps, dtype=np.float64)
        self._data = np.asarray(data)
        self._identity_hint = identity_hint

        if len(self._timestamps) >= 2:
            dt = np.median(np.diff(self._timestamps))
            self._rate = 1.0 / dt if dt > 0 else 1.0
        else:
            self._rate = 1.0

        self._n_channels = self._data.shape[1] if self._data.ndim > 1 else 1

    @classmethod
    def from_xarray(
        cls,
        da: xr.DataArray,
        name: str | None = None,
        sel_kwargs: dict | None = None,
    ) -> ArrayTimeseriesSource:
        """Build from an xarray DataArray, auto-extracting time coordinate.

        Parameters
        ----------
        da
            Source DataArray (any number of dims; must have a time coord).
        name
            Override for the source name (defaults to ``da.name``).
        sel_kwargs
            Optional dimension selections passed to ``sel_valid()``.
            When given, data is sliced before wrapping.
        """
        from ethograph.utils.xr_utils import get_time_coord, sel_valid

        tc = get_time_coord(da)
        if tc is None:
            raise ValueError(f"No time coordinate found in DataArray '{da.name}'")
        timestamps = np.asarray(tc.values, dtype=np.float64)

        if sel_kwargs:
            data, filt = sel_valid(da, sel_kwargs)
            hint = str(sorted(filt.items()))
        else:
            time_dim = tc.name
            other_dims = [d for d in da.dims if d != time_dim]
            squeezed = da
            for d in other_dims:
                if da.sizes[d] == 1:
                    squeezed = squeezed.squeeze(d)
            if time_dim in squeezed.dims:
                squeezed = squeezed.transpose(time_dim, ...)
            data = squeezed.values
            hint = ""

        source_name = name or str(da.name) or "xarray"
        return cls(source_name, timestamps, data, identity_hint=f"{source_name}:{hint}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_range(self) -> TimeRange:
        if len(self._timestamps) == 0:
            return TimeRange(0.0, 0.0)
        return TimeRange(float(self._timestamps[0]), float(self._timestamps[-1]))

    @property
    def sampling_rate(self) -> float:
        return self._rate

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def n_samples(self) -> int:
        return len(self._timestamps)

    @property
    def identity(self) -> str:
        return f"array:{self._identity_hint or self._name}:{self.n_samples}"

    def index_for_time(self, t: float) -> int:
        idx = int(np.searchsorted(self._timestamps, t))
        return max(0, min(idx, len(self._timestamps) - 1))

    def time_for_index(self, i: int) -> float:
        i = max(0, min(i, len(self._timestamps) - 1))
        return float(self._timestamps[i])

    def get_data(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        i0 = int(np.searchsorted(self._timestamps, t0, side="left"))
        i1 = int(np.searchsorted(self._timestamps, t1, side="right"))
        if i1 <= i0:
            empty_t = np.array([], dtype=np.float64)
            empty_d = np.array([], dtype=self._data.dtype)
            return empty_t, empty_d
        return self._timestamps[i0:i1].copy(), self._data[i0:i1].copy()


# ---------------------------------------------------------------------------
# Event source: spike times
# ---------------------------------------------------------------------------


class SpikeEventSource:
    """Spike times with optional per-spike cluster ID and channel.

    Data is sorted by time on construction for efficient ``searchsorted``
    lookups.

    Parameters
    ----------
    name
        Human-readable label (e.g. ``"spikes"``).
    spike_times
        1-D array of spike times in seconds.
    cluster_ids
        Optional 1-D array of cluster IDs per spike.
    channels
        Optional 1-D array of channel indices per spike.
    """

    def __init__(
        self,
        name: str,
        spike_times: np.ndarray,
        *,
        cluster_ids: np.ndarray | None = None,
        channels: np.ndarray | None = None,
    ):
        self._name = name
        self._times = np.asarray(spike_times, dtype=np.float64)
        self.cluster_ids = np.asarray(cluster_ids) if cluster_ids is not None else None
        self.channels = np.asarray(channels) if channels is not None else None

        order = np.argsort(self._times)
        self._times = self._times[order]
        if self.cluster_ids is not None:
            self.cluster_ids = self.cluster_ids[order]
        if self.channels is not None:
            self.channels = self.channels[order]

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_range(self) -> TimeRange:
        if len(self._times) == 0:
            return TimeRange(0.0, 0.0)
        return TimeRange(float(self._times[0]), float(self._times[-1]))

    @property
    def identity(self) -> str:
        return f"spikes:{self._name}:{len(self._times)}"

    def get_events(self, t0: float, t1: float) -> np.ndarray:
        i0 = np.searchsorted(self._times, t0, side="left")
        i1 = np.searchsorted(self._times, t1, side="right")
        return self._times[i0:i1]

    def get_events_with_metadata(
        self, t0: float, t1: float
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return ``(times, cluster_ids, channels)`` for events in ``[t0, t1]``."""
        i0 = np.searchsorted(self._times, t0, side="left")
        i1 = np.searchsorted(self._times, t1, side="right")
        times = self._times[i0:i1]
        clusters = self.cluster_ids[i0:i1] if self.cluster_ids is not None else None
        chans = self.channels[i0:i1] if self.channels is not None else None
        return times, clusters, chans


# ---------------------------------------------------------------------------
# Trial alignment container
# ---------------------------------------------------------------------------


@dataclass
class TrialAlignment:
    """All data sources discovered for a single trial.

    Neurosift-inspired: each source has independent time bounds and sampling
    rate.  The :attr:`global_range` is the union of all source time ranges.

    Examples
    --------
    >>> alignment = TrialAlignment(trial_id="1")
    >>> alignment.continuous["speed"] = ArrayTimeseriesSource.from_xarray(ds["speed"])
    >>> alignment.continuous["audio"] = RegularTimeseriesSource("audio", loader)
    >>> alignment.events["spikes"] = SpikeEventSource("spikes", spike_times)
    >>> print(alignment.summary())
    """

    trial_id: str
    video_offset: float = 0.0
    continuous: dict[str, TimeseriesSource] = field(default_factory=dict)
    events: dict[str, EventSource] = field(default_factory=dict)
    extra_ranges: list[TimeRange] = field(default_factory=list)

    @property
    def global_range(self) -> TimeRange | None:
        ranges = [s.time_range for s in self.continuous.values()]
        ranges += [s.time_range for s in self.events.values()]
        ranges += self.extra_ranges
        valid = [r for r in ranges if r.duration > 0]
        if not valid:
            return None
        result = valid[0]
        for r in valid[1:]:
            result = result.union(r)
        return result

    @property
    def all_sources(self) -> dict[str, TimeseriesSource | EventSource]:
        return {**self.continuous, **self.events}

    def summary(self) -> str:
        lines = [f"TrialAlignment(trial={self.trial_id!r})"]
        gr = self.global_range
        if gr:
            lines.append(f"  Global: {gr}")
        if self.continuous:
            lines.append("  Continuous:")
            for name, src in self.continuous.items():
                lines.append(
                    f"    {name:20s}  {src.time_range}  "
                    f"rate={src.sampling_rate:>10.1f} Hz  "
                    f"ch={src.n_channels}  n={src.n_samples}"
                )
        if self.events:
            lines.append("  Events:")
            for name, src in self.events.items():
                lines.append(f"    {name:20s}  {src.time_range}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discovery: build TrialAlignment from a trial dataset + file paths
# ---------------------------------------------------------------------------


def discover_trial_sources(
    trial_id: str,
    ds: xr.Dataset,
    *,
    video_offset: float = 0.0,
    audio_path: str | None = None,
    audio_channel: int = 0,
    audio_offset: float = 0.0,
    audio_timestamps: np.ndarray | None = None,
    ephys_path: str | None = None,
    ephys_stream_id: str = "0",
    ephys_offset: float = 0.0,
    ephys_timestamps: np.ndarray | None = None,
    spike_times: np.ndarray | None = None,
    spike_clusters: np.ndarray | None = None,
    spike_channels: np.ndarray | None = None,
    label_intervals: pd.DataFrame | None = None,
) -> TrialAlignment:
    """Build a :class:`TrialAlignment` by discovering all available sources.

    Scans the dataset for feature variables with time coordinates and
    optionally adds file-based audio/ephys sources and spike event data.

    For audio and ephys streams, alignment can be specified in two ways
    (mutually exclusive per stream, timestamps take priority):

    1. **Scalar offset** — a single ``start_time`` shift applied via
       :class:`RegularTimeseriesSource`.
    2. **Aligned timestamps** — an explicit 1-D array of corrected
       sample times (e.g. computed via TTL interpolation / neuroconv).
       When provided, the stream is wrapped in an
       :class:`ArrayTimeseriesSource` instead, giving per-sample
       irregular timing.

    Parameters
    ----------
    trial_id
        Trial identifier.
    ds
        The trial's xarray Dataset.
    video_offset
        Video stream offset in seconds (stored on TrialAlignment).
    audio_path
        Path to audio file.
    audio_channel
        Channel index within the audio file.
    audio_offset
        Scalar offset for audio (ignored when *audio_timestamps* given).
    audio_timestamps
        Explicit aligned timestamps for every audio sample.
    ephys_path
        Path to ephys recording file.
    ephys_stream_id
        Stream identifier for multi-stream ephys formats.
    ephys_offset
        Scalar offset for ephys (ignored when *ephys_timestamps* given).
    ephys_timestamps
        Explicit aligned timestamps for every ephys sample.
    spike_times
        1-D array of spike times in seconds.
    spike_clusters, spike_channels
        Optional per-spike metadata arrays.
    label_intervals
        DataFrame with ``onset_s``, ``offset_s``, ``labels``, ``individual``.
    """
    from ethograph.utils.xr_utils import get_time_coord

    alignment = TrialAlignment(trial_id=trial_id, video_offset=video_offset)

    for var_name in ds.data_vars:
        da = ds[var_name]
        tc = get_time_coord(da)
        if tc is None:
            continue
        var_type = da.attrs.get("type", "")
        if var_type in ("features", "colors", ""):
            try:
                alignment.continuous[var_name] = ArrayTimeseriesSource.from_xarray(
                    da, name=var_name
                )
            except (ValueError, IndexError):
                continue

    if audio_path:
        try:
            from ethograph.gui.plots_spectrogram import SharedAudioCache

            loader = SharedAudioCache.get_loader(audio_path)
            if loader is not None and len(loader) > 0:
                if audio_timestamps is not None and len(audio_timestamps) > 0:
                    data = loader[:len(audio_timestamps)]
                    if data.ndim > 1:
                        ch = min(audio_channel, data.shape[1] - 1)
                        data = data[:, ch]
                    alignment.continuous["audio"] = ArrayTimeseriesSource(
                        "audio", audio_timestamps, data
                    )
                else:
                    alignment.continuous["audio"] = RegularTimeseriesSource(
                        "audio",
                        loader,
                        start_time=audio_offset,
                        channel=audio_channel,
                    )
        except (ImportError, OSError):
            pass

    if ephys_path:
        try:
            from ethograph.gui.plots_ephystrace import get_loader as get_ephys_loader

            loader = get_ephys_loader(ephys_path, ephys_stream_id)
            if loader is not None and len(loader) > 0:
                if ephys_timestamps is not None and len(ephys_timestamps) > 0:
                    data = loader[:len(ephys_timestamps)]
                    alignment.continuous["ephys"] = ArrayTimeseriesSource(
                        "ephys", ephys_timestamps, data
                    )
                else:
                    alignment.continuous["ephys"] = RegularTimeseriesSource(
                        "ephys", loader, start_time=ephys_offset
                    )
        except (ImportError, OSError):
            pass

    if spike_times is not None and len(spike_times) > 0:
        alignment.events["spikes"] = SpikeEventSource(
            "spikes",
            spike_times,
            cluster_ids=spike_clusters,
            channels=spike_channels,
        )

    if "audio_cp_onsets" in ds and "audio_cp_offsets" in ds:
        onsets = ds["audio_cp_onsets"].values.astype(np.float64)
        offsets = ds["audio_cp_offsets"].values.astype(np.float64)
        if len(onsets) > 0:
            alignment.extra_ranges.append(TimeRange(float(onsets.min()), float(offsets.max())))

    if label_intervals is not None:
        import pandas as pd

        if isinstance(label_intervals, pd.DataFrame) and not label_intervals.empty:
            alignment.extra_ranges.append(
                TimeRange(float(label_intervals["onset_s"].min()), float(label_intervals["offset_s"].max()))
            )

    return alignment
