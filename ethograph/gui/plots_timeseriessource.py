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

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import pandas as pd
    from ethograph.utils.trialtree import TrialTree


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
# Alignment builder — handles per-trial and session-wide streams
# ---------------------------------------------------------------------------


def _try_add_regular_source(
    alignment: TrialAlignment, name: str, loader, start_time: float
) -> None:
    """Add a RegularTimeseriesSource to alignment when loader is valid."""
    if loader is not None and len(loader) > 0:
        alignment.continuous[name] = RegularTimeseriesSource(
            name, loader, start_time=start_time
        )


def build_trial_alignment(
    dt: TrialTree,
    trial_id,
    ds: xr.Dataset,
    *,
    video_folder: str | None = None,
    audio_folder: str | None = None,
    cameras_sel: str | None = None,
) -> TrialAlignment:
    """Build a :class:`TrialAlignment` from a :class:`TrialTree`.

    Handles three media layouts transparently:

    * **Per-trial files** — each trial has its own file starting at t=0
      (most common; DLC/video recordings per trial).
    * **Session-wide files** — one long file covers all trials (e.g. a
      continuous ephys recording).
    * **Mixed** — e.g. per-trial videos with a session-wide ephys file.

    All timestamps in the returned alignment are **trial-relative**
    (t = 0 is the start of this trial).  For session-wide files the
    source's ``start_time`` is shifted by ``-trial_start_abs`` so that
    ``source.get_data(0, duration)`` reads the correct slice of the file.

    Parameters
    ----------
    dt
        TrialTree with session table and media references.
    trial_id
        Trial identifier.
    ds
        xarray Dataset for this trial (xarray-based sources).
    video_folder
        Directory containing video files.
    audio_folder
        Directory containing audio files.
    cameras_sel
        Selected camera label (resolves the primary video file).
    """
    from ethograph.utils.xr_utils import get_time_coord

    trial_stop_abs = dt.get_stop_time(trial_id)

    alignment = TrialAlignment(trial_id=str(trial_id))

    for var_name in ds.data_vars:
        da = ds[var_name]
        tc = get_time_coord(da)
        if tc is None:
            continue
        if da.attrs.get("type", "") in ("features", "colors", ""):
            try:
                alignment.continuous[var_name] = ArrayTimeseriesSource.from_xarray(
                    da, name=var_name
                )
            except (ValueError, IndexError):
                continue

    # Priority: session table stop_time > xarray data > 0
    if trial_stop_abs is not None:
        trial_duration = trial_stop_abs - dt.get_start_time(trial_id)
    elif alignment.continuous:
        trial_duration = max(s.time_range.end_s for s in alignment.continuous.values())
    else:
        trial_duration = 0.0

    if trial_duration > 0:
        alignment.extra_ranges.append(TimeRange(0.0, trial_duration))

    # Video
    video_file = dt.get_media(trial_id, "video", cameras_sel)
    if video_file:
        video_path = os.path.join(video_folder, video_file) if (video_folder and not os.path.isabs(video_file)) else video_file
        video_start = dt.get_display_start(trial_id, "video") or 0.0
        alignment.video_offset = video_start
        try:
            from napari_pyav._reader import FastVideoReader
            reader = FastVideoReader(video_path, read_format="rgb24")
            n_frames = reader.shape[0]
            fps = float(reader.stream.guessed_rate)
            if n_frames > 0 and fps > 0:
                # Only per-trial files contribute a range; session-wide covered by trial_duration.
                if dt.stream_is_per_trial("video"):
                    alignment.extra_ranges.append(
                        TimeRange(video_start, video_start + n_frames / fps)
                    )
        except Exception:
            pass

    # Audio
    audio_devices = dt.devices("audio")
    audio_file = dt.get_media(trial_id, "audio", audio_devices[0] if audio_devices else None)
    if audio_file and audio_folder:
        audio_path = os.path.join(audio_folder, audio_file) if not os.path.isabs(audio_file) else audio_file
        audio_start = dt.get_display_start(trial_id, "audio")
        if audio_start is not None:
            try:
                from ethograph.gui.plots_spectrogram import SharedAudioCache
                _try_add_regular_source(alignment, "audio", SharedAudioCache.get_loader(audio_path), audio_start)
            except Exception:
                pass

    # Ephys
    ephys_file = dt.get_media(trial_id, "ephys")
    if ephys_file:
        ephys_start = dt.get_display_start(trial_id, "ephys")
        if ephys_start is not None:
            try:
                from ethograph.gui.plots_ephystrace import get_loader as get_ephys_loader
                _try_add_regular_source(alignment, "ephys", get_ephys_loader(ephys_file, "0"), ephys_start)
            except Exception:
                pass

    return alignment
