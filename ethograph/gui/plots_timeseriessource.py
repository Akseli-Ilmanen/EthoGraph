"""Data source abstractions for time-aligned multi-modal data.

Key types:
    TimeRange               -- Immutable time interval with set operations
    TimeseriesSource        -- Protocol for continuous sampled data (audio, ephys)
    RegularTimeseriesSource -- Uniform-rate file-based loader wrapper
    TrialAlignment          -- Trial time range + video offset
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import xarray as xr


if TYPE_CHECKING:
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
# Protocol for continuous sampled data (audio, ephys)
# ---------------------------------------------------------------------------


@runtime_checkable
class TimeseriesSource(Protocol):
    """Uniform interface for slice-loading large files (audio, ephys).

    Used by AudioTracePlot and EphysTracePlot so they can call
    ``get_data(t0, t1)`` without loading the whole file into memory.
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
        """Return ``(timestamps, data)`` for the window ``[t0, t1]``."""
        ...


# ---------------------------------------------------------------------------
# Continuous source: uniform sampling (file-based loaders)
# ---------------------------------------------------------------------------


class RegularTimeseriesSource:
    """Wraps a file-based loader with uniform sampling rate.

    Compatible with ``AudioLoader`` (audioio), ``EphysLoader``,
    ``MemmapLoader``, or any object exposing ``rate``, ``__len__``,
    and ``__getitem__``.

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
# Trial alignment: time context for one trial
# ---------------------------------------------------------------------------


@dataclass
class TrialAlignment:
    """Time context for a single trial.

    Parameters
    ----------
    trial_id
        Trial identifier.
    trial_range
        Effective time window in seconds (t=0 is trial start).
        ``None`` when no source could determine the duration.
    video_offset
        Added to ``frame / fps`` to get trial-relative time.
        Zero for per-trial video files; negative for session-wide files.
    ephys_offset
        Session-absolute start of this trial in the ephys file (seconds).
        Used to convert trial-relative display times to file sample indices:
        ``t_file = t_trial + ephys_offset``.
    """

    trial_id: str
    trial_range: TimeRange | None = None
    video_offset: float = 0.0
    ephys_offset: float = 0.0

    def summary(self) -> str:
        lines = [f"TrialAlignment(trial={self.trial_id!r})"]
        if self.trial_range:
            lines.append(f"  range:         {self.trial_range}")
        if self.video_offset:
            lines.append(f"  video_offset:  {self.video_offset:.3f}s")
        if self.ephys_offset:
            lines.append(f"  ephys_offset:  {self.ephys_offset:.3f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alignment builder
# ---------------------------------------------------------------------------


def compute_trial_alignment(
    dt: TrialTree,
    trial_id,
    ds: xr.Dataset,
    *,
    video_folder: str | None = None,
    audio_folder: str | None = None,
    cameras_sel: str | None = None,
) -> TrialAlignment:
    """Compute a :class:`TrialAlignment` for one trial.

    Priority for trial duration:
    1. Session table ``stop_time`` (most authoritative).
    2. Last timestamp of any xarray feature variable.
    3. Video length (frames / fps).
    4. Per-trial audio length.
    """

    video_path: str | None = None
    video_offset = 0.0
    video_file = dt.get_media(trial_id, "video", cameras_sel)
    if video_file:
        video_path = (
            os.path.join(video_folder, video_file)
            if (video_folder and not os.path.isabs(video_file))
            else video_file
        )
        video_offset = dt.source_start_time(trial_id, "video")

    audio_devices = dt.devices("audio")
    audio_device = audio_devices[0] if audio_devices else None
    audio_file = dt.get_media(trial_id, "audio", audio_device)
    audio_path: str | None = None
    if audio_file and audio_folder:
        audio_path = (
            os.path.join(audio_folder, audio_file)
            if not os.path.isabs(audio_file)
            else audio_file
        )

    ephys_offset = 0.0
    try:
        ephys_offset = dt.start_time(str(trial_id))
    except (KeyError, AttributeError):
        pass

    trial_end = _compute_trial_end(
        dt, trial_id, ds, video_path, video_offset, audio_path
    )
    trial_range = TimeRange(0.0, trial_end) if trial_end and trial_end > 0 else None
    return TrialAlignment(
        trial_id=str(trial_id),
        trial_range=trial_range,
        video_offset=video_offset,
        ephys_offset=ephys_offset,
    )


def _compute_trial_end(
    dt: TrialTree,
    trial_id,
    ds: xr.Dataset,
    video_path: str | None,
    video_offset: float,
    audio_path: str | None,
) -> float | None:
    """Return trial duration in seconds using the highest-priority source."""
    from ethograph.utils.xr_utils import get_time_coord

    # 1. Session stop_time
    try:
        stop = dt.stop_time(trial_id)
        if stop is not None:
            return stop - dt.start_time(trial_id)
    except (KeyError, AttributeError):
        pass

    # 2. xarray features (always trial-scoped)
    for var_name in ds.data_vars:
        da = ds[var_name]
        if da.attrs.get("type", "") in ("features", "colors", ""):
            tc = get_time_coord(da)
            if tc is not None:
                vals = tc if not hasattr(tc, "values") else tc.values
                if len(vals) > 0:
                    t_end = float(vals[-1])
                    if t_end > 0:
                        return t_end

    # 3. Video
    if video_path:
        try:
            from napari_pyav._reader import FastVideoReader
            reader = FastVideoReader(video_path, read_format="rgb24")
            n_frames = reader.shape[0]
            fps = float(reader.stream.guessed_rate)
            if n_frames > 0 and fps > 0:
                return video_offset + n_frames / fps
        except Exception:
            pass

    # 4. Audio (per-trial only — session-wide sources have large negative start)
    if audio_path:
        try:
            from ethograph.gui.plots_spectrogram import SharedAudioCache
            loader = SharedAudioCache.get_loader(audio_path)
            audio_start = dt.source_start_time(trial_id, "audio")
            if loader is not None and len(loader) > 0 and audio_start >= -0.5:
                return len(loader) / loader.rate
        except Exception:
            pass

    return None
