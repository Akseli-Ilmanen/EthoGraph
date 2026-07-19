"""Time model: core types for time navigation and trial alignment.

Canonical home for types previously in ``gui.plots_timeseriessource``.

Key types:
    TimeRange               -- Immutable time interval with set operations
    RestrictionWindow       -- Display window for trial/label/sequence/session navigation
    TrialVideoBounds        -- Trial-local bounds + video offset
    TimeSource              -- Protocol for any time-aligned data source
    SourceCollection        -- Registry of sources with union/intersection queries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    pass


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


@dataclass(frozen=True)
class RestrictionWindow:
    """Describes the currently active display window.

    Modes:
        ``"session"``      — full session range (all trials + inter-trial gaps)
        ``"trial"``        — one trial's time range (start + stop known)
        ``"trial_start"``  — trial start to next trial start (stop unknown)
        ``"label"``        — a single label instance (with context padding)
        ``"sequence"``     — a matched label sequence span
        ``"fixed"``        — fixed-size window anchored at an interval start
    """

    mode: str  # "session" | "trial" | "trial_start" | "label" | "sequence" | "fixed"
    time_range: TimeRange  # effective display window (including extra context)
    core_range: TimeRange  # the actual interval (without extra context)
    trial_id: int | str | None = None
    label_info: dict | None = None
    sequence_info: dict | None = None


# ---------------------------------------------------------------------------
# Trial bounds: navigation context for one trial
# ---------------------------------------------------------------------------


@dataclass
class TrialVideoBounds:
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
    """

    trial_id: str
    trial_range: TimeRange | None = None
    video_offset: float = 0.0

    def summary(self) -> str:
        lines = [f"  TrialVideoBounds(trial={self.trial_id!r})"]
        if self.trial_range:
            lines.append(f"    effective range: {self.trial_range}")
        if self.video_offset:
            lines.append(f"    video_offset:   {self.video_offset:.3f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alignment builder
# ---------------------------------------------------------------------------


def compute_trial_video_bounds(
    nwb_alignment,
    trial_id,
    ds: xr.Dataset,
    *,
    video_folder: str | None = None,
    audio_folder: str | None = None,
    cameras_sel: str | None = None,
    source_collection: SourceCollection | None = None,
) -> TrialVideoBounds:
    """Compute :class:`TrialVideoBounds` for one trial.

    Priority for trial duration:
    1. ``nwb_alignment.stop_time(trial)`` — authoritative when available
    2. Video file duration (probed from file)
    3. ``source_collection.union_range`` — last resort
    """
    sio = nwb_alignment
    video_path = sio.resolve_media_path(
        trial_id,
        "video",
        device=cameras_sel,
        fallback_folder=video_folder,
    )
    video_offset = sio.stream_offset_for_trial(trial_id, "video", cameras_sel) if video_path else 0.0

    # 1. Alignment timing (authoritative)
    trial_end: float | None = None
    stop = sio.stop_time(trial_id)
    if stop is not None:
        trial_end = stop - sio.start_time(trial_id)

    # 2. Fallbacks only when alignment has no stop time
    if trial_end is None or trial_end <= 0:
        trial_end = _resolve_trial_end(video_path, video_offset, source_collection)

    trial_range = TimeRange(0.0, trial_end) if trial_end and trial_end > 0 else None

    return TrialVideoBounds(
        trial_id=str(trial_id),
        trial_range=trial_range,
        video_offset=video_offset,
    )


def _resolve_trial_end(
    video_path: str | None,
    video_offset: float,
    source_collection: SourceCollection | None,
) -> float | None:
    """Fallback trial-end resolution when alignment has no stop time.

    Priority: video file duration, then source collection union range.
    """
    # 1. Video length
    if video_path:
        try:
            from ethograph.gui.video_manager import probe_video

            probe = probe_video(video_path)
            if probe.nframes > 0 and probe.fps > 0:
                return video_offset + probe.nframes / probe.fps
        except Exception:
            pass

    # 2. SourceCollection union_range (last resort)
    if source_collection is not None:
        ur = source_collection.union_range
        if ur is not None and ur.duration > 0:
            return ur.end_s

    return None


# ---------------------------------------------------------------------------
# TimeSource protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TimeSource(Protocol):
    """Navigation-layer metadata for one time-aligned data stream.

    Used by ``SourceCollection`` to compute session extent
    (``union_range``), find which trial contains a given time, etc.
    Concrete adapters live in ``io/time_sources.py``.

    Not to be confused with ``PlotSource`` (``gui/plot_sources``),
    which is the rendering-layer protocol that plot widgets use for
    actual data loading and viewport caching.
    """

    @property
    def name(self) -> str: ...

    @property
    def time_range(self) -> TimeRange: ...

    @property
    def sampling_rate(self) -> float | None: ...

    def get_data(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(timestamps, values)`` for the requested window.

        ``timestamps`` is a 1-D float64 array.
        ``values`` is ``(T,)`` or ``(T, D)`` depending on the source.
        """
        ...


# ---------------------------------------------------------------------------
# SourceCollection — Neurosift-inspired union/intersection of sources
# ---------------------------------------------------------------------------


class SourceCollection:
    """Registry of loaded :class:`TimeSource` objects with range queries.

    Holds data sources and trial bookmarks.  Provides union/intersection
    of time ranges for Neurosift-style navigation where the user can view
    any time period, not just individual trials.
    """

    def __init__(self) -> None:
        self._sources: dict[str, TimeSource] = {}
        self._trial_intervals: list[tuple[float, float]] = []
        self._trial_ids: list[int | str] = []

    # -- Source management --------------------------------------------------

    def add(self, source: TimeSource) -> None:
        self._sources[source.name] = source

    @property
    def sources(self) -> dict[str, TimeSource]:
        return dict(self._sources)

    # -- Trial bookmarks ----------------------------------------------------

    def set_trials(
        self,
        ids: list[int | str],
        starts: list[float],
        stops: list[float] | None = None,
    ) -> None:
        """Set trial intervals.

        Parameters
        ----------
        ids
            Trial identifiers.
        starts
            Session-absolute start time for each trial.
        stops
            Session-absolute stop time for each trial.  If ``None``,
            inferred as ``start[i+1]`` for all but the last trial;
            the last trial's stop is the max of all source time ranges.
        """
        if stops is None:
            stops = []
            for i in range(len(starts)):
                if i < len(starts) - 1:
                    stops.append(starts[i + 1])
                else:
                    ur = self.union_range
                    stops.append(ur.end_s if ur else starts[i])
        self._trial_ids = list(ids)
        self._trial_intervals = list(zip(starts, stops))

    @property
    def trial_ids(self) -> list[int | str]:
        return list(self._trial_ids)

    @property
    def n_trials(self) -> int:
        return len(self._trial_intervals)

    def trial_range(self, idx: int) -> TimeRange:
        start, end = self._trial_intervals[idx]
        return TimeRange(start, end)

    def trial_offset(self, idx: int) -> float:
        """Session-absolute start time of a trial."""
        return self._trial_intervals[idx][0] if self._trial_intervals else 0.0

    def find_trial(self, t: float) -> int | None:
        """Return index of the trial containing time *t*, or ``None``."""
        for i, (start, end) in enumerate(self._trial_intervals):
            if start <= t <= end:
                return i
        # Fallback: find closest trial
        if not self._trial_intervals:
            return None
        dists = [min(abs(s - t), abs(e - t)) for s, e in self._trial_intervals]
        return int(np.argmin(dists))

    # -- Range queries (Neurosift-inspired) ---------------------------------

    @property
    def union_range(self) -> TimeRange | None:
        """Full navigable extent — union of all source time ranges."""
        if not self._sources:
            return None
        ranges = [s.time_range for s in self._sources.values()]
        result = ranges[0]
        for r in ranges[1:]:
            result = result.union(r)
        return result

    @property
    def intersection_range(self) -> TimeRange | None:
        """Time range where all sources overlap, or ``None``."""
        if not self._sources:
            return None
        ranges = [s.time_range for s in self._sources.values()]
        result = ranges[0]
        for r in ranges[1:]:
            result = result.intersect(r)
            if result is None:
                return None
        return result

    @property
    def session_range(self) -> TimeRange | None:
        """Full session extent: min(trial starts) .. max(trial ends).

        Falls back to :attr:`union_range` if no trial intervals are set.
        """
        if self._trial_intervals:
            starts = [s for s, _ in self._trial_intervals]
            ends = [e for _, e in self._trial_intervals]
            return TimeRange(min(starts), max(ends))
        return self.union_range


# ---------------------------------------------------------------------------
# Restriction window builders (moved from io/restrict.py)
# ---------------------------------------------------------------------------


def build_trial_window(
    trial_alignment: TrialVideoBounds,
    trial_id: int | str,
    extra_t0: float = 0.0,
    extra_t1: float = 0.0,
) -> RestrictionWindow:
    """Build a restriction window covering an entire trial."""
    core = trial_alignment.trial_range
    time_range = TimeRange(
        core.start_s - extra_t0,
        core.end_s + extra_t1,
    )
    return RestrictionWindow(
        mode="trial",
        time_range=time_range,
        core_range=core,
        trial_id=trial_id,
    )


def build_label_window(
    labels_df,
    label_idx: int,
    trial_bounds: TimeRange,
    extra_t0: float = 0.0,
    extra_t1: float = 0.0,
) -> RestrictionWindow:
    """Build a restriction window around a single label instance."""
    row = labels_df.iloc[label_idx]
    onset = float(row["onset_s"])
    offset = float(row["offset_s"])
    core = TimeRange(onset, offset)
    time_range = TimeRange(
        max(trial_bounds.start_s, onset - extra_t0),
        min(trial_bounds.end_s, offset + extra_t1),
    )
    return RestrictionWindow(
        mode="label",
        time_range=time_range,
        core_range=core,
        trial_id=row.get("trial"),
        label_info={
            "label_id": int(row["labels"]),
            "individual": row.get("individual"),
            "onset_s": onset,
            "offset_s": offset,
            "row_idx": label_idx,
        },
    )


def build_sequence_window(
    match: dict,
    trial_bounds: TimeRange,
    extra_t0: float = 0.0,
    extra_t1: float = 0.0,
) -> RestrictionWindow:
    """Build a restriction window spanning a matched label sequence."""
    onset = float(match["onset_s"])
    offset = float(match["offset_s"])
    core = TimeRange(onset, offset)
    time_range = TimeRange(
        max(trial_bounds.start_s, onset - extra_t0),
        min(trial_bounds.end_s, offset + extra_t1),
    )
    return RestrictionWindow(
        mode="sequence",
        time_range=time_range,
        core_range=core,
        trial_id=match["trial"],
        sequence_info={
            "pattern": match.get("pattern"),
            "match_rows": match.get("match_rows"),
        },
    )


def infer_slider_range(
    nwb_alignment,
    trial_id,
    source_collection: SourceCollection | None = None,
) -> tuple[str, TimeRange | None]:
    """Infer slider scope and time range from the best available timing.

    Returns ``(scope, time_range)`` where *scope* is one of:
    - ``"trial"``       — trial has start + stop
    - ``"trial_start"`` — trial has start only, extends to next trial or session end
    - ``"session"``     — no trial timing, use full session extent
    """
    start = nwb_alignment.start_time(trial_id)
    stop = nwb_alignment.stop_time(trial_id)

    if stop is not None:
        return "trial", TimeRange(0.0, stop - start)

    # Start-only: extend to next trial's start
    tsr = trial_start_range(nwb_alignment, trial_id)
    if tsr is not None:
        return "trial_start", tsr

    # Last trial or no next: use session extent
    if source_collection is not None:
        sr = source_collection.session_range
        if sr is not None:
            return "session", TimeRange(0.0, sr.end_s - start)

    return "session", None


def trial_start_range(nwb_alignment, trial_id) -> TimeRange | None:
    """Trial-relative range from this trial's start to the next trial's start.

    Returns ``None`` when the trial has no start time or no later trial
    exists (last trial).
    """
    start = nwb_alignment.start_time(trial_id)
    if start is None:
        return None
    df = nwb_alignment.trials_df
    if df.empty or "start_time" not in df.columns:
        return None
    next_starts = [s for s in df["start_time"].dropna().values if s > start]
    if not next_starts:
        return None
    return TimeRange(0.0, float(min(next_starts)) - start)


def find_closest_trial(nwb_alignment, trials: list, global_time: float) -> tuple[int | str, float]:
    """Given a session-absolute time, find the enclosing trial.

    Returns ``(trial_id, trial_relative_time)``.
    """
    ep = nwb_alignment.trials_ep
    if ep is None:
        raise ValueError("No trial timing information available")

    starts = np.asarray(ep.start)
    ends = np.asarray(ep.end)
    mask = (starts <= global_time) & (global_time <= ends)

    if mask.any():
        idx = int(np.argmax(mask))
    else:
        dists = np.minimum(
            np.abs(starts - global_time),
            np.abs(ends - global_time),
        )
        idx = int(np.argmin(dists))

    trial_id = trials[idx]
    trial_start = float(starts[idx])
    return trial_id, global_time - trial_start


def restrict_xarray(
    ds: xr.Dataset,
    time_range: TimeRange,
    time_coord_name: str = "time",
) -> xr.Dataset:
    """Slice an xarray Dataset to a time range."""
    return ds.sel({time_coord_name: slice(time_range.start_s, time_range.end_s)})


def restrict_pynapple(obj, time_range: TimeRange):
    """Restrict a pynapple object to a time range."""
    import pynapple as nap

    ep = nap.IntervalSet(start=time_range.start_s, end=time_range.end_s)
    return obj.restrict(ep)
