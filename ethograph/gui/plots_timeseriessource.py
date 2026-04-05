"""Trial alignment and restriction window abstractions.

Key types:
    TimeRange               -- Immutable time interval with set operations
    RestrictionWindow       -- Display window for trial/label/sequence navigation
    TrialAlignment          -- Trial time range + video offset
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from ethograph.io.trialtree import TrialTree


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


@dataclass(frozen=True)
class RestrictionWindow:
    """Describes the currently active display window.

    In "trial" mode this wraps a full trial range.  In "label" or
    "sequence" mode it wraps a sub-interval (with optional extra
    context padding).
    """

    mode: str  # "trial" | "label" | "sequence"
    time_range: TimeRange  # effective display window (including extra context)
    core_range: TimeRange  # the actual interval (without extra context)
    trial_id: int | str | None = None
    label_info: dict | None = None
    sequence_info: dict | None = None


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
    source_ranges
        Per-source time ranges discovered during alignment, keyed by
        source name (e.g. ``"session_table"``, ``"xarray"``, ``"video"``,
        ``"audio"``).
    """

    trial_id: str
    trial_range: TimeRange | None = None
    video_offset: float = 0.0
    ephys_offset: float = 0.0
    source_ranges: dict[str, TimeRange] | None = None
    _cached_video_reader: object = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [f"  TrialAlignment(trial={self.trial_id!r})"]
        if self.trial_range:
            lines.append(f"    effective range: {self.trial_range}")
        if self.video_offset:
            lines.append(f"    video_offset:   {self.video_offset:.3f}s")
        if self.ephys_offset:
            lines.append(f"    ephys_offset:   {self.ephys_offset:.3f}s")
        if self.source_ranges:
            lines.append("    sources:")
            for name, tr in self.source_ranges.items():
                lines.append(f"      {name:20s} {tr}")
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

    video_path = dt.resolve_media_path(
        trial_id, "video", device=cameras_sel, fallback_folder=video_folder,
    )
    video_offset = dt.stream_offset_for_trial(trial_id, "video", cameras_sel) if video_path else 0.0

    audio_devices = dt.devices("audio")
    audio_device = audio_devices[0] if audio_devices else None
    audio_path = dt.resolve_media_path(
        trial_id, "audio", device=audio_device, fallback_folder=audio_folder,
    ) if audio_device else None

    ephys_offset = 0.0
    try:
        ephys_offset = dt.start_time(str(trial_id))
    except (KeyError, AttributeError):
        pass

    trial_end, source_ranges, video_reader = _compute_trial_end(
        dt, trial_id, ds, video_path, video_offset, audio_path, audio_device
    )
    trial_range = TimeRange(0.0, trial_end) if trial_end and trial_end > 0 else None
    return TrialAlignment(
        trial_id=str(trial_id),
        trial_range=trial_range,
        video_offset=video_offset,
        ephys_offset=ephys_offset,
        source_ranges=source_ranges or None,
        _cached_video_reader=video_reader,
    )


def _compute_trial_end(
    dt: TrialTree,
    trial_id,
    ds: xr.Dataset,
    video_path: str | None,
    video_offset: float,
    audio_path: str | None,
    audio_device: str | None = None,
) -> tuple[float | None, dict[str, TimeRange], object]:
    """Return (trial_duration, per_source_ranges, video_reader).

    The first non-None duration wins as the effective trial end.
    All discovered source ranges are always returned.
    ``video_reader`` is the :class:`FastVideoReader` opened to probe
    the video length so the caller can reuse it instead of re-opening.
    """
    from ethograph.utils.xr_utils import get_time_coord

    best_end: float | None = None
    source_ranges: dict[str, TimeRange] = {}
    video_reader = None

    # 1. Session stop_time
    try:
        stop = dt.stop_time(trial_id)
        if stop is not None:
            duration = stop - dt.start_time(trial_id)
            source_ranges["session_table"] = TimeRange(0.0, duration)
            if best_end is None:
                best_end = duration
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
                    t_start = float(vals[0])
                    t_end = float(vals[-1])
                    if t_end > 0:
                        source_ranges[f"xarray:{var_name}"] = TimeRange(t_start, t_end)
                        if best_end is None:
                            best_end = t_end

    # 3. Video
    if video_path:
        try:
            from napari_pyav._reader import FastVideoReader
            video_reader = FastVideoReader(video_path, read_format="rgb24")
            n_frames = video_reader.shape[0]
            fps = float(video_reader.stream.guessed_rate)
            if n_frames > 0 and fps > 0:
                vid_end = video_offset + n_frames / fps
                source_ranges["video"] = TimeRange(video_offset, vid_end)
                if best_end is None:
                    best_end = vid_end
        except Exception:
            video_reader = None

    # 4. Audio (per-trial only — session-wide sources have large negative start)
    if audio_path:
        try:
            from ethograph.gui.plots_spectrogram import SharedAudioCache
            loader = SharedAudioCache.get_loader(audio_path)
            audio_start = dt.stream_offset_for_trial(trial_id, "audio", audio_device)
            if loader is not None and len(loader) > 0:
                audio_end = len(loader) / loader.rate
                source_ranges["audio"] = TimeRange(audio_start, audio_start + audio_end)
                if best_end is None and audio_start >= -0.5:
                    best_end = audio_end
        except Exception:
            pass

    return best_end, source_ranges, video_reader
