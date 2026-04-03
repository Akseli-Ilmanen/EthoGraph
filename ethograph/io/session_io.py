"""SessionIO: NWB-backed session metadata access for TrialTree."""

from __future__ import annotations

import logging
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPOCH_GAP = 1e-4


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionIO(Protocol):
    """Read-only interface for trial-level session metadata."""

    @property
    def trials_df(self) -> pd.DataFrame: ...

    def get_media(self, trial, stream: str, device: str | None = None) -> str | None: ...

    def devices(self, stream: str) -> list[str]: ...

    @property
    def cameras(self) -> list[str]: ...

    @property
    def mics(self) -> list[str]: ...

    def start_time(self, trial) -> float: ...

    def stop_time(self, trial) -> float | None: ...

    def trial_duration(self, trial) -> float: ...

    def source_start_time(self, trial, stream: str, device: str | None = None) -> float: ...

    def source_start_time_trial_relative(self, trial, stream: str, device: str | None = None) -> float: ...

    def get_video_fps(self, camera: str | None = None) -> float | None: ...

    def set_video_fps(self, fps: float, camera: str | None = None) -> None: ...

    @property
    def trials_ep(self) -> Any: ...

    def trial_epoch(self, trial) -> Any: ...

    def restrict(self, obj: Any, trial) -> Any: ...

    def print_session(self) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Column parsing helpers
# ---------------------------------------------------------------------------

_STREAM_COL_RE = re.compile(r"^(video|pose|audio|ephys)_(.+)$")


def _parse_stream_columns(columns: list[str], stream: str) -> list[str]:
    """Extract device names from trial table columns for a given stream.

    Columns like ``video_cam_1``, ``pose_cam_2`` -> ``["cam_1", "cam_2"]``.
    Excludes ``_start`` suffix columns (those are timing, not media).
    """
    devices = []
    prefix = f"{stream}_"
    for col in columns:
        if col.startswith(prefix) and not col.endswith("_start"):
            dev = col[len(prefix):]
            if dev and dev not in devices:
                devices.append(dev)
    return devices


# ---------------------------------------------------------------------------
# NWBSessionIO
# ---------------------------------------------------------------------------


class NWBSessionIO:
    """Session metadata backed by an NWB file.

    Opens the NWB file read-only and caches the trials table.
    Supports both NWB timing schemes (timestamps array and rate+starting_time).
    """

    def __init__(self, nwb_path: str | Path) -> None:
        self._path = Path(nwb_path)
        self._io: Any = None
        self._nwb: Any = None
        self._trials_df_cache: pd.DataFrame | None = None
        self._fps_overlay: dict[str | None, float] = {}

    def _open(self) -> None:
        if self._nwb is not None:
            return
        from pynwb import NWBHDF5IO

        self._io = NWBHDF5IO(str(self._path), "r")
        self._nwb = self._io.read()

    @property
    def nwb(self):
        self._open()
        return self._nwb

    # ── Trials table ──

    @property
    def trials_df(self) -> pd.DataFrame:
        if self._trials_df_cache is not None:
            return self._trials_df_cache
        nwb = self.nwb
        if nwb.trials is None:
            self._trials_df_cache = pd.DataFrame()
            return self._trials_df_cache
        df = nwb.trials.to_dataframe()
        self._trials_df_cache = df
        return self._trials_df_cache

    def _trial_row(self, trial) -> pd.Series | None:
        df = self.trials_df
        if df.empty:
            return None
        if "trial" in df.columns:
            match = df[df["trial"] == trial]
            if match.empty:
                match = df[df["trial"] == str(trial)]
            if match.empty and isinstance(trial, str) and trial.isdigit():
                match = df[df["trial"] == int(trial)]
            if not match.empty:
                return match.iloc[0]
        return None

    def _trial_index(self, trial) -> int | None:
        df = self.trials_df
        if df.empty:
            return None
        if "trial" in df.columns:
            for i, val in enumerate(df["trial"]):
                if val == trial or str(val) == str(trial):
                    return i
        return None

    # ── Media access ──

    def get_media(self, trial, stream: str, device: str | None = None) -> str | None:
        row = self._trial_row(trial)
        if row is None:
            return None
        col = f"{stream}_{device}" if device else stream
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val):
                return str(val)
        return None

    def devices(self, stream: str) -> list[str]:
        df = self.trials_df
        if df.empty:
            return []
        return _parse_stream_columns(list(df.columns), stream)

    @property
    def cameras(self) -> list[str]:
        return self.devices("video")

    @property
    def mics(self) -> list[str]:
        return self.devices("audio")

    # ── Timing ──

    def start_time(self, trial) -> float:
        row = self._trial_row(trial)
        if row is not None and "start_time" in row.index:
            val = row["start_time"]
            if pd.notna(val):
                return float(val)
        return 0.0

    def stop_time(self, trial) -> float | None:
        row = self._trial_row(trial)
        if row is None:
            return None
        # Check placeholder flag
        if "stop_time_is_placeholder" in row.index and row["stop_time_is_placeholder"] == 1:
            return None
        if "stop_time" in row.index:
            val = row["stop_time"]
            if pd.notna(val):
                return float(val)
        return None

    def trial_duration(self, trial) -> float:
        stop = self.stop_time(trial)
        if stop is None:
            raise ValueError(f"Trial {trial} has no known stop time")
        return stop - self.start_time(trial)

    def source_start_time(self, trial, stream: str, device: str | None = None) -> float:
        """Session-absolute time of sample 0 for a stream's file.

        For per-trial media: returns the trial's start_time (files are aligned).
        For session-wide media: reads from NWB acquisition ImageSeries timing.
        """
        nwb = self.nwb

        # Check acquisition for this stream+device
        acq_name = f"{stream}_{device}" if device else stream
        acq = nwb.acquisition.get(acq_name) if nwb.acquisition else None
        if acq is not None:
            from ethograph.utils.nwb import resolve_timeseries_timing
            try:
                _rate, t0 = resolve_timeseries_timing(acq)
                # For per-trial: find the start of this trial's frames
                starting_frame = getattr(acq, "starting_frame", None)
                if starting_frame is not None and len(starting_frame) > 0:
                    trial_idx = self._trial_index(trial)
                    if trial_idx is not None:
                        ts = getattr(acq, "timestamps", None)
                        if ts is not None and len(ts) > 0:
                            frame_idx = int(starting_frame[trial_idx]) if trial_idx < len(starting_frame) else 0
                            if frame_idx < len(ts):
                                return float(ts[frame_idx])
                        # Fall back to rate-based calculation
                        frame_idx = int(starting_frame[trial_idx]) if trial_idx < len(starting_frame) else 0
                        return t0 + frame_idx / _rate
                return t0
            except (ValueError, TypeError):
                pass

        # Fall back to trial start_time
        return self.start_time(trial)

    def source_start_time_trial_relative(self, trial, stream: str, device: str | None = None) -> float:
        """Trial-relative time of sample 0 for a stream's file.

        For per-trial aligned media returns 0.0 (unless a ``_start``
        column provides an explicit offset).
        For session-wide media returns source_start_time - trial_start.
        """
        df = self.trials_df
        col = f"{stream}_{device}" if device else stream

        # Check for an explicit per-trial offset column (e.g. video_cam-1_start)
        start_col = f"{col}_start"
        if not df.empty and start_col in df.columns:
            row = self._trial_row(trial)
            if row is not None and start_col in row.index and pd.notna(row[start_col]):
                return float(row[start_col])

        if not df.empty and col in df.columns:
            # Per-trial media with no explicit offset: file is aligned to trial
            return 0.0

        abs_start = self.source_start_time(trial, stream, device)
        return abs_start - self.start_time(trial)

    # ── FPS ──

    def get_video_fps(self, camera: str | None = None) -> float | None:
        # Check in-memory overlay first
        if camera in self._fps_overlay:
            return self._fps_overlay[camera]
        if None in self._fps_overlay and camera is not None:
            return self._fps_overlay[None]

        # Read from acquisition ImageSeries
        nwb = self.nwb
        if nwb.acquisition:
            if camera:
                acq_name = f"video_{camera}"
                acq = nwb.acquisition.get(acq_name)
                if acq is not None:
                    from ethograph.utils.nwb import resolve_timeseries_timing
                    try:
                        rate, _ = resolve_timeseries_timing(acq)
                        return rate
                    except ValueError:
                        pass
            # Try any video acquisition
            for name, acq in nwb.acquisition.items():
                if name.startswith("video_"):
                    from ethograph.utils.nwb import resolve_timeseries_timing
                    try:
                        rate, _ = resolve_timeseries_timing(acq)
                        return rate
                    except ValueError:
                        continue
        return None

    def set_video_fps(self, fps: float, camera: str | None = None) -> None:
        self._fps_overlay[camera] = fps

    # ── Epochs (pynapple IntervalSet) ──

    @cached_property
    def trials_ep(self):
        import pynapple as nap

        df = self.trials_df
        if df.empty or "start_time" not in df.columns:
            return None

        starts = df["start_time"].values.astype(np.float64)
        n = len(starts)
        if n == 0:
            return None

        ends = np.full(n, np.nan, dtype=np.float64)
        is_placeholder = "stop_time_is_placeholder" in df.columns
        if "stop_time" in df.columns and not is_placeholder:
            ends = df["stop_time"].values.astype(np.float64)

        has_monotonic = n == 1 or bool(np.all(np.diff(starts) > 0))
        if not has_monotonic:
            return None

        has_stop = ~np.isnan(ends)
        safe_ends = ends.copy()
        for i in range(n - 1):
            if np.isnan(safe_ends[i]):
                safe_ends[i] = starts[i + 1] - _EPOCH_GAP
        if np.isnan(safe_ends[-1]):
            safe_ends[-1] = starts[-1] + 1.0

        trial_ids = df["trial"].values if "trial" in df.columns else np.arange(1, n + 1)

        return nap.IntervalSet(
            start=starts,
            end=safe_ends,
            metadata={
                "trial": np.array(trial_ids),
                "has_stop": has_stop.astype(float),
            },
        )

    def _trial_ep_idx(self, trial) -> int:
        ep = self.trials_ep
        if ep is None:
            raise KeyError(f"Trial {trial} not found in timing table")
        for i, t in enumerate(ep.metadata["trial"]):
            if t == trial or str(t) == str(trial):
                return i
        raise KeyError(f"Trial {trial} not found in timing table")

    def trial_epoch(self, trial):
        import pynapple as nap

        ep = self.trials_ep
        if ep is None:
            raise ValueError("No timing information available")
        idx = self._trial_ep_idx(trial)
        return nap.IntervalSet(start=ep.start[idx], end=ep.end[idx])

    def restrict(self, obj, trial):
        return obj.restrict(self.trial_epoch(trial))

    # ── Display ──

    def print_session(self) -> None:
        df = self.trials_df
        if df.empty:
            print("No session table.")
            return
        print(f"\n{'=' * 60}")
        print(f"  NWB Session: {self._path.name}")
        print(f"  {len(df)} trials, columns: {list(df.columns)}")
        print(f"{'=' * 60}")
        print(df.to_string(max_rows=20))

        # Show acquisition items
        nwb = self.nwb
        if nwb.acquisition:
            print(f"\n  Acquisition items:")
            for name, acq in nwb.acquisition.items():
                info = f"    {name}"
                if hasattr(acq, "rate") and acq.rate:
                    info += f" (rate={acq.rate} Hz)"
                if hasattr(acq, "external_file") and acq.external_file:
                    info += f" ({len(acq.external_file)} files)"
                if hasattr(acq, "timestamps") and acq.timestamps is not None:
                    info += f" ({len(acq.timestamps)} timestamps)"
                print(info)

    # ── Cleanup ──

    def close(self) -> None:
        if self._io is not None:
            try:
                self._io.close()
            except Exception:
                pass
            self._io = None
            self._nwb = None


# ---------------------------------------------------------------------------
# EmptySessionIO
# ---------------------------------------------------------------------------


class EmptySessionIO:
    """Null-object session: no metadata available."""

    @property
    def trials_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_media(self, trial, stream: str, device: str | None = None) -> str | None:
        return None

    def devices(self, stream: str) -> list[str]:
        return []

    @property
    def cameras(self) -> list[str]:
        return []

    @property
    def mics(self) -> list[str]:
        return []

    def start_time(self, trial) -> float:
        return 0.0

    def stop_time(self, trial) -> float | None:
        return None

    def trial_duration(self, trial) -> float:
        raise ValueError(f"Trial {trial} has no known stop time")

    def source_start_time(self, trial, stream: str, device: str | None = None) -> float:
        return 0.0

    def source_start_time_trial_relative(self, trial, stream: str, device: str | None = None) -> float:
        return 0.0

    def get_video_fps(self, camera: str | None = None) -> float | None:
        return None

    def set_video_fps(self, fps: float, camera: str | None = None) -> None:
        pass

    @property
    def trials_ep(self):
        return None

    def trial_epoch(self, trial):
        raise ValueError("No timing information available")

    def restrict(self, obj, trial):
        raise ValueError("No timing information available")

    def print_session(self) -> None:
        print("No session table.")

    def close(self) -> None:
        pass
