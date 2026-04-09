""" Alignment of media streams, feature data and trial timing from alignment.nwb."""

from __future__ import annotations

import logging
import os
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPOCH_GAP = 1e-4
_KNOWN_STREAMS = ("video", "pose", "audio", "ephys")
_NWB_FILENAME = "alignment.nwb"
_SETTINGS_DIR = ".ethograph"
_SENTINEL = object()


def discover_nwb(nc_path: str | Path) -> Path | None:
    """Find an NWB session file near a data file.

    Search order:
    1. ``<dir>/.ethograph/alignment.nwb``
    2. Any ``.nwb`` file in ``<dir>/.ethograph/``
    """
    d = Path(nc_path).resolve().parent
    ethograph_dir = d / _SETTINGS_DIR
    if ethograph_dir.is_dir():
        candidate = ethograph_dir / _NWB_FILENAME
        if candidate.exists():
            return candidate
        nwb_files = list(ethograph_dir.glob("*.nwb"))
        if nwb_files:
            return nwb_files[0]
    return None


def make_nwb_alignment(nwb_path: str | Path | None = None):
    """Create a EmpytAlignment from an NWB path, falling back to base EmpytAlignment."""
    if nwb_path and Path(nwb_path).exists():
        return NWBAlignment(nwb_path)
    return EmpytAlignment()


# ---------------------------------------------------------------------------
# Base class (also serves as null-object when no NWB is loaded)
# ---------------------------------------------------------------------------


class EmpytAlignment:
    """Base session metadata interface with null-object defaults.

    ``NWBAlignment`` overrides these with real NWB-backed implementations.
    When no NWB file is available, the base class is used directly.
    """

    @property
    def trials_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_media(self, trial, stream: str, device: str | None = None) -> str | None:
        return None

    def devices(self, stream: str) -> list[str]:
        return []

    @property
    def cameras(self) -> list[str]:
        return self.devices("video")

    @property
    def mics(self) -> list[str]:
        return self.devices("audio")

    def start_time(self, trial) -> float:
        return 0.0

    def stop_time(self, trial) -> float | None:
        return None

    def stream_offset_for_trial(self, trial, stream: str, device: str | None = None) -> float:
        return 0.0

    def get_stream_rate(self, stream: str, device: str | None = None) -> float | None:
        return None

    def set_stream_rate(self, rate: float, stream: str, device: str | None = None) -> None:
        pass

    def resolve_media_path(
        self, trial, stream: str, device: str | None = None,
        fallback_folder: str | None = None,
    ) -> str | None:
        return None
    
    def electrical_series(self) -> list[dict]:
        """Discover ElectricalSeries in acquisition. Returns list of {name, path, n_channels, rate}."""
        return []

    @property
    def trials_ep(self) -> Any:
        """Pynapple IntervalSet built from trials_df start/stop times."""
        cached = getattr(self, "_trials_ep_cache", _SENTINEL)
        if cached is not _SENTINEL:
            return cached
        ep = _build_trials_ep(self.trials_df)
        self._trials_ep_cache = ep
        return ep

    def trial_epoch(self, trial) -> Any:
        raise ValueError("No timing information available")

    def restrict(self, obj: Any, trial) -> Any:
        raise ValueError("No timing information available")

    def print_session(self) -> None:
        print("No session table.")

    def close(self) -> None:
        pass


class TableAlignment(EmpytAlignment):
    """Alignment backed by a tabular dataframe with trial timing columns.

    Expected columns are ``trial``, ``start_time``, and ``stop_time``.
    This is used as a fallback when no suitable alignment NWB is available.
    """

    def __init__(self, trials_df: pd.DataFrame) -> None:
        self._trials_df = trials_df.copy()

    @property
    def trials_df(self) -> pd.DataFrame:
        return self._trials_df

    def _trial_row(self, trial) -> pd.Series | None:
        df = self._trials_df
        if df.empty or "trial" not in df.columns:
            return None
        match = df[df["trial"] == trial]
        if match.empty:
            match = df[df["trial"] == str(trial)]
        if match.empty and isinstance(trial, str) and trial.isdigit():
            match = df[df["trial"] == int(trial)]
        if match.empty:
            return None
        return match.iloc[0]

    def start_time(self, trial) -> float:
        row = self._trial_row(trial)
        if row is None or "start_time" not in row.index or pd.isna(row["start_time"]):
            return 0.0
        return float(row["start_time"])

    def stop_time(self, trial) -> float | None:
        row = self._trial_row(trial)
        if row is None or "stop_time" not in row.index or pd.isna(row["stop_time"]):
            return None
        return float(row["stop_time"])


# ---------------------------------------------------------------------------
# Column parsing helpers
# ---------------------------------------------------------------------------



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
# NWBAlignment
# ---------------------------------------------------------------------------


class NWBAlignment:
    """Session metadata backed by an NWB file.

    All external media (video, audio, pose) are stored as ImageSeries
    in acquisition with ``{stream}_{device}`` naming.  Timing comes from
    ``rate`` or ``timestamps`` on the ImageSeries.
    """

    def __init__(self, nwb_path: str | Path) -> None:
        self._path = Path(nwb_path)
        self._io: Any = None
        self._nwb: Any = None
        self._trials_df_cache: pd.DataFrame | None = None
        self._rate_dict: dict[tuple[str, str | None], float] = {}

    @classmethod
    def from_nwb_object(cls, nwb_obj) -> "NWBAlignment":
        """Create alignment from an already-opened pynwb NWBFile object."""
        instance = cls.__new__(cls)
        instance._path = Path(".")
        instance._io = None
        instance._nwb = nwb_obj
        instance._trials_df_cache = None
        instance._rate_dict = {}
        return instance

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
        # No trial column: use 1-based integer lookup into row index
        idx = self._trial_to_iloc(trial)
        if idx is not None and 0 <= idx < len(df):
            return df.iloc[idx]
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
        return self._trial_to_iloc(trial)

    @staticmethod
    def _trial_to_iloc(trial) -> int | None:
        """Convert a trial ID to a 0-based row index (assumes 1-based trial IDs)."""
        try:
            idx = int(trial) - 1
            return idx if idx >= 0 else None
        except (ValueError, TypeError):
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
        """Discover devices from trials table columns AND acquisition items."""
        devs: list[str] = []

        # From trials table columns
        df = self.trials_df
        if not df.empty:
            devs = _parse_stream_columns(list(df.columns), stream)

        # From acquisition ImageSeries names
        nwb = self.nwb
        prefix = f"{stream}_"
        if nwb.acquisition:
            for name in nwb.acquisition:
                if name.startswith(prefix):
                    dev = name[len(prefix):]
                    if dev and dev not in devs:
                        devs.append(dev)

        return devs

    @property
    def cameras(self) -> list[str]:
        return self.devices("video")

    @property
    def mics(self) -> list[str]:
        return self.devices("audio")

    def electrical_series(self) -> list[dict]:
        """Discover ElectricalSeries in acquisition. Returns list of {name, path, n_channels, rate}."""
        import pynwb.ecephys

        nwb = self.nwb
        if not nwb.acquisition:
            return []
        results = []
        for name, obj in nwb.acquisition.items():
            if not isinstance(obj, pynwb.ecephys.ElectricalSeries):
                continue
            n_ch = obj.data.shape[1] if hasattr(obj.data, "shape") and obj.data.ndim > 1 else 1
            rate = float(obj.rate) if obj.rate else None
            results.append({
                "name": name,
                "path": str(self._path),
                "n_channels": n_ch,
                "rate": rate,
            })
        return results

    # ── Timing ──

    @cached_property
    def has_real_timing(self) -> bool:
        """Whether the trials table has meaningful start/stop times.

        Returns False when all trials have identical placeholder timing
        (e.g. start=0.0, stop=1.0 for every row), which indicates the
        NWB was generated without real session timing.
        """
        df = self.trials_df
        if df.empty or "start_time" not in df.columns:
            return False
        starts = df["start_time"].dropna()
        if len(starts) == 0:
            return False
        if len(starts) > 1 and starts.nunique() > 1:
            return True
        # Single unique start: check if stop_time varies or is > start + 1.001
        if "stop_time" in df.columns:
            stops = df["stop_time"].dropna()
            durations = stops - starts.iloc[0]
            if durations.nunique() > 1:
                return True
        return False

    @cached_property
    def provenance(self) -> dict | None:
        """Provenance metadata from ``provenance.yaml`` next to alignment.nwb."""
        from ethograph.utils.nwb import read_provenance
        return read_provenance(self._path.parent)

    @property
    def pose_keys(self) -> list[str]:
        """Ordered list of pose estimation container names from provenance."""
        prov = self.provenance
        
        # Defined by provenance field (e.g. from GUI video-pose matcher).
        if prov and "nwb_pose_keys" in prov:
            return list(prov["nwb_pose_keys"])
        
        # Defined by acquisition ImageSeries names
        return self.devices("pose")

    def update_provenance(self, updates: dict) -> None:
        """Merge *updates* into ``provenance.yaml`` next to alignment.nwb."""
        from ethograph.utils.nwb import read_provenance, write_provenance

        current = dict(read_provenance(self._path.parent) or {})
        current.update(updates)
        write_provenance(self._path.parent, current)

        if "provenance" in self.__dict__:
            del self.__dict__["provenance"]

    def start_time(self, trial) -> float:
        if not self.has_real_timing:
            return 0.0
        row = self._trial_row(trial)
        if row is not None and "start_time" in row.index:
            val = row["start_time"]
            if pd.notna(val):
                return float(val)
        return 0.0

    def stop_time(self, trial) -> float | None:
        if not self.has_real_timing:
            return None
        row = self._trial_row(trial)
        if row is None:
            return None
        if "stop_time" in row.index:
            val = row["stop_time"]
            if pd.notna(val):
                return float(val)
        return None

    def stream_offset_for_trial(
        self, trial, stream: str, device: str | None = None,
    ) -> float:
        """Trial-relative time of sample 0 for a stream's file.

        For per-trial aligned media returns 0.0.
        For session-wide media returns the file's start relative to the trial.
        Reads timing from the acquisition ImageSeries.
        """
        trial_start = self.start_time(trial)
        trial_idx = self._trial_index(trial)

        nwb = self.nwb
        acq_name = f"{stream}_{device}" if device else stream
        acq = nwb.acquisition.get(acq_name) if nwb.acquisition else None
        if acq is None:
            return 0.0

        starting_frame = getattr(acq, "starting_frame", None)
        timestamps = getattr(acq, "timestamps", None)
        rate = getattr(acq, "rate", None)

        if starting_frame is not None and trial_idx is not None and trial_idx < len(starting_frame):
            frame_idx = int(starting_frame[trial_idx])

            if timestamps is not None and frame_idx < len(timestamps):
                # Timestamps mode: read directly
                file_start_time = float(timestamps[frame_idx])
            elif rate and rate > 0:
                # Rate mode: compute from starting_time + frame/rate
                t0 = float(acq.starting_time) if acq.starting_time is not None else 0.0
                file_start_time = t0 + frame_idx / rate
            else:
                return 0.0

            return file_start_time - trial_start

        # No starting_frame or trial not found — use first timestamp
        if timestamps is not None and len(timestamps) > 0:
            return float(timestamps[0]) - trial_start
        if rate and rate > 0:
            t0 = float(acq.starting_time) if acq.starting_time is not None else 0.0
            return t0 - trial_start

        return 0.0

    # ── Stream rate ──

    def get_stream_rate(self, stream: str, device: str | None = None) -> float | None:
        """Read the sampling rate for a stream from its acquisition ImageSeries."""
        key = (stream, device)
        if key in self._rate_dict:
            return self._rate_dict[key]

        nwb = self.nwb
        if not nwb.acquisition:
            return None

        from ethograph.utils.nwb import resolve_timeseries_timing

        if device:
            acq = nwb.acquisition.get(f"{stream}_{device}")
            if acq is not None:
                rate, _ = resolve_timeseries_timing(acq)
                return rate

        for name, acq in nwb.acquisition.items():
            if name.startswith(f"{stream}_"):
                rate, _ = resolve_timeseries_timing(acq)
                return rate

        return None

    def set_stream_rate(self, rate: float, stream: str, device: str | None = None) -> None:
        self._rate_dict[(stream, device)] = rate

    def stream_rates(self) -> dict[str, tuple[float, float | None]]:
        """Return ``{acq_name: (starting_time, rate)}`` for all acquisition streams.

        Only returns rate and starting_time from NWB ImageSeries metadata.
        Does not compute duration — external files must be probed for that.
        """
        from ethograph.utils.nwb import resolve_timeseries_timing

        nwb = self.nwb
        if not nwb.acquisition:
            return {}
        result: dict[str, tuple[float, float | None]] = {}
        for name, acq in nwb.acquisition.items():
            rate, _ = resolve_timeseries_timing(acq)
            start = float(acq.starting_time) if acq.starting_time is not None else 0.0
            result[name] = (start, rate)
        return result

    # ── Media path resolution ──

    def resolve_media_path(
        self,
        trial,
        stream: str,
        device: str | None = None,
        fallback_folder: str | None = None,
    ) -> str | None:
        """Resolve the full path for a media file.

        1. Try the ImageSeries ``external_file`` path for this trial (if on disk).
        2. Fallback: trial table filename + ``fallback_folder``.
        3. Returns ``None`` if unresolvable.
        """
        nwb = self.nwb
        acq_name = f"{stream}_{device}" if device else stream
        acq = nwb.acquisition.get(acq_name) if nwb.acquisition else None

        trial_idx = self._trial_index(trial)
        nwb_base_dir = self._path.parent

        if acq is not None and hasattr(acq, "external_file") and acq.external_file:
            starting_frame = getattr(acq, "starting_frame", None)
            files = list(acq.external_file)

            if trial_idx is not None and starting_frame is not None and trial_idx < len(starting_frame):
                file_idx = _file_index_for_trial(starting_frame, trial_idx, len(files))
            elif trial_idx is not None and trial_idx < len(files):
                file_idx = trial_idx
            else:
                file_idx = 0

            if file_idx < len(files):
                raw_path = files[file_idx]
                # URLs returned directly
                if _is_url(raw_path):
                    # If fallback_folder has a local copy, prefer that
                    if fallback_folder:
                        filename = _filename_from_url_or_path(raw_path)
                        candidate = os.path.normpath(os.path.join(fallback_folder, filename))
                        if os.path.isfile(candidate):
                            return candidate
                    return raw_path
                # Try the stored path directly
                if os.path.isfile(raw_path):
                    return raw_path
                # Try relative to NWB file location
                rel = nwb_base_dir / raw_path
                if rel.is_file():
                    return str(rel)
                # Fallback: filename + folder
                filename = _filename_from_url_or_path(raw_path)
                if fallback_folder:
                    candidate = os.path.normpath(os.path.join(fallback_folder, filename))
                    if os.path.isfile(candidate):
                        return candidate

        # Last resort: trial table filename + fallback_folder
        media_file = self.get_media(trial, stream, device)
        if media_file:
            if _is_url(media_file):
                return media_file
            if os.path.isfile(media_file):
                return media_file
            if fallback_folder:
                filename = _filename_from_url_or_path(media_file)
                candidate = os.path.normpath(os.path.join(fallback_folder, filename))
                if os.path.isfile(candidate):
                    return candidate

        return None

    # ── Epochs (pynapple IntervalSet) ──

    @cached_property
    def trials_ep(self):
        """Pynapple IntervalSet built from the NWB trials table."""
        return _build_trials_ep(self.trials_df)

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
# Helpers
# ---------------------------------------------------------------------------


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def _filename_from_url_or_path(path: str) -> str:
    """Extract filename from a URL or filesystem path (Windows-safe)."""
    if _is_url(path):
        from urllib.parse import urlparse
        return Path(urlparse(path).path).name
    return Path(path).name


def _build_trials_ep(df: pd.DataFrame, session_end: float | None = None):
    """Build a pynapple IntervalSet from a trials DataFrame with start/stop times.

    Parameters
    ----------
    df
        Trials DataFrame with ``start_time`` and optionally ``stop_time``.
    session_end
        Session end time (seconds). Used as fallback stop for the last trial
        when its stop time is unknown. Pass ``source_collection.union_range.end_s``
        after data loading.

    Returns None when timing data is missing or non-monotonic.
    """
    if df.empty or "start_time" not in df.columns:
        return None

    import pynapple as nap

    starts = df["start_time"].values.astype(np.float64)
    n = len(starts)
    if n == 0:
        return None

    ends = np.full(n, np.nan, dtype=np.float64)
    if "stop_time" in df.columns:
        ends = df["stop_time"].values.astype(np.float64)

    has_monotonic = n == 1 or bool(np.all(np.diff(starts) > 0))
    if not has_monotonic:
        return None

    has_stop = ~np.isnan(ends)
    safe_ends = ends.copy()
    for i in range(n - 1):
        if np.isnan(safe_ends[i]):
            safe_ends[i] = starts[i + 1] - _EPOCH_GAP

    trial_ids = df["trial"].values if "trial" in df.columns else np.arange(1, n + 1)

    # Last trial with unknown stop: use session end if available, else drop
    if np.isnan(safe_ends[-1]):
        if session_end is not None and session_end > starts[-1]:
            safe_ends[-1] = session_end
        else:
            mask = np.ones(n, dtype=bool)
            mask[-1] = False
            starts = starts[mask]
            safe_ends = safe_ends[mask]
            has_stop = has_stop[mask]
            trial_ids = trial_ids[mask]
            if len(starts) == 0:
                return None

    return nap.IntervalSet(
        start=starts,
        end=safe_ends,
        metadata={
            "trial": np.array(trial_ids),
            "has_stop": has_stop.astype(float),
        },
    )


def _file_index_for_trial(
    starting_frames: list | np.ndarray,
    trial_idx: int,
    n_files: int,
) -> int:
    """Map a trial index to the corresponding file index via starting_frame."""
    sf = [int(f) for f in starting_frames]
    if trial_idx >= len(sf):
        return min(trial_idx, n_files - 1)
    target_frame = sf[trial_idx]
    for i in range(n_files - 1, -1, -1):
        if i < len(sf) and sf[i] <= target_frame:
            return i
    return 0
