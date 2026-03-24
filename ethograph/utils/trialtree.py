from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List
import json

import numpy as np
import pandas as pd
import pynapple as nap
import xarray as xr

from ethograph.utils.label_intervals import (
    empty_intervals,
    intervals_to_xr,
)
from ethograph.utils.validation import validate_datatree


# ---------------------------------------------------------------------------
# SessionIO — pynapple-backed timing + xarray media manifest
# ---------------------------------------------------------------------------

_SENTINEL_END = 1e9  # placeholder end for trials without a known stop time


def _ep_to_dataset(ep: nap.IntervalSet) -> xr.Dataset:
    data_vars: dict[str, xr.DataArray] = {
        "start": xr.DataArray(ep.start, dims=["row"]),
        "end": xr.DataArray(ep.end, dims=["row"]),
    }
    for col in ep.metadata.columns:
        vals = ep.metadata[col].values
        if vals.dtype == object:
            vals = vals.astype(str)
        data_vars[col] = xr.DataArray(vals, dims=["row"])
    return xr.Dataset(data_vars)


def _dataset_to_ep(ds: xr.Dataset) -> nap.IntervalSet:
    meta = {k: ds[k].values for k in ds.data_vars if k not in {"start", "end"}}
    return nap.IntervalSet(
        start=ds["start"].values,
        end=ds["end"].values,
        metadata=meta or None,
    )


@dataclass
class SessionIO:
    """Timing (pynapple IntervalSet) + file manifest (xarray Dataset).

    Parameters
    ----------
    trials_ep : nap.IntervalSet
        One interval per trial with ``trial`` metadata column.
        End times >= ``_SENTINEL_END`` indicate unknown stop time.
    media : xr.Dataset
        File paths as DataArrays.  Per-trial streams have a ``trial`` dim;
        session-wide streams (e.g. a single ephys file) do not.
        Camera/mic device labels are encoded as a second coord dimension.
    stream_epochs : dict[str, nap.IntervalSet]
        Session-absolute time coverage of each stream's file(s).

        * **Per-trial** (one file per trial): same intervals as ``trials_ep``
          — ``start[i]`` equals ``trials_ep.start[i]``, so
          ``source_start_time`` returns 0 for every trial.
        * **Session-wide** (one file for the whole session): single interval,
          e.g. ``IntervalSet(start=[0.0], end=[session_end])``.
          An offset (e.g. audio started 230 ms after t=0) is encoded directly
          as the interval start: ``IntervalSet(start=[0.23], end=[...])``.

        All cameras of the same stream share one ``stream_epochs`` entry —
        timing is per-stream-type, not per-device.  Device selection is
        handled by ``media``.
    attrs : dict
        Arbitrary session-level string attributes (e.g. ``ephys_stream_id``).
    """

    trials_ep: nap.IntervalSet
    media: xr.Dataset
    stream_epochs: dict[str, nap.IntervalSet] = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Timing API
    # ------------------------------------------------------------------

    def start_time(self, trial) -> float:
        if not self._has_timing():
            return 0.0
        return float(self.trials_ep.start[self._trial_idx(trial)])

    def stop_time(self, trial) -> float | None:
        """Session-absolute stop time, or ``None`` if not explicitly known."""
        if not self._has_timing():
            return None
        idx = self._trial_idx(trial)
        meta = self.trials_ep.metadata
        if "has_stop" in meta.columns and not bool(meta["has_stop"].iloc[idx]):
            return None
        end = float(self.trials_ep.end[idx])
        return None if end >= _SENTINEL_END else end

    def trial_duration(self, trial) -> float:
        stop = self.stop_time(trial)
        if stop is None:
            raise ValueError(f"Trial {trial} has no known stop time")
        return stop - self.start_time(trial)

    def trial_epoch(self, trial) -> nap.IntervalSet:
        idx = self._trial_idx(trial)
        return nap.IntervalSet(
            start=self.trials_ep.start[idx],
            end=self.trials_ep.end[idx],
        )

    def restrict(self, tsgroup_or_tsd, trial):
        """Restrict any pynapple object to a single trial window."""
        return tsgroup_or_tsd.restrict(self.trial_epoch(trial))

    # ------------------------------------------------------------------
    # Media / stream API
    # ------------------------------------------------------------------

    def source_start_time(self, trial, stream: str) -> float:
        """Trial-relative time of sample 0 of this stream's file.

        Passed as ``start_time`` to :class:`RegularTimeseriesSource`.
        Works identically for per-trial and session-wide layouts::

            result = stream_file_start_session_abs - trial_start_session_abs

        * Per-trial  → stream start == trial start → returns 0.
        * Session-wide starting at t=0, trial at t=30 → returns -30.
        * Session-wide with 230 ms offset, trial at t=30 → returns -29.77.
        """
        ep = self.stream_epochs.get(stream)
        if ep is None or len(ep) == 0:
            return 0.0
        if not self._has_timing():
            # No session-absolute timing — per-trial files always start at 0.
            return 0.0
        trial_start = self.trials_ep.start[self._trial_idx(trial)]
        file_idx = max(0, int(np.searchsorted(ep.start, trial_start, side="right")) - 1)
        return float(ep.start[file_idx]) - float(trial_start)

    def get_media(self, trial, stream: str, device: str | None = None) -> str | None:
        if stream not in self.media:
            return None
        da = self.media[stream]
        sel: dict = {}
        if "trial" in da.dims:
            sel["trial"] = trial
        device_dims = [d for d in da.dims if d != "trial"]
        if device is not None and device_dims:
            sel[device_dims[0]] = device
        val = da.sel(**sel) if sel else da
        result = str(val.values.flat[0])
        return result or None

    def devices(self, stream: str) -> list[str]:
        if stream not in self.media:
            return []
        da = self.media[stream]
        device_dims = [d for d in da.dims if d != "trial"]
        if not device_dims:
            return []
        return [str(v) for v in da.coords[device_dims[0]].values]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        tree = xr.DataTree()
        tree["media"] = xr.DataTree(self.media)
        tree["trials_ep"] = xr.DataTree(_ep_to_dataset(self.trials_ep))
        for stream, ep in self.stream_epochs.items():
            tree[f"epoch_{stream}"] = xr.DataTree(_ep_to_dataset(ep))
        tree.attrs["session_attrs"] = json.dumps(self.attrs)
        tree.to_netcdf(path)

    @classmethod
    def load(cls, path: str) -> "SessionIO":
        tree = xr.open_datatree(path, engine="netcdf4")
        trials_ep = _dataset_to_ep(tree["trials_ep"].ds)
        media = tree["media"].ds
        stream_epochs = {
            name[len("epoch_"):]: _dataset_to_ep(node.ds)
            for name, node in tree.children.items()
            if name.startswith("epoch_")
        }
        attrs = json.loads(tree.attrs.get("session_attrs", "{}"))
        return cls(trials_ep=trials_ep, media=media, stream_epochs=stream_epochs, attrs=attrs)

    @classmethod
    def empty(cls) -> "SessionIO":
        """Return an empty SessionIO that silently returns None for all queries."""
        return cls(
            trials_ep=nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float)),
            media=xr.Dataset(),
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _has_timing(self) -> bool:
        """True when ``trials_ep`` has valid non-empty session-absolute timing."""
        return (
            len(self.trials_ep) > 0
            and "trial" in self.trials_ep.metadata.columns
        )

    def trial_row(self, trial) -> int:
        """Row index of *trial* in ``trials_ep`` (0-based position)."""
        return self._trial_idx(trial)

    def trial_at_row(self, row_idx: int):
        """Trial ID at position *row_idx* in ``trials_ep``."""
        return self.trials_ep.metadata["trial"].iloc[row_idx]

    def _trial_idx(self, trial) -> int:
        if not hasattr(self, "_idx_cache"):
            metadata = self.trials_ep.metadata["trial"]
            object.__setattr__(self, "_idx_cache", {str(t): i for i, t in enumerate(metadata)})
        trial_str = str(trial)
        try:
            return self._idx_cache[trial_str]
        except KeyError:
            raise KeyError(f"Trial {trial} not found in SessionIO")


# ---------------------------------------------------------------------------

def _attrs_equal(a: Any, b: Any) -> bool:
    """Compare two attribute values safely, handling numpy arrays."""
    try:
        result = a == b
        if isinstance(result, np.ndarray):
            return result.all()
        return bool(result)
    except (ValueError, TypeError):
        return False


SESSION_NODE = "session"


def _select_from_da(
    da: xr.DataArray,
    trial=None,
    device: str | None = None,
) -> str | None:
    """Pick a scalar value from a media DataArray.
 
    Handles both per-trial (has 'trial' dim) and session-long layouts
    in one place — the branching that was duplicated across every old method.
    """
    try:
        sel: dict = {}
        if "trial" in da.dims and trial is not None:
            sel["trial"] = trial
 
        device_dims = [d for d in da.dims if d != "trial"]
        if device is not None and device_dims:
            sel[device_dims[0]] = device
 
        val = da.sel(**sel) if sel else da
        scalar = val.values.flat[0] if val.ndim > 0 else val.item()
        result = str(scalar)
        return result if result else None
    except (KeyError, ValueError, IndexError):
        return None
    
    

class TrialTree(xr.DataTree):
    """DataTree subclass with trial-specific functionality."""

    def __init__(self, data=None, children=None, name=None):
        """Initialize TrialTree from DataTree or other arguments."""
        if isinstance(data, xr.DataTree):
            super().__init__(dataset=data.ds, children=children, name=name)
            for child_name, child_node in data.children.items():
                self[child_name] = child_node
        else:
            super().__init__(dataset=data, children=children, name=name)

    # -------------------------------------------------------------------------
    # Node name resolution (handles both old trial_* and new bare names)
    # -------------------------------------------------------------------------

    def _trial_node_name(self, trial) -> str:
        trial_str = str(trial)
        if trial_str in self.children:
            return trial_str
        sanitized = trial_str.replace("/", "_")
        if sanitized != trial_str and sanitized in self.children:
            return sanitized
        legacy = f"trial_{trial_str}"
        if legacy in self.children:
            return legacy
        raise KeyError(f"No node found for trial {trial!r}")

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(self._trial_node_name(key))
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self._trial_node_name(key) if self._has_trial_node(key) else str(key)
        if isinstance(value, xr.Dataset):
            value = xr.DataTree(value)
        super().__setitem__(key, value)

    def _has_trial_node(self, trial) -> bool:
        trial_str = str(trial)
        sanitized = trial_str.replace("/", "_")
        return (
            trial_str in self.children
            or sanitized in self.children
            or f"trial_{trial_str}" in self.children
        )

    # -------------------------------------------------------------------------
    # Iteration helpers
    # -------------------------------------------------------------------------

    def trial_items(self):
        """Yield (trial_id, dataset) for each trial child node."""
        for node in self.children.values():
            if node.ds is not None and "trial" in node.ds.attrs:
                trial_id = node.ds.attrs["trial"]
                if hasattr(trial_id, 'item'):
                    trial_id = trial_id.item()
                yield trial_id, node.ds

    def map_trials(self, func: Callable[[xr.Dataset], xr.Dataset]) -> "TrialTree":
        def _apply(ds):
            if ds is None or "trial" not in ds.attrs:
                return ds
            return func(ds)
        return self.from_datatree(
            self.map_over_datasets(_apply),
            attrs=self.attrs,
        )

    def update_trial(self, trial, func: Callable[[xr.Dataset], xr.Dataset]) -> None:
        node_name = self._trial_node_name(trial)
        old_ds = self[node_name].ds
        self[node_name] = xr.DataTree(func(old_ds))

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def open(cls, path: str) -> "TrialTree":
        """Open TrialTree from a NetCDF file."""
        tree = xr.open_datatree(path, engine="netcdf4")
        tree.__class__ = cls
        tree._source_path = path
        return tree

    @classmethod
    def from_datasets(
        cls,
        datasets: List[xr.Dataset],
        session_table: xr.Dataset | pd.DataFrame | None = None,
        validate: bool = True,
    ) -> "TrialTree":
        """Create from list of datasets.

        Parameters
        ----------
        datasets
            Each must have attrs["trial"] set.
        session_table
            Session-level timing table with a ``trial`` dimension/index.
            Contains ``start_time``, ``stop_time``, and optional
            ``offset_<stream>`` columns.
        validate
            Whether to validate the tree after creation.
        """
        tree = cls()
        trials = []
        for ds in datasets:
            trial_num = ds.attrs.get('trial')
            if trial_num is None:
                raise ValueError("Each dataset must have 'trial' attribute")
            if trial_num in trials:
                raise ValueError(f"Duplicate trial number: {trial_num}")
            trials.append(trial_num)
            node_name = str(trial_num).replace("/", "_")
            tree[node_name] = xr.DataTree(ds)
        if session_table is not None:
            tree.set_session_table(session_table)
        if validate:
            tree._validate_tree()
        return tree

    @classmethod
    def from_datatree(cls, dt: xr.DataTree, attrs: dict | None = None) -> "TrialTree":
        tree = cls()
        for name, child in dt.children.items():
            tree[name] = child
        if dt.ds is not None:
            tree.ds = dt.ds
        tree.attrs = (attrs if attrs is not None else dt.attrs).copy()
        return tree

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        source_path = getattr(self, '_source_path', None)
        if path is None and source_path is None:
            raise ValueError("No path provided and no source path stored.")

        path = Path(path) if path else Path(source_path)
        temp_path = path.with_suffix('.tmp.nc')

        try:
            self.load()
            self.to_netcdf(temp_path, mode='w')
            self.close()
            temp_path.replace(path)
            self._source_path = str(path)
        finally:
            self.close()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def trials(self) -> List[int | str]:
        """Get list of trial numbers."""
        raw = [node.ds.attrs["trial"] for node in self.children.values() if node.ds is not None and "trial" in node.ds.attrs]
        trials = [val.item() if hasattr(val, 'item') else val for val in raw]
        if not trials:
            raise ValueError("No datasets with 'trial' attribute found in the tree.")
        return trials

    # -------------------------------------------------------------------------
    # Session table
    # -------------------------------------------------------------------------

    @property
    def session(self) -> xr.Dataset | None:
        """Access the session timing table, or None if not present."""
        if SESSION_NODE in self.children:
            return self[SESSION_NODE].ds
        return None

    @property
    def session_io(self) -> SessionIO | None:
        """Build a :class:`SessionIO` view from the current session node.

        Returns ``None`` when no session node exists.  Trial IDs without a
        known ``stop_time`` use :data:`_SENTINEL_END` as a placeholder; call
        ``session_io.stop_time(trial)`` which returns ``None`` for those.
        """
        sess = self.session
        if sess is None:
            return None

        trials_list = self.trials
        n = len(trials_list)
        trial_arr = np.array(trials_list)

        # Legacy compatibility
        starts = np.zeros(n, dtype=np.float64)
        if "start_time" in sess:
            for i, t in enumerate(trials_list):
                try:
                    v = float(sess["start_time"].sel(trial=t))
                    if not np.isnan(v):
                        starts[i] = v
                except (KeyError, ValueError):
                    pass
        # Legacy compatibility
        elif "video_start" in sess:
            da = sess["video_start"]
            for i, t in enumerate(trials_list):
                try:
                    row = da.sel(trial=t)
                    v = float(row.values.flat[0])
                    if not np.isnan(v):
                        starts[i] = v
                except (KeyError, ValueError):
                    pass

        ends = np.full(n, _SENTINEL_END, dtype=np.float64)
        if "stop_time" in sess:
            for i, t in enumerate(trials_list):
                try:
                    v = float(sess["stop_time"].sel(trial=t))
                    if not np.isnan(v):
                        ends[i] = v
                except (KeyError, ValueError):
                    pass

        # Track which ends were explicitly provided (used by stop_time()).
        has_stop = (ends < _SENTINEL_END).astype(float)

        # Only build a non-degenerate trials_ep when starts are strictly
        # monotonic.  If start_time is absent, all starts are 0.0 — filling
        # safe_ends with (next_start - gap) would produce end < start, which
        # pynapple silently drops, corrupting the metadata alignment.
        has_valid_timing = n > 0 and (n == 1 or bool(np.all(np.diff(starts) > 0)))
        if has_valid_timing:
            _GAP = 1e-4  # 100 µs — well below any real frame rate
            safe_ends = ends.copy()
            for i in range(n - 1):
                if safe_ends[i] >= _SENTINEL_END:
                    safe_ends[i] = starts[i + 1] - _GAP
            if safe_ends[-1] >= _SENTINEL_END:
                safe_ends[-1] = starts[-1] + 1.0
            trials_ep = nap.IntervalSet(
                start=starts,
                end=safe_ends,
                metadata={"trial": trial_arr, "has_stop": has_stop},
            )
        else:
            trials_ep = nap.IntervalSet(
                start=np.array([], dtype=np.float64),
                end=np.array([], dtype=np.float64),
            )

        media_vars = {
            k: sess[k]
            for k in sess.data_vars
            if k not in {"start_time", "stop_time"}
        }

        # Build stream_epochs from media layout + offset_* attrs:
        #   per-trial streams  → same intervals as trials_ep (seek = 0)
        #   session-wide       → single interval whose start = offset attr
        stream_epochs: dict[str, nap.IntervalSet] = {}
        for stream_name, da in sess.data_vars.items():
            if stream_name in {"start_time", "stop_time"}:
                continue
            if "trial" in da.dims:
                stream_epochs[stream_name] = trials_ep
            else:
                offset = float(sess.attrs.get(f"offset_{stream_name}", 0.0))
                stream_epochs[stream_name] = nap.IntervalSet(
                    start=[offset], end=[_SENTINEL_END]
                )

        extra_attrs = {k: v for k, v in sess.attrs.items() if not k.startswith("offset_")}

        return SessionIO(
            trials_ep=trials_ep,
            media=xr.Dataset(media_vars),
            stream_epochs=stream_epochs,
            attrs=extra_attrs,
        )

    def set_session_table(self, table: xr.Dataset | pd.DataFrame) -> None:
        """Set the session-level timing table.

        The table must have a ``trial`` dimension (Dataset) or index
        (DataFrame) containing trial identifiers that match the trial nodes.
        If ``start_time`` and/or ``stop_time`` columns are present they
        are preserved in the session node.
        """
        if isinstance(table, pd.DataFrame):
            if table.index.name != "trial":
                if "trial" in table.columns:
                    table = table.set_index("trial")
                else:
                    table.index.name = "trial"
            table = xr.Dataset.from_dataframe(table)
        self[SESSION_NODE] = xr.DataTree(table)
        
        

        
    def set_media_files(
        self,
        *,
        video=None,
        audio=None,
        pose=None,
        ephys: str | None = None,
        cameras: list[str] | None = None,
        mics: list[str] | None = None,
        per_trial: bool = True,
        video_start=None,
    ) -> None:
        """Store media file paths in the session node.

        video/audio/pose: list[list[str]] (per-trial) or list[str] (session-wide).
        ephys: single session-wide file path.
        cameras/mics: device label lists.
        per_trial: True = indexed by trial, False = session-wide file.
        video_start: dict {cam: offset_s} or float applied as offset_video attr.
        """
        self._ensure_session()
        session_ds = self[SESSION_NODE].to_dataset()

        def _store_stream(key, files, device_dim, device_labels):
            if key in ["video", "pose"]:
                device = "cam"
            elif key == "audio":
                device = "mic"
            
            if files is None:
                return
            if per_trial and isinstance(files, list) and files and isinstance(files[0], list):
                devices = device_labels or [f"{device}-{i+1}" for i in range(len(files[0]))]
                session_ds[key] = xr.DataArray(
                    files,
                    dims=["trial", device_dim],
                    coords={"trial": self.trials, device_dim: devices},
                )
            else:
                if isinstance(files, str):
                    files = [files]
                devices = device_labels or [f"{device}-1"]
                session_ds[key] = xr.DataArray(
                    files,
                    dims=[device_dim],
                    coords={device_dim: devices},
                )

        cam_labels = cameras
        mic_labels = mics
        _store_stream("video", video, "cameras", cam_labels)
        _store_stream("audio", audio, "mics", mic_labels)
        _store_stream("pose", pose, "cameras", cam_labels)

        if ephys is not None:
            session_ds["ephys"] = xr.DataArray(str(ephys))

        if video_start is not None:
            if isinstance(video_start, dict):
                for cam, offset in video_start.items():
                    session_ds.attrs[f"offset_video_{cam}"] = float(offset)
            else:
                session_ds.attrs["offset_video"] = float(video_start)

        self[SESSION_NODE] = xr.DataTree(session_ds)

    def set_ephys_stream_id(self, stream_id: str) -> None:
        """Store the selected Neo stream ID in session attrs."""
        self._ensure_session()
        session_ds = self[SESSION_NODE].to_dataset()
        session_ds.attrs["ephys_stream_id"] = stream_id
        self[SESSION_NODE] = xr.DataTree(session_ds)

    def set_stream_offset(self, stream: str, offset: float) -> None:
        """Set a global time offset for a data stream.

        For **per-trial** streams: ``offset`` is a small trial-relative
        alignment correction (e.g. ``0.2`` if audio lags video by 200 ms).

        For **session-wide** streams: ``offset`` is the session-absolute time
        (seconds) at which sample 0 of the file was recorded.  The most
        common value is ``0.0`` (recording started at session t=0).
        ``session_io.source_start_time(trial, stream)`` subtracts ``trial_start_abs``
        to produce a trial-relative start time for each trial.

        Parameters
        ----------
        stream
            Stream name (e.g. ``"audio"``, ``"ephys"``, ``"video"``).
        offset
            See above; meaning depends on whether the stream is per-trial
            or session-wide.
        """
        self._ensure_session()
        session_ds = self[SESSION_NODE].to_dataset()
        session_ds.attrs[f"offset_{stream}"] = offset
        self[SESSION_NODE] = xr.DataTree(session_ds)

    def _ensure_session(self) -> None:
        """Create an empty session node if one does not exist."""
        if SESSION_NODE not in self.children:
            self[SESSION_NODE] = xr.DataTree(xr.Dataset())

    def session_to_dataframe(self) -> pd.DataFrame | None:
        """Return the session table as a DataFrame, or None."""
        if self.session is None:
            return None
        return self.session.to_dataframe()

    def print_session(self) -> None:
        """Print all session data vars grouped by primary dimension, with attrs."""
        ds = self.session
        if ds is None:
            print("No session table.")
            return

        groups: dict[str, list[str]] = defaultdict(list)
        for name, var in ds.data_vars.items():
            key = var.dims[0] if var.dims else "(scalar)"
            groups[key].append(name)

        with xr.set_options(display_width=120, display_max_rows=100):
            for dim, names in groups.items():
                print(f"\n{'='*60}")
                print(f"  dim: {dim}")
                print(f"{'='*60}")
                for name in names:
                    print(f"\n  [{name}]")
                    print(ds[name].values)

        if ds.attrs:
            print(f"\n{'='*60}")
            print("  attrs")
            print(f"{'='*60}")
            for k, v in ds.attrs.items():
                print(f"  {k}: {v}")

    # -------------------------------------------------------------------------
    # Media files
    # -------------------------------------------------------------------------

    def _resolve_media_da(self, stream: str) -> xr.DataArray | None:
        if self.session is None or stream not in self.session:
            return None
        return self.session[stream]

    def get_media(
        self, trial, stream: str, device: str | None = None,
    ) -> str | None:
        """Get media filename for a trial and stream.
    
        >>> dt.get_media(1, "video", device="cam-1")
        'trial001_cam-1.mp4'
        """
        da = self._resolve_media_da(stream)
        if da is None:
            return None
        return _select_from_da(da, trial=trial, device=device)
    
    
    

    
    def devices(self, stream: str) -> list[str]:
        """Device labels for a stream, inferred from its non-trial dim.
    
        >>> dt.devices("video")   # ["cam-1", "cam-2"]
        >>> dt.devices("audio")   # ["mic-1", "mic-2"]
        """
        da = self._resolve_media_da(stream)
        if da is None:
            return []
        device_dims = [d for d in da.dims if d != "trial"]
        if not device_dims:
            return []
        return [str(v) for v in da.coords[device_dims[0]].values]

    @property
    def cameras(self) -> list[str]:
        """Camera coordinate labels from the session table."""
        if self.session is not None and "cameras" in self.session.coords:
            return [str(c) for c in self.session.coords["cameras"].values]
        return []

    @property
    def mics(self) -> list[str]:
        """Microphone coordinate labels from the session table."""
        if self.session is not None and "mics" in self.session.coords:
            return [str(m) for m in self.session.coords["mics"].values]
        return []


    # -------------------------------------------------------------------------
    # Trial access
    # -------------------------------------------------------------------------

    def trial(self, trial) -> xr.Dataset:
        ds = self[self._trial_node_name(trial)].ds
        if ds is None:
            raise ValueError(f"Trial {trial} has no dataset")
        return ds

    def itrial(self, trial_idx) -> xr.Dataset:
        """Index select from a specific trial dataset."""
        trial_nodes = [
            k for k in self.children
            if self.children[k].ds is not None and "trial" in self.children[k].ds.attrs
        ]
        if trial_idx >= len(trial_nodes):
            raise IndexError(f"Trial index {trial_idx} out of range")
        ds = self[trial_nodes[trial_idx]].ds
        if ds is None:
            raise ValueError(f"Trial at index {trial_idx} has no dataset")
        return ds

    def get_all_trials(self) -> Dict[int, xr.Dataset]:
        """Get all trials as a dictionary."""
        return {num: self.trial(num) for num in self.trials}

    def get_common_attrs(self) -> Dict[str, Any]:
        """Extract attributes common to all trial datasets."""
        trials_dict = self.get_all_trials()
        if not trials_dict:
            return {}
        common = dict(next(iter(trials_dict.values())).attrs)
        for ds in trials_dict.values():
            common = {
                k: v for k, v in common.items()
                if k in ds.attrs and _attrs_equal(ds.attrs[k], v)
            }
        return common

    # -------------------------------------------------------------------------
    # Label operations
    # -------------------------------------------------------------------------

    def get_label_dt(self, empty: bool = False, empty_confidence: bool = False) -> "TrialTree":
        def filter_node(ds):
            if ds is None:
                return xr.Dataset()
            

            orig_attrs = ds.attrs.copy()

            if "onset_s" in ds.data_vars and "segment" in ds.dims:
                if empty:
                    result = intervals_to_xr(empty_intervals())
                else:
                    interval_vars = [v for v in ("onset_s", "offset_s", "labels", "individual") if v in ds.data_vars]
                    result = ds[interval_vars].load()
                    if "labels_confidence" in ds.data_vars:
                        result["labels_confidence"] = ds["labels_confidence"].load()
                    elif empty_confidence:
                        result["labels_confidence"] = xr.DataArray(
                            np.ones(len(result["onset_s"])),
                            dims=["segment"],
                        )
                        
                result.attrs = orig_attrs
                return result

            result = xr.Dataset()
            result.attrs = orig_attrs
            return result

        tree = self.from_datatree(self.map_over_datasets(filter_node), attrs=self.attrs)
        tree.ds = xr.Dataset(attrs=tree.ds.attrs if tree.ds is not None else {})
        return tree

    def overwrite_with_attrs(self, labels_tree: xr.DataTree) -> "TrialTree":
        """Overwrite attrs in this tree with that from another tree."""
        def merge_func(self_ds, labels_ds):
            self_ds.attrs.update(labels_ds.attrs)
            return self_ds
        tree = self.map_over_datasets(merge_func, labels_tree)
        tree.attrs = labels_tree.attrs.copy()
        return TrialTree(tree)

    def overwrite_with_labels(self, labels_tree: xr.DataTree) -> "TrialTree":
        """Overwrite interval labels and attrs in this tree from another tree."""
        def merge_func(data_ds, labels_ds):
            if labels_ds is not None and data_ds is not None:
                tree = data_ds.copy()
                label_vars = list(labels_ds.data_vars)
                existing = [v for v in label_vars if v in tree]
                if existing:
                    tree = tree.drop_vars(existing)
                for var_name in label_vars:
                    tree[var_name] = labels_ds[var_name]
                tree.attrs.update(labels_ds.attrs)
                return tree
            return data_ds

        tree = self.map_over_datasets(merge_func, labels_tree)
        tree.attrs = labels_tree.attrs.copy()
        return TrialTree(tree)

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    def filter_by_attr(self, attr_name: str, attr_value: Any) -> "TrialTree":
        """Filter trials by attribute value with type conversion."""
        new_tree = xr.DataTree()

        def values_match(stored: Any, target: Any) -> bool:
            if stored == target:
                return True
            for coerce in (str, int, float):
                try:
                    return coerce(stored) == coerce(target)
                except (ValueError, TypeError):
                    continue
            return False

        for name, node in self.children.items():
            if node.ds and attr_name in node.ds.attrs:
                if values_match(node.ds.attrs[attr_name], attr_value):
                    new_tree[name] = node
        return TrialTree(new_tree)

    # -------------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------------

    def _validate_tree(self) -> List[str]:
        ds = self.itrial(0)
        has_cameras = len(self.cameras) > 0
        has_fps = "fps" in ds.attrs
        errors = validate_datatree(
            self,
            require_fps=has_fps or has_cameras,
        )
        if errors:
            error_msg = "Dataset validation failed:\n"
            error_msg += "\n".join(f"• {e}" for e in errors)
            raise ValueError("TrialTree validation failed:\n" + error_msg)
