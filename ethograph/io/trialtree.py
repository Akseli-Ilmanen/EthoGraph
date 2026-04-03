from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

import pynapple as nap

from ethograph.labels.intervals import empty_intervals
from ethograph.io.validation import validate_datatree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_or_none(da: xr.DataArray) -> str | None:
    val = str(da.values.flat[0]) if da.ndim > 0 else str(da.item())
    return val or None


def _attrs_equal(a: Any, b: Any) -> bool:
    try:
        result = a == b
        if isinstance(result, np.ndarray):
            return result.all()
        return bool(result)
    except (ValueError, TypeError):
        return False


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


# ---------------------------------------------------------------------------
# NWB discovery
# ---------------------------------------------------------------------------

_NWB_FILENAME = "alignment.nwb"
_SETTINGS_DIR = ".ethograph"


def _discover_nwb(nc_path: str | Path) -> Path | None:
    """Find an NWB session file near a .nc file.

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


# ---------------------------------------------------------------------------
# TrialTree
# ---------------------------------------------------------------------------


class TrialTree(xr.DataTree):

    def __init__(self, data=None, children=None, name=None):
        if isinstance(data, xr.DataTree):
            super().__init__(dataset=data.ds, children=children, name=name)
            for child_name, child_node in data.children.items():
                self[child_name] = child_node
        else:
            super().__init__(dataset=data, children=children, name=name)

    # ------------------------------------------------------------------
    # Node name resolution
    # ------------------------------------------------------------------

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

    def _has_trial_node(self, trial) -> bool:
        trial_str = str(trial)
        sanitized = trial_str.replace("/", "_")
        return (
            trial_str in self.children
            or sanitized in self.children
            or f"trial_{trial_str}" in self.children
        )

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

    # ------------------------------------------------------------------
    # SessionIO: NWB-backed session metadata
    # ------------------------------------------------------------------

    @property
    def nwb_path(self) -> str | None:
        return getattr(self, "_nwb_path", None)

    @nwb_path.setter
    def nwb_path(self, value: str | Path | None) -> None:
        old = getattr(self, "_nwb_path", None)
        self._nwb_path = str(value) if value else None
        if old != self._nwb_path:
            # Invalidate cached session_io
            self.__dict__.pop("session_io", None)

    @property
    def session_io(self):
        """Lazy-loaded SessionIO adapter backed by the NWB file."""
        cached = self.__dict__.get("session_io")
        if cached is not None:
            return cached

        from ethograph.io.session_io import NWBSessionIO, EmptySessionIO

        nwb_path = getattr(self, "_nwb_path", None)
        if nwb_path and Path(nwb_path).exists():
            sio = NWBSessionIO(nwb_path)
        else:
            sio = EmptySessionIO()
        self.__dict__["session_io"] = sio
        return sio

    # ------------------------------------------------------------------
    # Trial list & iteration
    # ------------------------------------------------------------------

    @property
    def trials(self) -> list[int | str]:
        """List of trial identifiers extracted from ``ds.attrs["trial"]``."""
        raw = [
            node.ds.attrs["trial"]
            for node in self.children.values()
            if node.ds is not None and "trial" in node.ds.attrs
        ]
        trials = [val.item() if hasattr(val, "item") else val for val in raw]
        if not trials:
            raise ValueError("No datasets with 'trial' attribute found in the tree.")
        return trials

    def trial_items(self):
        """Iterate over ``(trial_id, dataset)`` pairs for all trial nodes."""
        for node in self.children.values():
            if node.ds is not None and "trial" in node.ds.attrs:
                trial_id = node.ds.attrs["trial"]
                if hasattr(trial_id, "item"):
                    trial_id = trial_id.item()
                yield trial_id, node.ds

    def map_trials(self, func: Callable[[xr.Dataset], xr.Dataset]) -> TrialTree:
        """Apply *func* to every trial dataset and return a new TrialTree."""
        def _apply(ds):
            if ds is None or "trial" not in ds.attrs:
                return ds
            return func(ds)

        return self.from_datatree(
            self.map_over_datasets(_apply), attrs=self.attrs, source=self,
        )

    def update_trial(self, trial, func: Callable[[xr.Dataset], xr.Dataset]) -> None:
        """Read-modify-write a single trial's dataset."""
        node_name = self._trial_node_name(trial)
        self[node_name] = xr.DataTree(func(self[node_name].ds))

    # ------------------------------------------------------------------
    # Delegated session access (reads from NWB via session_io)
    # ------------------------------------------------------------------

    @property
    def cameras(self) -> list[str]:
        """Camera device labels."""
        return self.session_io.cameras

    @property
    def mics(self) -> list[str]:
        """Microphone device labels."""
        return self.session_io.mics

    def devices(self, stream: str) -> list[str]:
        """List device labels for a stream."""
        return self.session_io.devices(stream)

    def get_media(self, trial, stream: str, device: str | None = None) -> str | None:
        """Retrieve a media filename for a trial and stream."""
        return self.session_io.get_media(trial, stream, device)

    def start_time(self, trial) -> float:
        """Session-absolute start time of a trial in seconds."""
        return self.session_io.start_time(trial)

    def stop_time(self, trial) -> float | None:
        """Session-absolute stop time of a trial in seconds."""
        return self.session_io.stop_time(trial)

    def trial_duration(self, trial) -> float:
        """Duration of the trial in seconds."""
        return self.session_io.trial_duration(trial)

    def source_start_time(self, trial, stream: str, device: str | None = None) -> float:
        """Session-absolute time of sample 0 for this stream's file."""
        return self.session_io.source_start_time(trial, stream, device)

    def source_start_time_trial_relative(self, trial, stream: str, device: str | None = None) -> float:
        """Trial-relative time of sample 0 for this stream's file."""
        return self.session_io.source_start_time_trial_relative(trial, stream, device)

    def get_video_fps(self, camera: str | None = None) -> float | None:
        """Return video FPS, checking NWB acquisition then trial attrs."""
        fps = self.session_io.get_video_fps(camera)
        if fps is not None:
            return fps
        # Fallback to trial dataset attrs
        try:
            ds = self.itrial(0)
            if "fps" in ds.attrs:
                return float(ds.attrs["fps"])
        except (StopIteration, IndexError):
            pass
        return None

    def set_video_fps(self, fps: float, camera: str | None = None) -> None:
        """Store detected video FPS in session_io's in-memory overlay."""
        self.session_io.set_video_fps(fps, camera)

    @property
    def trials_ep(self) -> nap.IntervalSet | None:
        """All trial epochs as a pynapple IntervalSet."""
        return self.session_io.trials_ep

    def trial_epoch(self, trial) -> nap.IntervalSet:
        """Return the IntervalSet for a single trial."""
        return self.session_io.trial_epoch(trial)

    def restrict(self, obj, trial):
        """Restrict a pynapple object to a trial's epoch."""
        return self.session_io.restrict(obj, trial)

    def print_session(self) -> None:
        """Print a formatted summary of the session metadata."""
        self.session_io.print_session()

    # ------------------------------------------------------------------
    # Backward compat: session property (returns None for NWB-backed trees)
    # ------------------------------------------------------------------

    @property
    def session(self) -> xr.Dataset | None:
        """Legacy session access. Returns None for NWB-backed trees."""
        if "session" in self.children:
            return self["session"].ds
        return None

    # ------------------------------------------------------------------
    # Trial data access
    # ------------------------------------------------------------------

    def trial(self, trial) -> xr.Dataset:
        """Return the dataset for the given trial ID."""
        ds = self[self._trial_node_name(trial)].ds
        if ds is None:
            raise ValueError(f"Trial {trial} has no dataset")
        return ds

    def itrial(self, trial_idx: int) -> xr.Dataset:
        """Return the dataset at an integer index (0-based)."""
        trial_nodes = [
            k
            for k in self.children
            if self.children[k].ds is not None and "trial" in self.children[k].ds.attrs
        ]
        if trial_idx >= len(trial_nodes):
            raise IndexError(f"Trial index {trial_idx} out of range")
        ds = self[trial_nodes[trial_idx]].ds
        if ds is None:
            raise ValueError(f"Trial at index {trial_idx} has no dataset")
        return ds

    def get_all_trials(self) -> dict[int, xr.Dataset]:
        """Return a dict mapping trial ID to Dataset for all trials."""
        return {num: self.trial(num) for num in self.trials}

    def get_common_attrs(self) -> dict[str, Any]:
        """Return attributes that are identical across all trials."""
        trials_dict = self.get_all_trials()
        if not trials_dict:
            return {}
        common = dict(next(iter(trials_dict.values())).attrs)
        for ds in trials_dict.values():
            common = {
                k: v
                for k, v in common.items()
                if k in ds.attrs and _attrs_equal(ds.attrs[k], v)
            }
        return common

    # ------------------------------------------------------------------
    # Label operations (labels are now stored in TSV files, not in .nc)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_attr(self, attr_name: str, attr_value: Any) -> TrialTree:
        """Return a new TrialTree containing only trials that match an attribute."""
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: str) -> TrialTree:
        """Load a TrialTree from a NetCDF file.

        Auto-discovers ``.ethograph/alignment.nwb`` next to the file.
        """
        tree = xr.open_datatree(path, engine="netcdf4")
        tree.__class__ = cls
        tree._source_path = path
        # Auto-discover NWB session file
        nwb = _discover_nwb(path)
        if nwb is not None:
            tree._nwb_path = str(nwb)
        return tree

    @classmethod
    def from_datasets(
        cls,
        datasets: list[xr.Dataset],
        session_table: xr.Dataset | pd.DataFrame | None = None,
        validate: bool = True,
    ) -> TrialTree:
        """Build a TrialTree from a list of xarray Datasets.

        Parameters
        ----------
        datasets
            Each dataset must have a unique ``attrs["trial"]`` key.
        session_table
            Deprecated. Use NWB files for session metadata.
        validate
            Run validation after construction.
        """
        tree = cls()
        seen: set = set()
        for ds in datasets:
            trial_num = ds.attrs.get("trial")
            if trial_num is None:
                raise ValueError("Each dataset must have 'trial' attribute")
            if trial_num in seen:
                raise ValueError(f"Duplicate trial number: {trial_num}")
            seen.add(trial_num)
            node_name = str(trial_num).replace("/", "_")
            tree[node_name] = xr.DataTree(ds)
        if validate:
            tree._validate_tree()
        return tree

    @classmethod
    def from_datatree(cls, dt: xr.DataTree, attrs: dict | None = None,
                      *, source: "TrialTree | None" = None) -> TrialTree:
        """Wrap an existing DataTree as a TrialTree.

        Parameters
        ----------
        source
            Original TrialTree to copy ``_nwb_path`` and ``_source_path`` from.
        """
        tree = cls()
        for name, child in dt.children.items():
            tree[name] = child
        if dt.ds is not None:
            tree.ds = dt.ds
        tree.attrs = (attrs if attrs is not None else dt.attrs).copy()
        if source is not None:
            nwb = getattr(source, "_nwb_path", None)
            if nwb is not None:
                tree._nwb_path = nwb
            sp = getattr(source, "_source_path", None)
            if sp is not None:
                tree._source_path = sp
        return tree

    def save(self, path: str | Path | None = None) -> None:
        """Write the TrialTree to a NetCDF file.

        Uses an atomic write (temp file then rename) to avoid partial writes.
        """
        source_path = getattr(self, "_source_path", None)
        if path is None and source_path is None:
            raise ValueError("No path provided and no source path stored.")

        path = Path(path) if path else Path(source_path)
        temp_path = path.with_suffix(".tmp.nc")

        try:
            self.load()
            self.to_netcdf(temp_path, mode="w")
            self.close()
            temp_path.replace(path)
            self._source_path = str(path)
        finally:
            self.close()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_tree(self) -> list[str]:
        ds = self.itrial(0)
        has_cameras = len(self.cameras) > 0
        has_fps = "fps" in ds.attrs
        has_session_fps = self.get_video_fps() is not None
        errors = validate_datatree(
            self,
            require_fps=(has_fps or has_cameras) and not has_session_fps,
        )
        if errors:
            raise ValueError(
                "TrialTree validation failed:\n" + "\n".join(f"• {e}" for e in errors)
            )
        return errors
