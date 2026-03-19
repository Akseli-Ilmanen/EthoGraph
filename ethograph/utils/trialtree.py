from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from ethograph.utils.label_intervals import (
    empty_intervals,
    intervals_to_xr,
)
from ethograph.utils.validation import validate_datatree


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
        
        
    def get_start_time(self, trial) -> float:
        """Get session-absolute start time for a trial.
        """
        if self.session is not None and "start_time" in self.session:
            try:
                return float(self.session.start_time.sel(trial=trial))
            except (KeyError, ValueError):
                pass
            
        # Compatibility to legacy code.
        ds = self.trial(trial)
        if "video_onset" in ds.attrs:
            return float(ds.attrs["video_onset"])
        elif "trial_onset" in ds.attrs:
            return float(ds.attrs["trial_onset"])


        return 0.0
        
        
    def get_stop_time(self, trial) -> float | None:
        """Get session-absolute stop time for a trial.
        """
        if self.session is not None and "stop_time" in self.session:
            try:
                return float(self.session.stop_time.sel(trial=trial))
            except (KeyError, ValueError):
                pass
        
        return None

    # TODO: rewrite to work with offset_video, or offset_video_camera -> or better way to do this?
    def set_stream_offset(self, stream: str, offset: float) -> None:
        """Set a global time offset for a data stream.

        The offset is the time in seconds at which this stream starts
        relative to the reference clock (typically video).  Stored as a
        session-level attribute so it applies to all trials.

        Parameters
        ----------
        stream
            Stream name (e.g. ``"audio"``, ``"ephys"``, ``"video"``).
        offset
            Offset in seconds.  Positive means the stream starts *after*
            the reference; negative means *before*.
        """
        self._ensure_session()
        session_ds = self[SESSION_NODE].to_dataset()
        session_ds.attrs[f"offset_{stream}"] = offset
        self[SESSION_NODE] = xr.DataTree(session_ds)


    def get_stream_offset(self, trial, stream: str) -> float | None:
        """Get offset for a data stream in a trial.

        Lookup order:
        1. Global session attr ``offset_<stream>`` (set via
           :meth:`set_stream_offset`).
        2. Per-trial session column ``offset_<stream>`` (legacy /
           per-trial override).
        3. Per-trial dataset attr ``offset_<stream>``.
        4. Default ``0.0``.

        Returns ``None`` when the stream is explicitly marked
        unavailable (NaN in the per-trial column).
        """
        col = f"offset_{stream}"
        if self.session is not None:
            # 1. Global attr
            if col in self.session.attrs:
                return float(self.session.attrs[col])
            # 2. Per-trial column
            if col in self.session:
                try:
                    val = float(self.session[col].sel(trial=trial))
                    return None if np.isnan(val) else val
                except (KeyError, ValueError):
                    pass
        # 3. Per-trial dataset attr
        ds = self.trial(trial)
        return float(ds.attrs.get(col, 0.0))

    def _ensure_session(self) -> None:
        """Create an empty session node if one does not exist."""
        if SESSION_NODE not in self.children:
            self[SESSION_NODE] = xr.DataTree(xr.Dataset())

    def session_to_dataframe(self) -> pd.DataFrame | None:
        """Return the session table as a DataFrame, or None."""
        if self.session is None:
            return None
        return self.session.to_dataframe()

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
