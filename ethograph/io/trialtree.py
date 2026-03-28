from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

import pynapple as nap

from ethograph.labels.intervals import empty_intervals, intervals_to_xr
from ethograph.io.validation import validate_datatree


# ---------------------------------------------------------------------------
# Stream schema
# ---------------------------------------------------------------------------


class StreamLayout(Enum):
    PER_TRIAL = "per_trial"
    SESSION_WIDE = "session_wide"


@dataclass(frozen=True)
class StreamSpec:
    name: str
    layout: StreamLayout
    device_dim: str | None = None


STREAMS: dict[str, StreamSpec] = {
    "video": StreamSpec("video", StreamLayout.PER_TRIAL, device_dim="cameras"),
    "audio": StreamSpec("audio", StreamLayout.PER_TRIAL, device_dim="mics"),
    "pose": StreamSpec("pose", StreamLayout.PER_TRIAL, device_dim="cameras"),
    "ephys": StreamSpec("ephys", StreamLayout.SESSION_WIDE),
}

SESSION_NODE = "session"
_EPOCH_GAP = 1e-4  # 100 µs gap between inferred trial boundaries


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
    # Trial list & iteration
    # ------------------------------------------------------------------

    @property
    def trials(self) -> list[int | str]:
        """List of trial identifiers extracted from ``ds.attrs["trial"]``.

        Returns
        -------
        list[int | str]
            Trial identifiers from all child nodes.

        Raises
        ------
        ValueError
            If no child nodes contain a ``"trial"`` attribute.
        """
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
        """Iterate over ``(trial_id, dataset)`` pairs for all trial nodes.

        Yields
        ------
        trial_id : int or str
            The ``ds.attrs["trial"]`` value.
        ds : xarray.Dataset
        """
        for node in self.children.values():
            if node.ds is not None and "trial" in node.ds.attrs:
                trial_id = node.ds.attrs["trial"]
                if hasattr(trial_id, "item"):
                    trial_id = trial_id.item()
                yield trial_id, node.ds

    def map_trials(self, func: Callable[[xr.Dataset], xr.Dataset]) -> TrialTree:
        """Apply *func* to every trial dataset and return a new TrialTree.

        Parameters
        ----------
        func : callable
            ``func(ds) -> ds``.  Receives and returns an :class:`xarray.Dataset`.

        Returns
        -------
        TrialTree
            New tree with transformed trial datasets.
        """
        def _apply(ds):
            if ds is None or "trial" not in ds.attrs:
                return ds
            return func(ds)

        return self.from_datatree(self.map_over_datasets(_apply), attrs=self.attrs)

    def update_trial(self, trial, func: Callable[[xr.Dataset], xr.Dataset]) -> None:
        """Read-modify-write a single trial's dataset.

        Use this for structural changes (adding/removing variables) that
        do not propagate when mutating the dataset returned by :meth:`trial`.

        Parameters
        ----------
        trial : int or str
            Trial identifier.
        func : callable
            ``func(ds) -> ds``.  Receives the current dataset and must
            return the replacement.
        """
        node_name = self._trial_node_name(trial)
        self[node_name] = xr.DataTree(func(self[node_name].ds))

    # ------------------------------------------------------------------
    # Session table
    # ------------------------------------------------------------------

    @property
    def session(self) -> xr.Dataset | None:
        """The session-level :class:`xarray.Dataset`, or ``None``.

        Returns
        -------
        xarray.Dataset or None
            Session dataset if a ``"session"`` child node exists.
        """
        if SESSION_NODE in self.children:
            return self[SESSION_NODE].ds
        return None

    def _ensure_session(self) -> None:
        if SESSION_NODE not in self.children:
            self[SESSION_NODE] = xr.DataTree(xr.Dataset())

    def _update_session(self, func: Callable[[xr.Dataset], xr.Dataset]) -> None:
        self._ensure_session()
        self[SESSION_NODE] = xr.DataTree(func(self[SESSION_NODE].to_dataset()))

    def set_session_table(self, table: xr.Dataset | pd.DataFrame) -> None:
        """Store session-level metadata (timing, file paths, etc.).

        Parameters
        ----------
        table : xarray.Dataset or pandas.DataFrame
            Must be indexed (or have a column) named ``"trial"``.
        """
        if isinstance(table, pd.DataFrame):
            if table.index.name != "trial":
                if "trial" in table.columns:
                    table = table.set_index("trial")
                else:
                    table.index.name = "trial"
            table = xr.Dataset.from_dataframe(table)
        self[SESSION_NODE] = xr.DataTree(table)

    def session_to_dataframe(self) -> pd.DataFrame | None:
        """Return the session table as a :class:`~pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame or None
            Session table as a DataFrame, or ``None`` if no session exists.
        """
        if self.session is None:
            return None
        return self.session.to_dataframe()

    def print_session(self) -> None:
        """Print a formatted summary of the session table to stdout."""
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
                print(f"\n{'=' * 60}")
                print(f"  dim: {dim}")
                print(f"{'=' * 60}")
                for name in names:
                    print(f"\n  [{name}]")
                    print(ds[name].values)

        if ds.attrs:
            print(f"\n{'=' * 60}")
            print("  attrs")
            print(f"{'=' * 60}")
            for k, v in ds.attrs.items():
                print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _invalidate_timing_cache(self) -> None:
        for attr in ("_trials_ep", "_trial_idx_cache"):
            try:
                del self.__dict__[attr]
            except KeyError:
                pass

    @cached_property
    def _trials_ep(self) -> nap.IntervalSet | None:
        sess = self.session
        if sess is None:
            return None

        trials_list = self.trials
        n = len(trials_list)
        if n == 0:
            return None

        starts = np.zeros(n, dtype=np.float64)
        ends = np.full(n, np.nan, dtype=np.float64)

        if "start_time" in sess:
            for i, t in enumerate(trials_list):
                try:
                    v = float(sess["start_time"].sel(trial=t))
                    if not np.isnan(v):
                        starts[i] = v
                except (KeyError, ValueError):
                    pass

        if "stop_time" in sess:
            for i, t in enumerate(trials_list):
                try:
                    v = float(sess["stop_time"].sel(trial=t))
                    if not np.isnan(v):
                        ends[i] = v
                except (KeyError, ValueError):
                    pass

        has_monotonic_starts = n == 1 or bool(np.all(np.diff(starts) > 0))
        if not has_monotonic_starts:
            return None

        has_stop = ~np.isnan(ends)
        safe_ends = ends.copy()
        for i in range(n - 1):
            if np.isnan(safe_ends[i]):
                safe_ends[i] = starts[i + 1] - _EPOCH_GAP
        if np.isnan(safe_ends[-1]):
            safe_ends[-1] = starts[-1] + 1.0

        return nap.IntervalSet(
            start=starts,
            end=safe_ends,
            metadata={
                "trial": np.array(trials_list),
                "has_stop": has_stop.astype(float),
            },
        )

    @cached_property
    def _trial_idx_cache(self) -> dict[str, int]:
        ep = self._trials_ep
        if ep is None:
            return {}
        return {str(t): i for i, t in enumerate(ep.metadata["trial"])}

    def _trial_ep_idx(self, trial) -> int:
        idx = self._trial_idx_cache.get(str(trial))
        if idx is None:
            raise KeyError(f"Trial {trial} not found in timing table")
        return idx

    def start_time(self, trial) -> float:
        """Session-absolute start time of a trial in seconds.

        Parameters
        ----------
        trial : int or str
            Trial identifier.

        Returns
        -------
        float
            Start time in seconds, or 0.0 if no timing information is available.
        """
        ep = self._trials_ep
        if ep is None:
            return 0.0
        return float(ep.start[self._trial_ep_idx(trial)])

    def stop_time(self, trial) -> float | None:
        """Session-absolute stop time of a trial in seconds.

        Parameters
        ----------
        trial : int or str
            Trial identifier.

        Returns
        -------
        float or None
            Stop time in seconds, or ``None`` if no stop time is recorded.
        """
        ep = self._trials_ep
        if ep is None:
            return None
        idx = self._trial_ep_idx(trial)
        if not bool(ep.metadata["has_stop"].iloc[idx]):
            return None
        return float(ep.end[idx])

    def trial_duration(self, trial) -> float:
        """Duration of the trial in seconds (``stop_time - start_time``).

        Parameters
        ----------
        trial : int or str
            Trial identifier.

        Returns
        -------
        float
            Duration in seconds.

        Raises
        ------
        ValueError
            If the trial has no known stop time.
        """
        stop = self.stop_time(trial)
        if stop is None:
            raise ValueError(f"Trial {trial} has no known stop time")
        return stop - self.start_time(trial)

    @property
    def trials_ep(self) -> nap.IntervalSet | None:
        """All trial epochs as a single :class:`pynapple.IntervalSet`.

        Returns
        -------
        pynapple.IntervalSet or None
            Combined interval set for all trials, or ``None`` if no
            timing information is available.
        """
        return self._trials_ep

    def trial_epoch(self, trial) -> nap.IntervalSet:
        """Return the :class:`pynapple.IntervalSet` for a single trial.

        Parameters
        ----------
        trial : int or str
            Trial identifier.

        Returns
        -------
        pynapple.IntervalSet
            Single-interval set spanning the trial's start to end time.

        Raises
        ------
        ValueError
            If no timing information is available.
        """
        ep = self._trials_ep
        if ep is None:
            raise ValueError("No timing information available")
        idx = self._trial_ep_idx(trial)
        return nap.IntervalSet(start=ep.start[idx], end=ep.end[idx])

    def restrict(self, obj: nap.Tsd | nap.TsdFrame | nap.TsdTensor | nap.Ts | nap.TsGroup, trial) -> nap.Tsd | nap.TsdFrame | nap.TsdTensor | nap.Ts | nap.TsGroup:
        """Restrict a pynapple object to a trial's epoch.

        Parameters
        ----------
        obj : :class:`~pynapple.Tsd` | :class:`~pynapple.TsdFrame` | :class:`~pynapple.TsdTensor` | :class:`~pynapple.Ts` | :class:`~pynapple.TsGroup`
            Any pynapple time series or group.
        trial : int or str
            Trial identifier.

        Returns
        -------
        :class:`~pynapple.Tsd` | :class:`~pynapple.TsdFrame` | :class:`~pynapple.TsdTensor` | :class:`~pynapple.Ts` | :class:`~pynapple.TsGroup`
            The input restricted to the trial's epoch.
        """
        return obj.restrict(self.trial_epoch(trial))

    # ------------------------------------------------------------------
    # Stream offsets & source timing
    # ------------------------------------------------------------------

    def _stream_offset(self, stream: str) -> float:
        sess = self.session
        if sess is None:
            return 0.0
        return float(sess.attrs.get(f"offset_{stream}", 0.0))

    def source_start_time(self, trial, stream: str) -> float:
        """Trial-relative time of sample 0 for this stream's file.

        Per-trial streams (``"trial"`` in dims) always return 0.
        Session-wide streams return ``stream_offset - trial_start``.

        Parameters
        ----------
        trial : int or str
            Trial identifier.
        stream : str
            Stream name (e.g. ``"ephys"``, ``"audio"``).

        Returns
        -------
        float
            Time offset in seconds for sample 0 of the stream's file,
            relative to the trial's start.
        """
        sess = self.session
        if sess is None or stream not in sess:
            return 0.0
        if "trial" in sess[stream].dims:
            return 0.0
        ep = self._trials_ep
        if ep is None:
            return 0.0
        offset = self._stream_offset(stream)
        return offset - float(ep.start[self._trial_ep_idx(trial)])

    def set_stream_offset(self, stream: str, offset: float) -> None:
        """Set the session-absolute offset for a stream in seconds.

        Sample 0 of the stream's file is at this time in session-absolute
        coordinates.

        Parameters
        ----------
        stream : str
            Stream name (e.g. ``"ephys"``, ``"audio"``).
        offset : float
            Offset in seconds.
        """
        def _set(ds):
            ds.attrs[f"offset_{stream}"] = offset
            return ds

        self._update_session(_set)
        self._invalidate_timing_cache()

    def set_ephys_stream_id(self, stream_id: str) -> None:
        """Store the selected ephys stream identifier in the session.

        Parameters
        ----------
        stream_id : str
            Identifier for the selected ephys stream (e.g. Neo stream name).
        """
        def _set(ds):
            ds.attrs["ephys_stream_id"] = stream_id
            return ds

        self._update_session(_set)

    # ------------------------------------------------------------------
    # Media — schema-driven
    # ------------------------------------------------------------------

    def set_media(
        self,
        stream: str,
        files: str | list[str] | list[list[str]],
        device_labels: list[str] | None = None,
        per_trial: bool | None = None,
    ) -> None:
        """Store media file paths for a stream.

        Parameters
        ----------
        stream
            Must be a key in ``STREAMS``.
        files
            Per-trial with devices: ``list[list[str]]`` shaped (trials, devices).
            Per-trial single device or session-wide devices: ``list[str]``.
            Session-wide scalar: ``str``.
        device_labels
            Explicit labels; auto-generated if ``None`` and spec has a device dim.
        per_trial
            Override the stream's default layout.  ``None`` (default) uses
            ``StreamSpec.layout``.  Pass ``False`` to store a normally
            per-trial stream (e.g. video) as session-wide, or ``True``
            to store a normally session-wide stream as per-trial.
        """
        spec = STREAMS[stream]

        if spec.device_dim is None and device_labels:
            raise ValueError(f"Stream {stream!r} has no device dimension")

        is_per_trial = spec.layout is StreamLayout.PER_TRIAL if per_trial is None else per_trial
        has_device = spec.device_dim is not None

        if is_per_trial and isinstance(files, list) and files and isinstance(files[0], list):
            n_devices = len(files[0])
            labels = device_labels or [f"{spec.device_dim}-{i + 1}" for i in range(n_devices)]
            dims = ["trial", spec.device_dim]
            coords = {"trial": self.trials, spec.device_dim: labels}
        elif is_per_trial:
            if isinstance(files, str):
                files = [files]
            if has_device:
                labels = device_labels or [f"{spec.device_dim}-1"]
                dims = ["trial", spec.device_dim]
                arr = [[f] for f in files]
                coords = {"trial": self.trials, spec.device_dim: labels}
                files = arr
            else:
                dims = ["trial"]
                coords = {"trial": self.trials}
        elif isinstance(files, str):
            if has_device:
                labels = device_labels or [f"{spec.device_dim}-1"]
                dims = [spec.device_dim]
                coords = {spec.device_dim: labels}
                files = [files]
            else:
                dims = []
                coords = {}
        else:
            if has_device:
                labels = device_labels or [f"{spec.device_dim}-{i + 1}" for i in range(len(files))]
                dims = [spec.device_dim]
                coords = {spec.device_dim: labels}
            else:
                dims = []
                coords = {}

        def _set(ds):
            ds[stream] = xr.DataArray(np.array(files, dtype=str), dims=dims, coords=coords)
            return ds

        self._update_session(_set)

    def get_media(self, trial, stream: str, device: str | None = None) -> str | None:
        """Retrieve a media filename for a trial and stream.

        Parameters
        ----------
        trial : int or str
            Trial identifier.
        stream : str
            Stream name (``"video"``, ``"audio"``, ``"pose"``).
        device : str, optional
            Device label (e.g. ``"left"``).  Required when the stream has
            multiple devices.

        Returns
        -------
        str or None
            The filename string, or ``None`` if not found.
        """
        sess = self.session
        if sess is None or stream not in sess:
            return None
        da = sess[stream]
        spec = STREAMS.get(stream)

        sel: dict[str, Any] = {}
        if "trial" in da.dims:
            sel["trial"] = trial
        if device is not None and spec and spec.device_dim and spec.device_dim in da.dims:
            sel[spec.device_dim] = device

        try:
            val = da.sel(**sel) if sel else da
            return _scalar_or_none(val)
        except (KeyError, ValueError):
            return None

    def devices(self, stream: str) -> list[str]:
        """List device labels for a stream (e.g. camera names or mic names).

        Parameters
        ----------
        stream : str
            Stream name (``"video"``, ``"audio"``, ``"pose"``).

        Returns
        -------
        list[str]
            Empty list if the stream has no device dimension or no session data.
        """
        spec = STREAMS.get(stream)
        if spec is None or spec.device_dim is None:
            return []
        sess = self.session
        if sess is None or stream not in sess:
            return []
        da = sess[stream]
        if spec.device_dim not in da.dims:
            return []
        return [str(v) for v in da.coords[spec.device_dim].values]

    @property
    def cameras(self) -> list[str]:
        """Camera device labels (shortcut for ``devices("video")``)."""
        return self.devices("video")

    @property
    def mics(self) -> list[str]:
        """Microphone device labels (shortcut for ``devices("audio")``)."""
        return self.devices("audio")

    # ------------------------------------------------------------------
    # Trial data access
    # ------------------------------------------------------------------

    def trial(self, trial) -> xr.Dataset:
        """Return the dataset for the given trial ID.

        Parameters
        ----------
        trial : int or str
            Matches ``ds.attrs["trial"]``.

        Returns
        -------
        xr.Dataset
        """
        ds = self[self._trial_node_name(trial)].ds
        if ds is None:
            raise ValueError(f"Trial {trial} has no dataset")
        return ds

    def itrial(self, trial_idx: int) -> xr.Dataset:
        """Return the dataset at an integer index (0-based).

        Parameters
        ----------
        trial_idx : int
            Zero-based index into the ordered list of trial nodes.

        Returns
        -------
        xr.Dataset
        """
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
        """Return a dict mapping trial ID to :class:`xarray.Dataset` for all trials.

        Returns
        -------
        dict[int, xarray.Dataset]
            Mapping from trial identifier to its dataset.
        """
        return {num: self.trial(num) for num in self.trials}

    def get_common_attrs(self) -> dict[str, Any]:
        """Return attributes that are identical across all trials.

        Returns
        -------
        dict[str, Any]
            Attributes shared by every trial dataset.
        """
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
    # Label operations
    # ------------------------------------------------------------------

    def get_label_dt(self, empty: bool = False, empty_confidence: bool = False) -> TrialTree:
        """Return a TrialTree containing only label interval data.

        Strips all feature variables, keeping only onset_s / offset_s /
        labels / individual.  Dense legacy label arrays are auto-converted
        to interval format on the fly.

        Parameters
        ----------
        empty : bool
            If True, return an empty interval dataset for every trial.
        empty_confidence : bool
            If True and no ``labels_confidence`` column exists, inject a
            column of ones.

        Returns
        -------
        TrialTree
        """
        def filter_node(ds):
            if ds is None:
                return xr.Dataset()

            orig_attrs = ds.attrs.copy()

            if "onset_s" in ds.data_vars and "segment" in ds.dims:
                if empty:
                    result = intervals_to_xr(empty_intervals())
                else:
                    interval_vars = [
                        v for v in ("onset_s", "offset_s", "labels", "individual") if v in ds.data_vars
                    ]
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

    def overwrite_with_attrs(self, labels_tree: xr.DataTree) -> TrialTree:
        """Merge attributes (not data variables) from *labels_tree* into this tree.

        Parameters
        ----------
        labels_tree : xarray.DataTree
            Source tree whose ``ds.attrs`` are copied into each matching node.

        Returns
        -------
        TrialTree
            New tree with updated attributes.
        """
        def merge_func(self_ds, labels_ds):
            self_ds.attrs.update(labels_ds.attrs)
            return self_ds

        tree = self.map_over_datasets(merge_func, labels_tree)
        tree.attrs = labels_tree.attrs.copy()
        return TrialTree(tree)

    def overwrite_with_labels(self, labels_tree: xr.DataTree) -> TrialTree:
        """Merge label variables from *labels_tree* into this TrialTree.

        Replaces any existing label variables (``onset_s``, ``offset_s``,
        ``labels``, ``individual``) in each trial's dataset with those from
        *labels_tree*.  Returns a **new** tree; the original is unchanged.

        In the GUI this is called when the user clicks
        **"Merge labels and save session"**, rejoining the editable
        ``label_dt`` back into the main ``dt`` before overwriting the
        ``.nc`` file.

        Parameters
        ----------
        labels_tree : xarray.DataTree
            A label-only tree (typically from :meth:`get_label_dt`).

        Returns
        -------
        TrialTree
            New tree with updated labels merged in.

        Examples
        --------
        .. code-block:: python

            import ethograph as eto

            dt = eto.open("trials.nc")
            label_dt = dt.get_label_dt()

            # ... user edits labels in the GUI or programmatically ...

            dt_merged = dt.overwrite_with_labels(label_dt)
            dt_merged.save("trials.nc")
        """
        def merge_func(data_ds, labels_ds):
            if labels_ds is not None and data_ds is not None:
                tree = data_ds.copy()
                existing = [v for v in labels_ds.data_vars if v in tree]
                if existing:
                    tree = tree.drop_vars(existing)
                for var_name in labels_ds.data_vars:
                    tree[var_name] = labels_ds[var_name]
                tree.attrs.update(labels_ds.attrs)
                return tree
            return data_ds

        tree = self.map_over_datasets(merge_func, labels_tree)
        tree.attrs = labels_tree.attrs.copy()
        return TrialTree(tree)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_attr(self, attr_name: str, attr_value: Any) -> TrialTree:
        """Return a new TrialTree containing only trials that match an attribute.

        Parameters
        ----------
        attr_name : str
            ``ds.attrs`` key to filter on.
        attr_value
            Target value. Compared after coercion to str / int / float.

        Returns
        -------
        TrialTree
        """
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

        Parameters
        ----------
        path : str
            Path to a ``.nc`` file previously saved with :meth:`save`.

        Returns
        -------
        TrialTree
        """
        tree = xr.open_datatree(path, engine="netcdf4")
        tree.__class__ = cls
        tree._source_path = path
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
        datasets : list[xr.Dataset]
            Each dataset must have a unique ``attrs["trial"]`` key.
        session_table : xarray.Dataset or pandas.DataFrame, optional
            Session-level metadata indexed by trial ID.
        validate : bool
            Run :func:`~ethograph.io.validation.validate_datatree`
            after construction. Default True.

        Returns
        -------
        TrialTree
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
        if session_table is not None:
            tree.set_session_table(session_table)
        if validate:
            tree._validate_tree()
        return tree

    @classmethod
    def from_datatree(cls, dt: xr.DataTree, attrs: dict | None = None) -> TrialTree:
        """Wrap an existing :class:`xarray.DataTree` as a TrialTree.

        Parameters
        ----------
        dt : xarray.DataTree
            Source tree.
        attrs : dict, optional
            Override root attributes.  Defaults to ``dt.attrs``.

        Returns
        -------
        TrialTree
            New TrialTree wrapping the source tree's children and data.
        """
        tree = cls()
        for name, child in dt.children.items():
            tree[name] = child
        if dt.ds is not None:
            tree.ds = dt.ds
        tree.attrs = (attrs if attrs is not None else dt.attrs).copy()
        return tree

    def save(self, path: str | Path | None = None) -> None:
        """Write the TrialTree to a NetCDF file.

        Uses an atomic write (temp file then rename) to avoid partial writes.

        Parameters
        ----------
        path : str or Path, optional
            Destination path.  If None, overwrites the file the tree was
            loaded from.
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
        errors = validate_datatree(
            self,
            require_fps=has_fps or has_cameras,
        )
        if errors:
            raise ValueError(
                "TrialTree validation failed:\n" + "\n".join(f"• {e}" for e in errors)
            )
        return errors