from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

import pynapple as nap

from ethograph.io.validation import validate_datatree
from ethograph.io.nwb_alignment import discover_nwb, make_nwb_alignment, EmpytAlignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def _attrs_equal(a: Any, b: Any) -> bool:
    try:
        result = a == b
        if isinstance(result, np.ndarray):
            return result.all()
        return bool(result)
    except (ValueError, TypeError):
        return False



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
        # Always initialize nwb_alignment to a null-object (empty EmpytAlignment)
        if not hasattr(self, "nwb_alignment"):
            self.nwb_alignment = EmpytAlignment()

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
    # Continuous mode support
    # ------------------------------------------------------------------

    @property
    def _is_continuous(self) -> bool:
        """True when backed by a single shared dataset + epoch boundaries."""
        return getattr(self, "_continuous_ds", None) is not None

    def _slice_continuous(self, trial_id) -> xr.Dataset:
        """Slice the continuous dataset for a single trial, shifting time to 0."""
        start, stop = self._trial_epochs[trial_id]
        ds = self._continuous_ds

        time_dims = [d for d in ds.dims if "time" in d.lower()]

        sel = {dim: slice(start, stop) for dim in time_dims}
        sliced = ds.sel(sel)

        for dim in time_dims:
            if dim in sliced.coords:
                sliced = sliced.assign_coords(
                    {dim: sliced.coords[dim].values - start}
                )

        sliced.attrs = dict(ds.attrs)
        sliced.attrs["trial"] = trial_id
        return sliced

    # ------------------------------------------------------------------
    # Trial list & iteration
    # ------------------------------------------------------------------

    @property
    def trials(self) -> list[int | str]:
        """List of trial identifiers."""
        if self._is_continuous:
            return sorted(self._trial_epochs.keys())

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
        if self._is_continuous:
            for trial_id in sorted(self._trial_epochs.keys()):
                yield trial_id, self._slice_continuous(trial_id)
            return

        for node in self.children.values():
            if node.ds is not None and "trial" in node.ds.attrs:
                trial_id = node.ds.attrs["trial"]
                if hasattr(trial_id, "item"):
                    trial_id = trial_id.item()
                yield trial_id, node.ds

    def map_trials(self, func: Callable[[xr.Dataset], xr.Dataset]) -> TrialTree:
        """Apply *func* to every trial dataset and return a new TrialTree.

        For continuous trees, materialises sliced datasets into a
        standard per-node TrialTree so *func* results can be stored.
        """
        if self._is_continuous:
            datasets = [func(self._slice_continuous(t)) for t in sorted(self._trial_epochs.keys())]
            return TrialTree.from_datasets(datasets, validate=False)

        def _apply(ds):
            if ds is None or "trial" not in ds.attrs:
                return ds
            return func(ds)

        return self.from_datatree(
            self.map_over_datasets(_apply), attrs=self.attrs, source=self,
        )

    def update_trial(self, trial, func: Callable[[xr.Dataset], xr.Dataset]) -> None:
        """Read-modify-write a single trial's dataset.

        Not supported for continuous trees — call :meth:`materialise`
        first if you need per-trial mutation.
        """
        if self._is_continuous:
            raise TypeError(
                "Cannot update trials in a continuous TrialTree. "
            )
        node_name = self._trial_node_name(trial)
        self[node_name] = xr.DataTree(func(self[node_name].ds))



    # ------------------------------------------------------------------
    # Trial data access
    # ------------------------------------------------------------------

    def trial(self, trial) -> xr.Dataset:
        """Return the dataset for the given trial ID.

        Parameters
        ----------
        trial : int or str
            Trial identifier matching ``ds.attrs["trial"]``.

        Examples
        --------
        >>> import ethograph as eto
        >>> dt = eto.open("session.nc")
        >>> ds = dt.trial(1)
        >>> ds.attrs["trial"]
        1
        >>> ds["speed"]  # access a feature variable
        <xarray.DataArray 'speed' (time: 9000, keypoints: 4)>
        """
        if self._is_continuous:
            if trial not in self._trial_epochs:
                raise KeyError(f"No epoch found for trial {trial!r}")
            return self._slice_continuous(trial)
        ds = self[self._trial_node_name(trial)].ds
        if ds is None:
            raise ValueError(f"Trial {trial} has no dataset")
        return ds

    def itrial(self, trial_idx: int) -> xr.Dataset:
        """Return the dataset at an integer index (0-based).

        Parameters
        ----------
        trial_idx : int
            Zero-based index into the list of trials.

        Examples
        --------
        >>> import ethograph as eto
        >>> dt = eto.open("session.nc")
        >>> dt.trials
        [1, 2, 3]
        >>> ds = dt.itrial(0)   # same as dt.trial(1)
        >>> ds.attrs["trial"]
        1
        >>> ds = dt.itrial(2)   # same as dt.trial(3)
        >>> ds.attrs["trial"]
        3
        """
        if self._is_continuous:
            trial_ids = sorted(self._trial_epochs.keys())
            if trial_idx >= len(trial_ids):
                raise IndexError(f"Trial index {trial_idx} out of range")
            return self._slice_continuous(trial_ids[trial_idx])

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
        """Return a new TrialTree containing only trials that match an attribute.

        Checks the metadata table first; falls back to ``ds.attrs``.
        """
        def values_match(stored: Any, target: Any) -> bool:
            if stored == target:
                return True
            for coerce in (str, int, float):
                try:
                    return coerce(stored) == coerce(target)
                except (ValueError, TypeError):
                    continue
            return False

        mdf = getattr(self, "_metadata_df", None)
        if mdf is not None and attr_name in mdf.columns:
            matching_trials = set()
            for _, row in mdf.iterrows():
                if values_match(row[attr_name], attr_value):
                    matching_trials.add(row["trial"])
            new_tree = xr.DataTree()
            for name, node in self.children.items():
                if node.ds and node.ds.attrs.get("trial") in matching_trials:
                    new_tree[name] = node
            result = TrialTree(new_tree)
            result._metadata_df = mdf[mdf["trial"].isin(result.trials)].reset_index(drop=True)
            return result

        new_tree = xr.DataTree()
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
        nwb = discover_nwb(path)
        tree.nwb_alignment = make_nwb_alignment(nwb)
        return tree

    @classmethod
    def from_datasets(
        cls,
        datasets: list[xr.Dataset],
        validate: bool = True,
    ) -> TrialTree:
        """Build a TrialTree from a list of xarray Datasets.

        Parameters
        ----------
        datasets
            Each dataset must have a unique ``attrs["trial"]`` key.
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
        # Ensure nwb_alignment is always set (inherited from __init__, but explicit)
        if not hasattr(tree, "nwb_alignment") or tree.nwb_alignment is None:
            tree.nwb_alignment = EmpytAlignment()
        if validate:
            tree._validate_tree()
        return tree

    @classmethod
    def from_continuous(
        cls,
        ds: xr.Dataset,
        epochs: "pd.DataFrame | nap.IntervalSet",
    ) -> TrialTree:
        """Build a TrialTree from a single continuous recording + trial epochs.

        Unlike :meth:`from_datasets` which requires pre-split data, this
        stores one shared dataset and slices on demand when :meth:`trial`
        is called.  Time coordinates are shifted to 0 per trial.

        Parameters
        ----------
        ds
            Full recording dataset.  Must have at least one dimension
            whose name contains ``"time"``.
        epochs
            Trial boundaries.  Accepts:

            - ``pd.DataFrame`` with columns ``trial``, ``start_time``,
              ``stop_time``.
            - ``nap.IntervalSet`` — trial IDs taken from a ``"trial"``
              metadata column, or ``1, 2, …`` if absent.

        Examples
        --------
        >>> import pandas as pd
        >>> epochs = pd.DataFrame({
        ...     "trial": [1, 2, 3],
        ...     "start_time": [0.0, 60.0, 120.0],
        ...     "stop_time": [60.0, 120.0, 180.0],
        ... })
        >>> dt = TrialTree.from_continuous(ds, epochs)
        >>> dt.trial(2)  # returns 60-120s slice, time shifted to 0
        """
        tree = cls()

        if isinstance(epochs, nap.IntervalSet):
            epoch_dict = {}
            has_trial = "trial" in epochs.metadata.columns
            for i in range(len(epochs)):
                trial_id = (
                    epochs.metadata["trial"].iloc[i] if has_trial else i + 1
                )
                if hasattr(trial_id, "item"):
                    trial_id = trial_id.item()
                epoch_dict[trial_id] = (float(epochs.start[i]), float(epochs.end[i]))
                
                
        elif isinstance(epochs, pd.DataFrame):
            epoch_dict = {}
            
            if "trial" in epochs.columns:
                trials_col = epochs["trial"].values
            else:
                trials_col = None

            starts_col = epochs["start_time"].values
            stops_col = epochs["stop_time"].values
            for i in range(len(epochs)):
                trial_id = trials_col[i] if trials_col is not None else i + 1
                if hasattr(trial_id, "item"):
                    trial_id = trial_id.item()
                epoch_dict[trial_id] = (float(starts_col[i]), float(stops_col[i]))
        else:
            raise ValueError("epochs must be a pandas DataFrame or pynapple IntervalSet")

        tree._continuous_ds = ds
        tree._trial_epochs = epoch_dict
        
        # Ensure nwb_alignment is always set (inherited from __init__, but explicit)
        if not hasattr(tree, "nwb_alignment") or tree.nwb_alignment is None:
            tree.nwb_alignment = EmpytAlignment()
        return tree

    @classmethod
    def from_datatree(cls, dt: xr.DataTree, attrs: dict | None = None,
                      *, source: "TrialTree | None" = None) -> TrialTree:
        """Wrap an existing DataTree as a TrialTree.

        Parameters
        ----------
        source
            Original TrialTree to copy ``nwb_alignment`` and ``_source_path`` from.
        """
        tree = cls()
        for name, child in dt.children.items():
            tree[name] = child
        if dt.ds is not None:
            tree.ds = dt.ds
        tree.attrs = (attrs if attrs is not None else dt.attrs).copy()
        if source is not None:
            tree.nwb_alignment = source.nwb_alignment
            sp = getattr(source, "_source_path", None)
            if sp is not None:
                tree._source_path = sp
            mdf = getattr(source, "_metadata_df", None)
            if mdf is not None:
                tree._metadata_df = mdf
        return tree

    def save(self, path: str | Path | None = None) -> None:
        """Write the TrialTree to a NetCDF file.

        Continuous trees are materialised into per-trial nodes before
        saving so the file can be re-opened with :meth:`open`.

        Uses an atomic write (temp file then rename) to avoid partial writes.
        If an alignment NWB exists and the save directory differs from where
        the NWB lives, a copy is placed in ``<save_dir>/.ethograph/`` so that
        :meth:`open` can discover it.
        """
        if self._is_continuous:
            self.materialise().save(path)
            return

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
            self._ensure_alignment_nwb(path.parent)
        finally:
            self.close()

    def _ensure_alignment_nwb(self, save_dir: Path) -> None:
        """Copy alignment NWB next to the save location if needed."""
        import shutil
        from ethograph.io.nwb_alignment import NWBAlignment, _NWB_FILENAME, _SETTINGS_DIR

        sio = self.nwb_alignment
        if not isinstance(sio, NWBAlignment):
            return
        nwb = str(sio._path)
        if not Path(nwb).exists():
            return

        target = save_dir / _SETTINGS_DIR / _NWB_FILENAME
        if target.exists() or Path(nwb).resolve() == target.resolve():
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nwb, target)
        self.nwb_alignment = make_nwb_alignment(target)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_tree(self) -> list[str]:
        ds = self.itrial(0)
        sio = self.nwb_alignment
        errors = validate_datatree(self)
        if errors:
            raise ValueError(
                "TrialTree validation failed:\n" + "\n".join(f"• {e}" for e in errors)
            )
        return errors




    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    @property
    def session(self) -> xr.Dataset | None:
        """Legacy session access. Returns None for NWB-backed trees."""
        if "session" in self.children:
            return self["session"].ds
        return None


    def get_trial_metadata(self, trial) -> dict:
        """Return condition metadata for a single trial as a dict."""
        df = getattr(self, "_metadata_df", None)
        if df is None:
            return {}
        if df.empty:
            return {}
        row = df[df["trial"] == trial]
        if row.empty:
            row = df[df["trial"] == str(trial)]
        if row.empty and isinstance(trial, (int, float)):
            row = df[df["trial"].astype(str) == str(int(trial))]
        if row.empty:
            return {}
        return {k: v for k, v in row.iloc[0].to_dict().items() if k != "trial" and pd.notna(v)}