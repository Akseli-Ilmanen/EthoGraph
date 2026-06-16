"""Unified combo catalog and loader for xarray, pynapple backends.

Replaces the old type_vars_dict pattern with:
- DataCatalog: what dimensions/features are available (builds combo boxes)
- DataLoader: how to load data (select by feature + combo dims + time window → PlotData)

Three backends, same interface. Differs in how combo dims are discovered,
how selection works (sel_valid principle: overspecified combos are OK), and
how time slicing works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from ethograph.io.trialtree import TrialTree


# ---------------------------------------------------------------------------
# PlotData — universal rendering output: (T,) or (T, D)
# ---------------------------------------------------------------------------


@dataclass
class PlotData:
    """Source-agnostic data ready for rendering."""

    time: np.ndarray
    data: np.ndarray  # (T,) or (T, D)
    dim_labels: list[str] | None = None
    title: str = ""
    ylabel: str = ""
    color_data: np.ndarray | None = None
    changepoints: dict[str, np.ndarray] | None = None


# ---------------------------------------------------------------------------
# ComboSpec + DataCatalog — what's available
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComboSpec:
    """A selectable dimension: name + allowed values."""

    name: str
    values: tuple[str, ...]


@dataclass
class DataCatalog:
    """Unified catalog of available features, dimensions, and streams.

    Built by ``catalog_from_xarray`` / ``catalog_from_pynapple`` /
    ``catalog_from_nwb``.  The GUI creates combo boxes from ``combos``
    and uses the loader returned alongside this catalog for data access.
    """

    combos: dict[str, ComboSpec] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    changepoints: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    mics: list[str] = field(default_factory=list)
    trial_conditions: list[str] = field(default_factory=list)

    def combo_values(self, name: str) -> tuple[str, ...]:
        spec = self.combos.get(name)
        return spec.values if spec else ()

    def to_type_vars_dict(self) -> dict:
        """Backwards-compatible dict for existing GUI combo creation."""
        tvd: dict[str, Any] = {}
        for name, spec in self.combos.items():
            tvd[name] = np.array(spec.values) if spec.values else []
        tvd["features"] = self.features
        if self.changepoints:
            tvd["changepoints"] = self.changepoints
        if self.cameras:
            tvd["cameras"] = np.array(self.cameras)
        if self.mics:
            tvd["mics"] = np.array(self.mics)
        tvd["trial_conditions"] = self.trial_conditions
        return tvd


# ---------------------------------------------------------------------------
# NWB data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeSeriesRecord:
    """Metadata for one TimeSeries discovered inside an NWB file."""

    path: str
    name: str
    neurodata_type: str
    description: str
    shape: tuple[int, ...]
    dtype: str
    unit: str
    rate: float | None
    starting_time: float | None
    n_timestamps: int | None
    timestamps_range: tuple[float, float] | None

    @property
    def duration(self) -> float | None:
        if self.rate and self.starting_time is not None:
            return self.shape[0] / self.rate
        if self.timestamps_range:
            return self.timestamps_range[1] - self.timestamps_range[0]
        return None

    @property
    def time_range(self) -> tuple[float, float] | None:
        if self.timestamps_range:
            return self.timestamps_range
        if self.rate and self.starting_time is not None:
            end = self.starting_time + self.shape[0] / self.rate
            return (self.starting_time, end)
        return None

    @property
    def is_regularly_sampled(self) -> bool:
        return self.rate is not None

    def time_to_slice(self, t_start: float, t_stop: float) -> slice:
        if self.rate and self.starting_time is not None:
            i0 = int((t_start - self.starting_time) * self.rate)
            i1 = int((t_stop - self.starting_time) * self.rate)
            return slice(max(i0, 0), min(i1, self.shape[0]))
        raise ValueError("Irregular timestamps: use np.searchsorted on the timestamps dataset")


@dataclass(frozen=True, slots=True)
class TimeIntervalsRecord:
    """Metadata for one TimeIntervals table discovered inside an NWB file."""

    path: str
    name: str
    neurodata_type: str
    description: str
    n_rows: int
    column_names: tuple[str, ...]
    time_range: tuple[float, float] | None


# ---------------------------------------------------------------------------
# DataLoader protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DataLoader(Protocol):
    """Backend-agnostic data access."""

    @property
    def backend(self) -> str: ...

    @property
    def catalog(self) -> DataCatalog: ...

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float,
        t1: float,
        color_variable: str | None = None,
    ) -> PlotData | None: ...

    def feature_dims(self, feature: str) -> dict[str, list[str]]: ...

    # Convenience properties — delegate to catalog
    @property
    def features(self) -> list[str]: ...

    @property
    def dims(self) -> dict[str, np.ndarray]: ...

    @property
    def changepoint_names(self) -> list[str]: ...

    def get_type_vars(self) -> dict: ...


# ---------------------------------------------------------------------------
# Shared mixin: catalog-backed convenience properties
# ---------------------------------------------------------------------------


class _CatalogMixin:
    """Delegates metadata properties to a DataCatalog."""

    _catalog: DataCatalog

    @property
    def catalog(self) -> DataCatalog:
        return self._catalog

    @property
    def features(self) -> list[str]:
        return self._catalog.features

    @property
    def dims(self) -> dict[str, np.ndarray]:
        return {n: np.array(s.values) for n, s in self._catalog.combos.items() if n != "features"}

    @property
    def changepoint_names(self) -> list[str]:
        return self._catalog.changepoints

    def get_type_vars(self) -> dict:
        return self._catalog.to_type_vars_dict()


# ---------------------------------------------------------------------------
# XarrayLoader
# ---------------------------------------------------------------------------


class XarrayLoader(_CatalogMixin):
    """Feature access from an ``xr.Dataset``.  Uses ``sel_valid`` for selection."""

    def __init__(self, ds: xr.Dataset, catalog: DataCatalog | None = None) -> None:
        self._ds = ds
        if catalog is None:
            catalog = _auto_catalog_xarray(ds)
        self._catalog = catalog

    @property
    def backend(self) -> str:
        return "xarray"

    def update_ds(self, ds: xr.Dataset) -> None:
        """Swap the backing dataset (called on trial change)."""
        self._ds = ds

    def feature_dims(self, feature: str) -> dict[str, list[str]]:
        if self._ds is None or feature not in self._ds.data_vars:
            return {}
        var = self._ds[feature]
        result: dict[str, list[str]] = {}
        for d in var.dims:
            if "time" in d.lower():
                continue
            if d in var.coords:
                result[d] = [str(v) for v in var.coords[d].values]
            else:
                result[d] = [str(i) for i in range(self._ds.sizes[d])]
        return result

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float | None = None,
        t1: float | None = None,
        color_variable: str | None = None,
    ) -> PlotData | None:
        import ethograph as eto

        ds = self._ds
        if ds is None or feature not in ds.data_vars:
            return None

        var = ds[feature]
        time_coord = eto.get_time_coord(var)
        if time_coord is None:
            return None

        if t0 is not None and t1 is not None:
            ds = ds.sel({time_coord.name: slice(t0, t1)})
            var = ds[feature]

        time = eto.get_time_coord(var).values
        # Translate "individuals" → "individual" for movement v0.17+ datasets
        _sel = dict(selections)
        if "individuals" in _sel and "individual" in var.dims and "individuals" not in var.dims:
            _sel["individual"] = _sel.pop("individuals")
        data, filt_kwargs = eto.sel_valid(var, _sel)
        var_sel = var.sel(**filt_kwargs)

        dim_labels = None
        if data.ndim == 2:
            non_time_dim = next((d for d in var_sel.dims if "time" not in d.lower()), None)
            if non_time_dim and non_time_dim in var_sel.coords:
                dim_labels = [str(c) for c in var_sel.coords[non_time_dim].values]
            else:
                dim_labels = [str(i) for i in range(data.shape[1])]

        color_data = None
        if data.ndim == 1 and color_variable and color_variable in ds.data_vars:
            color_kwargs = {k: v for k, v in selections.items() if k != "RGB"}
            color_data, _ = eto.sel_valid(ds[color_variable], color_kwargs)

        changepoints = None
        if data.ndim == 1:
            cp_ds = ds.filter_by_attrs(type="changepoints")
            cp_dict: dict[str, np.ndarray] = {}
            for cp_name in cp_ds.data_vars:
                cp_var = cp_ds[cp_name]
                cp_data, _ = eto.sel_valid(cp_var, selections)
                if cp_var.attrs.get("target_feature") == feature and not np.isnan(cp_data).all():
                    cp_dict[cp_name] = cp_data
            if cp_dict:
                changepoints = cp_dict

        ylabel = var.attrs.get("ylabel", feature)
        title_parts = [f"Trial: {ds.attrs.get('trial')}"]
        title_parts.extend(f"{k}={v}" for k, v in filt_kwargs.items())
        title = ", ".join(title_parts)

        return PlotData(
            time=time,
            data=data,
            dim_labels=dim_labels,
            title=title,
            ylabel=ylabel,
            color_data=color_data,
            changepoints=changepoints,
        )

    def time_range(self, feature: str | None = None) -> tuple[float, float]:
        import ethograph as eto

        if self._ds is None:
            return (0.0, 0.0)
        if feature and feature in self._ds.data_vars:
            tc = eto.get_time_coord(self._ds[feature])
        else:
            tc = None
            for var in self._ds.data_vars.values():
                tc = eto.get_time_coord(var)
                if tc is not None:
                    break
        if tc is None:
            return (0.0, 0.0)
        vals = tc.values
        return (float(vals[0]), float(vals[-1]))

    def get_cp_times(self, feature: str | None = None, **kwargs) -> np.ndarray:
        import ethograph as eto
        from ethograph.features.changepoints import extract_cp_times

        if self._ds is None:
            return np.array([], dtype=np.float64)
        tc = eto.get_time_coord(next(iter(self._ds.data_vars.values()), None))
        if tc is None:
            return np.array([], dtype=np.float64)
        return extract_cp_times(self._ds, tc.values)


# ---------------------------------------------------------------------------
# PynappleLoader
# ---------------------------------------------------------------------------


class PynappleLoader(_CatalogMixin):
    """Stateless feature access backed by raw pynapple objects.

    No trial state — callers always pass ``t0, t1`` (absolute session
    times).  The loader ``restrict()``-s to that range, subtracts ``t0``
    so returned time starts near 0, and returns numpy in a PlotData.
    """

    def __init__(
        self,
        data: dict,
        catalog: DataCatalog | None = None,
    ) -> None:
        import pynapple as nap

        self._data = data
        if catalog is None:
            catalog = catalog_from_pynapple(data)
        self._catalog = catalog

        self._feature_objs: dict = {
            k: v for k, v in data.items() if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))
        }
        self._dim_map = _compute_shared_column_dims(self._feature_objs)

    @property
    def backend(self) -> str:
        return "pynapple"

    def feature_dims(self, feature: str) -> dict[str, list[str]]:
        import pynapple as nap

        if feature == "pose_estimation":
            result: dict[str, list[str]] = {}
            kp_spec = self._catalog.combos.get("keypoint") if self._catalog else None
            if kp_spec and kp_spec.values:
                # Spatial columns first — _build_axis_items expands the first
                # dim into axis items (pose_estimation · x, etc.)
                first_kp = self._feature_objs.get(kp_spec.values[0])
                if isinstance(first_kp, nap.TsdFrame):
                    result["space"] = [str(c) for c in first_kp.columns]
                # Keypoints second — picked up by _populate_keypoint_combo
                result["keypoint"] = list(kp_spec.values)
            return result

        if feature not in self._feature_objs:
            return {}
        obj = self._feature_objs[feature]
        result: dict[str, list[str]] = {}
        dim_name = self._dim_map.get(feature)
        if isinstance(obj, nap.TsdFrame) and dim_name:
            result[dim_name] = [str(c) for c in obj.columns]
        return result

    def _restrict(self, obj: Any, t0: float, t1: float):
        """Restrict to ``[t0, t1]`` (absolute session times). Returns ``(obj, t0)``."""
        import pynapple as nap

        obj = obj.restrict(nap.IntervalSet(start=t0, end=t1))
        return obj, t0

    def _select_all_keypoints(
        self,
        selections: dict[str, str],
        t0: float,
        t1: float,
    ) -> PlotData | None:
        """Stack all keypoint into one (T, N) array, optionally slicing by space/column."""
        import pynapple as nap

        kp_spec = self._catalog.combos.get("keypoint") if self._catalog else None
        if not kp_spec or not kp_spec.values:
            return None

        arrays: list[np.ndarray] = []
        labels: list[str] = []
        time = None

        for kp_name in kp_spec.values:
            obj = self._feature_objs.get(kp_name)
            if obj is None:
                continue
            obj, offset = self._restrict(obj, t0, t1)
            if len(obj) == 0:
                continue
            if time is None:
                time = obj.t - offset

            if isinstance(obj, nap.TsdFrame):
                cols = set(str(c) for c in obj.columns)
                selected_col = None
                for val in selections.values():
                    if val in cols:
                        selected_col = val
                        break
                if selected_col:
                    arrays.append(obj[selected_col].values)
                    labels.append(kp_name)
                else:
                    for c in obj.columns:
                        arrays.append(obj[c].values)
                        labels.append(f"{kp_name}_{c}")
            elif isinstance(obj, nap.Tsd):
                arrays.append(obj.values)
                labels.append(kp_name)
            else:
                continue

        if not arrays or time is None:
            return None

        stacked = np.column_stack(arrays) if len(arrays) > 1 else arrays[0]
        dim_labels = labels if stacked.ndim == 2 else None

        title_parts = [f"{k}={v}" for k, v in selections.items() if k != "individuals"]

        return PlotData(
            time=time,
            data=stacked,
            dim_labels=dim_labels,
            title=", ".join(title_parts),
            ylabel="pose_estimation",
        )

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float,
        t1: float,
        color_variable: str | None = None,
    ) -> PlotData | None:
        import pynapple as nap

        actual_key = feature
        if feature == "pose_estimation":
            kp = selections.get("keypoint")
            if kp and kp in self._feature_objs:
                actual_key = kp
            else:
                return self._select_all_keypoints(selections, t0, t1)

        if actual_key not in self._feature_objs:
            return None

        obj = self._feature_objs[actual_key]
        obj, offset = self._restrict(obj, t0, t1)

        if len(obj) == 0:
            return None

        time = obj.t - offset

        # --- extract numpy based on type + selections ---
        if isinstance(obj, nap.Tsd):
            data = obj.values
            dim_labels = None

        elif isinstance(obj, nap.TsdFrame):
            cols = set(str(c) for c in obj.columns)
            selected_col = None
            col_dim = self._dim_map.get(actual_key)
            if col_dim and col_dim in selections:
                selected_col = selections[col_dim]
            if selected_col is None:
                for val in selections.values():
                    if val in cols:
                        selected_col = val
                        break

            if selected_col and selected_col in obj.columns:
                data = obj[selected_col].values
                dim_labels = None
            else:
                data = obj.values
                dim_labels = list(obj.columns)

        elif isinstance(obj, nap.TsdTensor):
            data = obj.values
            if data.ndim > 2:
                data = data.reshape(len(time), -1)
            dim_labels = [str(i) for i in range(data.shape[1])] if data.ndim == 2 else None
        else:
            return None

        # Color data
        color_data = None
        if data.ndim == 1 and color_variable and color_variable in self._feature_objs:
            color_obj = self._feature_objs[color_variable]
            color_obj, _ = self._restrict(color_obj, t0, t1)
            if isinstance(color_obj, nap.TsdFrame) and len(color_obj) > 0:
                color_data = color_obj.values

        # Changepoints (1-D features only)
        changepoints = None
        if data.ndim == 1:
            cp_dict: dict[str, np.ndarray] = {}
            for key, raw_obj in self._data.items():
                if not isinstance(raw_obj, nap.TsGroup):
                    continue
                if not (hasattr(raw_obj, "metadata") and raw_obj.metadata is not None):
                    continue
                meta = raw_obj.metadata
                if "type" not in meta.columns:
                    continue
                target = meta["target_feature"].iloc[0] if "target_feature" in meta.columns else None
                if target != feature:
                    continue
                cp_units = meta.index[meta["type"] == "changepoints"]
                for uid in cp_units:
                    ts_obj = raw_obj[uid]
                    ts_obj, _ = self._restrict(ts_obj, t0, t1)
                    if len(ts_obj) == 0:
                        continue
                    if isinstance(ts_obj, nap.Tsd):
                        mask = ts_obj.values.astype(bool)
                        cp_times = ts_obj.t[mask] - t0
                    else:
                        cp_times = ts_obj.t - t0
                    if len(cp_times) == 0:
                        continue
                    binary = np.zeros(len(time), dtype=np.int8)
                    idxs = np.searchsorted(time, cp_times)
                    idxs = np.clip(idxs, 0, len(time) - 1)
                    binary[idxs] = 1
                    cp_dict[key] = binary
            if cp_dict:
                changepoints = cp_dict

        title_parts = [f"{k}={v}" for k, v in selections.items() if k != "individuals"]
        title = ", ".join(title_parts)

        return PlotData(
            time=time,
            data=data,
            dim_labels=dim_labels,
            title=title,
            ylabel=feature,
            color_data=color_data,
            changepoints=changepoints,
        )

    def get_cp_times(
        self,
        feature: str | None = None,
        t0: float = 0.0,
        t1: float = 0.0,
    ) -> np.ndarray:
        import pynapple as nap

        all_times: list[np.ndarray] = []
        for key, obj in self._data.items():
            if not isinstance(obj, nap.TsGroup):
                continue
            if not (hasattr(obj, "metadata") and obj.metadata is not None):
                continue
            meta = obj.metadata
            if "type" not in meta.columns:
                continue
            if feature and "target_feature" in meta.columns:
                if meta["target_feature"].iloc[0] != feature:
                    continue
            cp_units = meta.index[meta["type"] == "changepoints"]
            for uid in cp_units:
                ts_obj = obj[uid]
                ts_obj, _ = self._restrict(ts_obj, t0, t1)
                if len(ts_obj) == 0:
                    continue
                if isinstance(ts_obj, nap.Tsd):
                    mask = ts_obj.values.astype(bool)
                    all_times.append(ts_obj.t[mask] - t0)
                else:
                    all_times.append(ts_obj.t - t0)
        if not all_times:
            return np.array([], dtype=np.float64)
        return np.unique(np.concatenate(all_times)).astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_shared_column_dims(
    feature_objs: dict,
) -> dict[str, str]:
    """Map each TsdFrame to a shared column dimension name.

    TsdFrame objects with identical columns share one xarray dimension
    → one combo in the GUI.
    """
    import pynapple as nap

    groups: dict[tuple, list[str]] = {}
    for name, obj in feature_objs.items():
        if isinstance(obj, nap.TsdFrame):
            key = tuple(obj.columns)
            groups.setdefault(key, []).append(name)

    dim_map: dict[str, str] = {}
    used: set[str] = set()

    for cols, names in groups.items():
        if len(names) == 1:
            dim_name = f"{names[0]}_columns"
        else:
            dim_name = "columns"
            suffix = 2
            while dim_name in used:
                dim_name = f"columns_{suffix}"
                suffix += 1
        used.add(dim_name)
        for name in names:
            dim_map[name] = dim_name

    return dim_map


# Dimensions that are internal to color variables — never show as user combos
_HIDDEN_DIMS = frozenset({"RGB", "RGBA"})

# Variables that are never user-selectable features
_EXCLUDED_VARS = frozenset({"onset_s", "offset_s", "labels", "individual", "boundary_events"})


def _feature_vars(ds: xr.Dataset) -> list[str]:
    """All data_vars with a time dimension, minus changepoints and internal vars."""
    features = []
    for name, var in ds.data_vars.items():
        if name in _EXCLUDED_VARS:
            continue
        if var.attrs.get("type") == "changepoints":
            continue
        if any("time" in str(d).lower() for d in var.dims):
            features.append(name)
    return features


def _auto_catalog_xarray(ds: xr.Dataset) -> DataCatalog:
    """Quick catalog from a Dataset when no TrialTree is available."""
    from ethograph.io.validation import find_temporal_dims

    combos: dict[str, ComboSpec] = {}

    # Support both wizard format ("individuals") and movement v0.17+ format ("individual")
    _ind_dim = next((n for n in ("individuals", "individual") if n in ds.coords), None)
    if _ind_dim is not None:
        vals = tuple(ds.coords[_ind_dim].values.astype(str))
        combos["individuals"] = ComboSpec("individuals", vals)

    features_list = _feature_vars(ds)
    changepoints_list = list(ds.filter_by_attrs(type="changepoints").data_vars)

    if features_list:
        combos["features"] = ComboSpec("features", tuple(features_list))

    for name in find_temporal_dims(ds):
        if name in combos or name.upper() in _HIDDEN_DIMS:
            continue
        # "individual" already normalized to "individuals" combo above
        if name == "individual" and _ind_dim == "individual":
            continue
        if name in ds.coords:
            coord = ds.coords[name]
            if coord.dtype.kind in ("U", "S", "O"):
                vals = tuple(coord.values.astype(str))
            else:
                vals = tuple(str(v) for v in coord.values)
        else:
            vals = tuple(str(i) for i in range(ds.sizes[name]))
        combos[name] = ComboSpec(name, vals)

    return DataCatalog(
        combos=combos,
        features=features_list,
        changepoints=changepoints_list,
    )


# ---------------------------------------------------------------------------
# Catalog builders
# ---------------------------------------------------------------------------


def catalog_from_xarray(ds: xr.Dataset, dt: TrialTree, nwb_alignment=None) -> DataCatalog:
    """Build a DataCatalog from an xarray Dataset + TrialTree."""
    from ethograph.io.validation import (
        _possible_trial_conditions,
        find_temporal_dims,
    )

    combos: dict[str, ComboSpec] = {}

    # Support both wizard format ("individuals") and movement v0.17+ format ("individual")
    _ind_dim = next((n for n in ("individuals", "individual") if n in ds.coords), None)
    if _ind_dim is not None:
        vals = tuple(ds.coords[_ind_dim].values.astype(str))
        combos["individuals"] = ComboSpec("individuals", vals)

    features_list = _feature_vars(ds)
    changepoints_list = list(ds.filter_by_attrs(type="changepoints").data_vars)

    if features_list:
        combos["features"] = ComboSpec("features", tuple(features_list))

    extra_dims = find_temporal_dims(ds)
    for name in extra_dims:
        if name in combos or name.upper() in _HIDDEN_DIMS:
            continue
        # "individual" already normalized to "individuals" combo above
        if name == "individual" and _ind_dim == "individual":
            continue
        if name in ds.coords:
            coord = ds.coords[name]
            if coord.dtype.kind in ("U", "S", "O"):
                vals = tuple(coord.values.astype(str))
            else:
                vals = tuple(str(v) for v in coord.values)
        else:
            vals = tuple(str(i) for i in range(ds.sizes[name]))
        combos[name] = ComboSpec(name, vals)

    sio = nwb_alignment or getattr(dt, "nwb_alignment", None)
    cameras = list(sio.cameras) if sio and sio.cameras else []
    mics = list(sio.mics) if sio and sio.mics else []
    trial_conditions = _possible_trial_conditions(ds, dt)

    return DataCatalog(
        combos=combos,
        features=features_list,
        changepoints=changepoints_list,
        cameras=cameras,
        mics=mics,
        trial_conditions=trial_conditions,
    )


def _find_pose_series_names(nwb_path: str | Path) -> list[str]:
    """Scan an NWB file with h5py and return PoseEstimationSeries leaf names."""
    import h5py

    hits: list[str] = []

    def _visit(name: str, obj: Any) -> None:
        ndt = obj.attrs.get("neurodata_type", "")
        if isinstance(ndt, bytes):
            ndt = ndt.decode()
        if ndt == "PoseEstimationSeries":
            hits.append(name.rsplit("/", 1)[-1])

    with h5py.File(nwb_path, "r") as f:
        f.visititems(_visit)
    return hits


def _discover_pose_keypoints(source_path: str | Path) -> set[str]:
    """Find PoseEstimationSeries names from NWB files at *source_path*.

    *source_path* can be a ``.nwb`` file, a ``.npz`` file (scans sibling
    NWB files), or a folder (scans contained NWB files).
    """
    p = Path(source_path)
    nwb_files: list[Path] = []
    if p.suffix == ".nwb" and p.is_file():
        nwb_files.append(p)
    elif p.is_dir():
        nwb_files.extend(p.glob("*.nwb"))
    elif p.is_file():
        nwb_files.extend(p.parent.glob("*.nwb"))

    keypoint_names: set[str] = set()
    for nwb in nwb_files:
        try:
            keypoint_names.update(_find_pose_series_names(nwb))
        except Exception:
            continue
    return keypoint_names


def catalog_from_pynapple(
    data: dict,
    *,
    source_path: str | Path | None = None,
) -> DataCatalog:
    """Build a DataCatalog from pynapple objects.

    Parameters
    ----------
    source_path
        Path to the original data source (``.nwb``, ``.npz``, or folder).
        When provided, NWB files are scanned for ``PoseEstimationSeries``
        to create a ``keypoint`` combo from matching pynapple keys.
    """
    import pynapple as nap

    pose_names: set[str] = set()
    if source_path is not None:
        pose_names = _discover_pose_keypoints(source_path)

    feature_objs = {k: v for k, v in data.items() if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))}
    dim_map = _compute_shared_column_dims(feature_objs)

    combos: dict[str, ComboSpec] = {}
    combos["individuals"] = ComboSpec("individuals", ("individual_0",))

    features: list[str] = []
    changepoints: list[str] = []
    keypoint_names: list[str] = []

    for key, obj in data.items():
        if isinstance(obj, nap.IntervalSet):
            continue

        if isinstance(obj, nap.TsGroup):
            if hasattr(obj, "metadata") and obj.metadata is not None:
                meta = obj.metadata
                if "type" in meta.columns and "changepoints" in meta["type"].unique():
                    changepoints.append(key)
            continue

        if isinstance(obj, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
            if key in pose_names:
                keypoint_names.append(key)
            else:
                features.append(key)

                if isinstance(obj, nap.TsdFrame):
                    dim_name = dim_map.get(key, f"{key}_columns")
                    if dim_name not in combos:
                        combos[dim_name] = ComboSpec(dim_name, tuple(str(c) for c in obj.columns))

    if keypoint_names:
        combos["keypoint"] = ComboSpec("keypoint", tuple(sorted(keypoint_names)))
        features.insert(0, "pose_estimation")
        # Detect column combo for keypoint TsdFrames
        first_kp = feature_objs.get(keypoint_names[0])
        if isinstance(first_kp, nap.TsdFrame):
            cols = tuple(str(c) for c in first_kp.columns)
            if set(cols) <= {"x", "y", "z"}:
                combos["space"] = ComboSpec("space", cols)
            else:
                combos["columns"] = ComboSpec("columns", cols)

    if features:
        combos["features"] = ComboSpec("features", tuple(features))

    return DataCatalog(
        combos=combos,
        features=features,
        changepoints=changepoints,
        trial_conditions=[],
    )
