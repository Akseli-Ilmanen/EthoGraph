"""Unified data catalog and loader for xarray, pynapple, and NWB backends.

Replaces the old type_vars_dict pattern with:
- DataCatalog: what dimensions/features are available (builds combo boxes)
- DataLoader: how to load data (select by feature + combo dims + time window → PlotData)

Three backends, same interface. Differs in how combo dims are discovered,
how selection works (sel_valid principle: overspecified combos are OK), and
how time slicing works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import pynapple as nap
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
    boundary_events: np.ndarray | None = None


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
    colors: list[str] = field(default_factory=list)
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
        if self.colors:
            tvd["colors"] = self.colors
        if self.changepoints:
            tvd["changepoints"] = self.changepoints
        if self.cameras:
            tvd["cameras"] = np.array(self.cameras)
        if self.mics:
            tvd["mics"] = np.array(self.mics)
        tvd["trial_conditions"] = self.trial_conditions
        return tvd


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
        t0: float | None = None,
        t1: float | None = None,
        color_variable: str | None = None,
    ) -> PlotData | None: ...

    def feature_dims(self, feature: str) -> dict[str, list[str]]: ...

    def time_range(self, feature: str | None = None) -> tuple[float, float]: ...

    def set_trial(self, trial_idx: int) -> None: ...

    def get_cp_times(self, feature: str | None = None) -> np.ndarray: ...

    # Convenience properties — delegate to catalog
    @property
    def features(self) -> list[str]: ...

    @property
    def dims(self) -> dict[str, np.ndarray]: ...

    @property
    def colors(self) -> list[str]: ...

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
        return {
            n: np.array(s.values)
            for n, s in self._catalog.combos.items()
            if n != "features"
        }

    @property
    def colors(self) -> list[str]:
        return self._catalog.colors

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
        data, filt_kwargs = eto.sel_valid(var, selections)
        var_sel = var.sel(**filt_kwargs)

        dim_labels = None
        if data.ndim == 2:
            non_time_dim = next(
                (d for d in var_sel.dims if "time" not in d.lower()), None
            )
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
                if (
                    cp_var.attrs.get("target_feature") == feature
                    and not np.isnan(cp_data).all()
                ):
                    cp_dict[cp_name] = cp_data
            if cp_dict:
                changepoints = cp_dict

        boundary_events = None
        if "boundary_events" in ds.data_vars:
            raw = ds["boundary_events"].values
            valid = raw[~np.isnan(raw)].astype(int)
            valid = valid[(valid >= 0) & (valid < len(time))]
            if len(valid) > 0:
                boundary_events = time[valid]

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
            boundary_events=boundary_events,
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

    def get_cp_times(self, feature: str | None = None) -> np.ndarray:
        import ethograph as eto
        from ethograph.features.changepoints import extract_cp_times

        if self._ds is None:
            return np.array([], dtype=np.float64)
        tc = eto.get_time_coord(next(iter(self._ds.data_vars.values()), None))
        if tc is None:
            return np.array([], dtype=np.float64)
        return extract_cp_times(self._ds, tc.values)

    def set_trial(self, trial_idx: int) -> None:
        pass  # Trial switching handled externally via update_ds


# ---------------------------------------------------------------------------
# PynappleLoader
# ---------------------------------------------------------------------------


class PynappleLoader(_CatalogMixin):
    """Lazy feature access backed by raw pynapple objects.

    Data is never copied into xarray. On each ``select()`` call the
    pynapple object is ``restrict()``-ed to the current trial, time-sliced,
    column-selected, and returned as numpy in a PlotData.
    """

    def __init__(
        self,
        data: dict,
        trials_ep: nap.IntervalSet | None = None,
        catalog: DataCatalog | None = None,
    ) -> None:
        import pynapple as nap

        self._data = data
        self._trials_ep = trials_ep
        if catalog is None:
            catalog = catalog_from_pynapple(data, trials_ep)
        self._catalog = catalog
        self._current_trial_idx = 0

        self._feature_objs: dict = {
            k: v
            for k, v in data.items()
            if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))
        }
        self._dim_map = _compute_shared_column_dims(self._feature_objs)

    @property
    def backend(self) -> str:
        return "pynapple"

    @property
    def n_trials(self) -> int:
        if self._trials_ep is None:
            return 1
        return len(self._trials_ep)

    @property
    def trials(self) -> list[int]:
        return list(range(1, self.n_trials + 1))

    @property
    def _trial_offset(self) -> float:
        if self._trials_ep is None:
            times = [obj.t[0] for obj in self._feature_objs.values() if len(obj) > 0]
            return min(times) if times else 0.0
        return float(self._trials_ep.start[self._current_trial_idx])

    @property
    def trial_bounds(self) -> tuple[float, float]:
        if self._trials_ep is None:
            all_t: list[float] = []
            for obj in self._feature_objs.values():
                if len(obj) > 0:
                    all_t.extend([obj.t[0], obj.t[-1]])
            return (0.0, max(all_t) - min(all_t)) if all_t else (0.0, 0.0)
        start = float(self._trials_ep.start[self._current_trial_idx])
        end = float(self._trials_ep.end[self._current_trial_idx])
        return (0.0, end - start)

    def feature_dims(self, feature: str) -> dict[str, list[str]]:
        import pynapple as nap

        if feature not in self._feature_objs:
            return {}
        obj = self._feature_objs[feature]
        result: dict[str, list[str]] = {}
        dim_name = self._dim_map.get(feature)
        if isinstance(obj, nap.TsdFrame) and dim_name:
            result[dim_name] = [str(c) for c in obj.columns]
        return result

    def set_trial(self, trial_idx: int) -> None:
        self._current_trial_idx = trial_idx

    def _restrict(self, obj: Any, t0: float | None = None, t1: float | None = None):
        """Restrict to current trial + optional time window. Returns (obj, offset)."""
        import pynapple as nap

        offset = self._trial_offset

        if self._trials_ep is not None:
            trial_start = float(self._trials_ep.start[self._current_trial_idx])
            trial_end = float(self._trials_ep.end[self._current_trial_idx])
            obj = obj.restrict(nap.IntervalSet(start=trial_start, end=trial_end))

        if t0 is not None and t1 is not None:
            abs_t0 = t0 + offset
            abs_t1 = t1 + offset
            obj = obj.restrict(nap.IntervalSet(start=abs_t0, end=abs_t1))

        return obj, offset

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float | None = None,
        t1: float | None = None,
        color_variable: str | None = None,
    ) -> PlotData | None:
        import pynapple as nap

        if feature not in self._feature_objs:
            return None

        obj = self._feature_objs[feature]
        obj, offset = self._restrict(obj, t0, t1)

        if len(obj) == 0:
            return None

        time = obj.t - offset

        # --- extract numpy based on type + selections ---
        if isinstance(obj, nap.Tsd):
            data = obj.values
            dim_labels = None

        elif isinstance(obj, nap.TsdFrame):
            col_dim = self._dim_map.get(feature)
            if col_dim and col_dim in selections:
                selected_col = selections[col_dim]
                if selected_col in obj.columns:
                    data = obj[selected_col].values
                    dim_labels = None
                else:
                    data = obj.values
                    dim_labels = list(obj.columns)
            else:
                data = obj.values
                dim_labels = list(obj.columns)

        elif isinstance(obj, nap.TsdTensor):
            data = obj.values
            if data.ndim > 2:
                data = data.reshape(len(time), -1)
            dim_labels = (
                [str(i) for i in range(data.shape[1])] if data.ndim == 2 else None
            )
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
                target = (
                    meta["target_feature"].iloc[0]
                    if "target_feature" in meta.columns
                    else None
                )
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
                        cp_times = ts_obj.t[mask] - self._trial_offset
                    else:
                        cp_times = ts_obj.t - self._trial_offset
                    if len(cp_times) == 0:
                        continue
                    binary = np.zeros(len(time), dtype=np.int8)
                    idxs = np.searchsorted(time, cp_times)
                    idxs = np.clip(idxs, 0, len(time) - 1)
                    binary[idxs] = 1
                    cp_dict[key] = binary
            if cp_dict:
                changepoints = cp_dict

        trial_num = self._current_trial_idx + 1
        title_parts = [f"Trial: {trial_num}"]
        title_parts.extend(
            f"{k}={v}" for k, v in selections.items() if k != "individuals"
        )
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

    def get_cp_times(self, feature: str | None = None) -> np.ndarray:
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
                ts_obj, _ = self._restrict(ts_obj)
                if len(ts_obj) == 0:
                    continue
                if isinstance(ts_obj, nap.Tsd):
                    mask = ts_obj.values.astype(bool)
                    all_times.append(ts_obj.t[mask] - self._trial_offset)
                else:
                    all_times.append(ts_obj.t - self._trial_offset)
        if not all_times:
            return np.array([], dtype=np.float64)
        return np.unique(np.concatenate(all_times)).astype(np.float64)

    def time_range(self, feature: str | None = None) -> tuple[float, float]:
        return self.trial_bounds


# ---------------------------------------------------------------------------
# NWBLoader
# ---------------------------------------------------------------------------


class NWBLoader(_CatalogMixin):
    """Feature access backed by an NWB file via nwb_catalog + ComboCatalog.

    Supports local files and remote URLs (via remfile).
    Time slicing via ``TimeSeriesRecord.time_to_slice`` (rate-based) or
    ``np.searchsorted`` on timestamps.
    """

    def __init__(
        self,
        source: str,
        catalog: DataCatalog,
        combo_catalog: Any | None = None,
    ) -> None:
        self._source = source
        self._catalog = catalog
        self._current_trial_idx = 0
        self._trial_intervals: list[tuple[float, float]] = []

        if combo_catalog is None:
            from ethograph.io.nwb_backend import build_combos, catalog_nwb

            nwb_cat = catalog_nwb(source)
            self._combo_catalog = build_combos(nwb_cat)
        else:
            self._combo_catalog = combo_catalog

    @property
    def backend(self) -> str:
        return "nwb"

    @property
    def _trial_offset(self) -> float:
        if not self._trial_intervals:
            return 0.0
        return self._trial_intervals[self._current_trial_idx][0]

    @property
    def trial_bounds(self) -> tuple[float, float]:
        if not self._trial_intervals:
            return (0.0, 0.0)
        start, end = self._trial_intervals[self._current_trial_idx]
        return (0.0, end - start)

    def set_trial(self, trial_idx: int) -> None:
        self._current_trial_idx = trial_idx

    def set_trial_intervals(self, intervals: list[tuple[float, float]]) -> None:
        self._trial_intervals = intervals

    def feature_dims(self, feature: str) -> dict[str, list[str]]:
        entries = self._combo_catalog.filter(feature=feature)
        if not entries:
            entries = self._combo_catalog.filter(keypoint=feature)
        result: dict[str, list[str]] = {}
        for entry in entries:
            if entry.columns:
                result["space"] = list(entry.columns)
                break
        return result

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float | None = None,
        t1: float | None = None,
        color_variable: str | None = None,
    ) -> PlotData | None:
        from ethograph.io.nwb_backend import open_nwb

        offset = self._trial_offset

        if t0 is not None and t1 is not None:
            abs_t0 = t0 + offset
            abs_t1 = t1 + offset
        elif self._trial_intervals:
            abs_t0, abs_t1 = self._trial_intervals[self._current_trial_idx]
        else:
            abs_t0, abs_t1 = 0.0, 1e9

        # Build combo selection — map feature name into the right tag
        combo_sel = {}
        for k, v in selections.items():
            if k in ("module", "group", "keypoint", "feature", "space"):
                combo_sel[k] = v

        # Add the feature itself to selection if it matches an entry
        display_names = {e.display_name for e in self._combo_catalog.features}
        keypoint_names = {
            e.tags.get("keypoint", "") for e in self._combo_catalog.features
        }
        if feature in display_names:
            combo_sel["feature"] = feature
        elif feature in keypoint_names:
            combo_sel["keypoint"] = feature

        with open_nwb(self._source) as h5:
            stacked = self._combo_catalog.load_stacked(
                h5, abs_t0, abs_t1, **combo_sel
            )

        if stacked.data.size == 0:
            return None

        time = stacked.timestamps - offset
        data = stacked.data
        dim_labels = (
            list(stacked.labels)
            if data.ndim == 2 and data.shape[1] > 1
            else None
        )

        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]
            dim_labels = None

        trial_num = self._current_trial_idx + 1
        title_parts = [f"Trial: {trial_num}"]
        title_parts.extend(f"{k}={v}" for k, v in selections.items())
        title = ", ".join(title_parts)

        return PlotData(
            time=time,
            data=data,
            dim_labels=dim_labels,
            title=title,
            ylabel=feature,
        )

    def time_range(self, feature: str | None = None) -> tuple[float, float]:
        return self.trial_bounds

    def get_cp_times(self, feature: str | None = None) -> np.ndarray:
        return np.array([], dtype=np.float64)


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


def _auto_catalog_xarray(ds: xr.Dataset) -> DataCatalog:
    """Quick catalog from a Dataset when no TrialTree is available."""
    from ethograph.io.validation import find_temporal_dims

    combos: dict[str, ComboSpec] = {}

    if "individuals" in ds.coords:
        vals = tuple(ds.coords["individuals"].values.astype(str))
        combos["individuals"] = ComboSpec("individuals", vals)

    features_list = list(ds.filter_by_attrs(type="features").data_vars)
    colors_list = list(ds.filter_by_attrs(type="colors").data_vars)
    changepoints_list = list(ds.filter_by_attrs(type="changepoints").data_vars)

    if features_list:
        combos["features"] = ComboSpec("features", tuple(features_list))

    for name in find_temporal_dims(ds):
        if name in combos:
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
        colors=colors_list,
        changepoints=changepoints_list,
    )


# ---------------------------------------------------------------------------
# Catalog builders
# ---------------------------------------------------------------------------


def catalog_from_xarray(ds: xr.Dataset, dt: TrialTree) -> DataCatalog:
    """Build a DataCatalog from an xarray Dataset + TrialTree."""
    from ethograph.io.validation import (
        _possible_trial_conditions,
        find_temporal_dims,
    )

    combos: dict[str, ComboSpec] = {}

    if "individuals" in ds.coords:
        vals = tuple(ds.coords["individuals"].values.astype(str))
        combos["individuals"] = ComboSpec("individuals", vals)

    features_list = list(ds.filter_by_attrs(type="features").data_vars)
    colors_list = list(ds.filter_by_attrs(type="colors").data_vars)
    changepoints_list = list(ds.filter_by_attrs(type="changepoints").data_vars)

    if features_list:
        combos["features"] = ComboSpec("features", tuple(features_list))

    extra_dims = find_temporal_dims(ds)
    for name in extra_dims:
        if name in combos:
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

    cameras = list(dt.cameras) if dt.cameras else []
    mics = list(dt.mics) if dt.mics else []
    trial_conditions = _possible_trial_conditions(ds, dt)

    return DataCatalog(
        combos=combos,
        features=features_list,
        colors=colors_list,
        changepoints=changepoints_list,
        cameras=cameras,
        mics=mics,
        trial_conditions=trial_conditions,
    )


def catalog_from_pynapple(
    data: dict,
    trials_ep: nap.IntervalSet | None = None,
) -> DataCatalog:
    """Build a DataCatalog from pynapple objects."""
    import pynapple as nap

    feature_objs = {
        k: v
        for k, v in data.items()
        if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))
    }
    dim_map = _compute_shared_column_dims(feature_objs)

    combos: dict[str, ComboSpec] = {}
    combos["individuals"] = ComboSpec("individuals", ("individual_0",))

    features: list[str] = []
    colors: list[str] = []
    changepoints: list[str] = []

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
            features.append(key)

            if isinstance(obj, nap.TsdFrame):
                cols_lower = [c.lower() for c in obj.columns]
                if "rgb" in key.lower() or set(cols_lower) == {"r", "g", "b"}:
                    colors.append(key)

                dim_name = dim_map.get(key, f"{key}_columns")
                if dim_name not in combos:
                    combos[dim_name] = ComboSpec(
                        dim_name, tuple(str(c) for c in obj.columns)
                    )

    if features:
        combos["features"] = ComboSpec("features", tuple(features))

    return DataCatalog(
        combos=combos,
        features=features,
        colors=colors,
        changepoints=changepoints,
        trial_conditions=[],
    )


def catalog_from_nwb(source: str) -> tuple[DataCatalog, Any]:
    """Build a DataCatalog from an NWB file (local or remote).

    Returns ``(catalog, combo_catalog)`` — the combo_catalog is passed
    to :class:`NWBLoader` to avoid re-scanning the file.
    """
    from ethograph.io.nwb_backend import build_combos, catalog_nwb

    nwb_cat = catalog_nwb(source)
    combo_cat = build_combos(nwb_cat)

    combos: dict[str, ComboSpec] = {}
    features: list[str] = []

    for name, spec in combo_cat.combos.items():
        combos[name] = ComboSpec(name, spec.values)

    for entry in combo_cat.features:
        features.append(entry.display_name)

    combos["individuals"] = ComboSpec("individuals", ("individual_0",))

    catalog = DataCatalog(
        combos=combos,
        features=features,
        trial_conditions=[],
    )
    return catalog, combo_cat
