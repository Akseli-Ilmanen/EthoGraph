"""Unified combo catalog and loader for xarray, pynapple, and NWB backends.

Replaces the old type_vars_dict pattern with:
- DataCatalog: what dimensions/features are available (builds combo boxes)
- DataLoader: how to load data (select by feature + combo dims + time window → PlotData)
- ComboCatalog: NWB-specific combo detection and time-slice loading

Three backends, same interface. Differs in how combo dims are discovered,
how selection works (sel_valid principle: overspecified combos are OK), and
how time slicing works.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import pynapple as nap
    import xarray as xr

    import pandas as pd

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
        raise ValueError(
            "Irregular timestamps: use np.searchsorted on the timestamps dataset"
        )


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
# NWB combo detection: FeatureEntry, path parsing, tag inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureEntry:
    path: str
    display_name: str
    neurodata_type: str
    shape: tuple[int, ...]
    rate: float | None
    tags: dict[str, str]
    record: TimeSeriesRecord
    columns: tuple[str, ...] = ()
    has_confidence: bool = False


@dataclass(frozen=True, slots=True)
class TimeSlice:
    data: np.ndarray
    timestamps: np.ndarray
    feature: FeatureEntry
    confidence: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class StackedSlice:
    data: np.ndarray
    timestamps: np.ndarray
    labels: tuple[str, ...]
    confidence: np.ndarray | None = None


def _parse_path(path: str) -> tuple[str, str | None, str]:
    """Split an NWB processing path into (module, group_or_none, leaf).

    4-level: /processing/pose_estimation/LeftCamera/SeriesNoseTip
             -> ("pose_estimation", "LeftCamera", "SeriesNoseTip")

    3-level: /processing/wheel/WheelPosition
             -> ("wheel", None, "WheelPosition")
    """
    parts = path.strip("/").split("/")
    if len(parts) >= 4:
        return parts[1], parts[2], parts[3]
    if len(parts) == 3:
        return parts[1], None, parts[2]
    return parts[-1], None, parts[-1]


def _detect_real_groups(records: Sequence[TimeSeriesRecord]) -> set[str]:
    """A group is 'real' if multiple leaves share the same (module, group) pair."""
    group_leaf_count: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        module, group, _ = _parse_path(r.path)
        if group is not None:
            group_leaf_count[(module, group)] += 1
    return {group for (_, group), count in group_leaf_count.items() if count > 1}


_SPATIAL_TYPES = frozenset({"PoseEstimationSeries", "SpatialSeries"})
_SPACE_LABELS = {2: ("x", "y"), 3: ("x", "y", "z")}


def _infer_columns(neurodata_type: str, shape: tuple[int, ...]) -> tuple[str, ...]:
    if neurodata_type in _SPATIAL_TYPES and len(shape) >= 2:
        labels = _SPACE_LABELS.get(shape[-1])
        if labels:
            return labels
    return ()


def _detect_combos(
    records: Sequence[TimeSeriesRecord],
) -> tuple[dict[str, ComboSpec], list[FeatureEntry]]:
    real_groups = _detect_real_groups(records)

    tag_values: dict[str, set[str]] = defaultdict(set)
    parsed: list[tuple[TimeSeriesRecord, dict[str, str], tuple[str, ...]]] = []

    for r in records:
        module, group, leaf = _parse_path(r.path)
        cols = _infer_columns(r.neurodata_type, r.shape)

        tags: dict[str, str] = {"module": module}

        if group is not None and group in real_groups:
            tags["group"] = group

        if r.neurodata_type == "PoseEstimationSeries":
            tags["keypoint"] = leaf
        else:
            feature_name = leaf if group is None or group in real_groups else leaf
            tags["feature"] = feature_name

        if cols:
            for c in cols:
                tag_values["space"].add(c)

        for k, v in tags.items():
            tag_values[k].add(v)

        parsed.append((r, tags, cols))

    combos: dict[str, ComboSpec] = {}
    for level in ("module", "group", "keypoint", "feature", "space"):
        vals = tag_values.get(level, set())
        if vals:
            combos[level] = ComboSpec(name=level, values=tuple(sorted(vals)))

    features: list[FeatureEntry] = []
    for r, tags, cols in parsed:
        _, group, leaf = _parse_path(r.path)
        has_confidence = r.neurodata_type == "PoseEstimationSeries"
        display = leaf

        features.append(FeatureEntry(
            path=r.path,
            display_name=display,
            neurodata_type=r.neurodata_type,
            shape=r.shape,
            rate=r.rate,
            tags=tags,
            record=r,
            columns=cols,
            has_confidence=has_confidence,
        ))

    return combos, features


# ---------------------------------------------------------------------------
# ComboCatalog: time-slice loading via H5Like handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComboCatalog:
    combos: dict[str, ComboSpec]
    features: tuple[FeatureEntry, ...]

    def filter(self, **kwargs: str) -> tuple[FeatureEntry, ...]:
        return tuple(
            f for f in self.features
            if all(
                f.tags.get(k) == v
                for k, v in kwargs.items()
                if k in f.tags
            )
            and any(k in f.tags for k in kwargs)
        )

    def load_stacked(
        self,
        h5: H5Like,
        t0: float,
        t1: float,
        **combo_sel: str,
    ) -> StackedSlice:
        entry_sel, column_sel = self._split_sel(combo_sel)
        entries = self._sel_valid_entries(entry_sel)

        slices: list[TimeSlice] = []
        for entry in entries:
            data, timestamps, confidence = _read_time_slice(h5, entry, t0, t1)
            slices.append(TimeSlice(
                data=data, timestamps=timestamps, feature=entry, confidence=confidence,
            ))

        if not slices:
            return StackedSlice(
                data=np.empty((0, 0)),
                timestamps=np.empty(0),
                labels=(),
            )

        stacked = _stack_slices(slices)
        return _apply_column_sel(stacked, column_sel)

    def _sel_valid_entries(
        self, sel: dict[str, str],
    ) -> tuple[FeatureEntry, ...]:
        if not sel:
            return self.features
        return tuple(
            f for f in self.features
            if all(f.tags[k] == v for k, v in sel.items() if k in f.tags)
            and any(k in f.tags for k in sel)
        )

    def _split_sel(
        self, combo_sel: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        entry_keys = {"module", "group", "keypoint", "feature"}
        entry_sel = {k: v for k, v in combo_sel.items() if k in entry_keys}
        column_sel = {k: v for k, v in combo_sel.items() if k not in entry_keys}
        return entry_sel, column_sel

    def __repr__(self) -> str:
        combo_summary = {k: len(v) for k, v in self.combos.items()}
        return f"ComboCatalog(features={len(self.features)}, combos={combo_summary})"


def _read_time_slice(
    h5: H5Like,
    entry: FeatureEntry,
    t0: float,
    t1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rec = entry.record
    grp = h5[rec.path]
    data_ds = grp["data"]
    ts_ds = grp.get("timestamps")
    conf_ds = grp.get("confidence") if entry.has_confidence else None

    if rec.rate and rec.starting_time is not None:
        s = rec.time_to_slice(t0, t1)
        data = data_ds[s]
        timestamps = rec.starting_time + np.arange(s.start, s.stop) / rec.rate
        confidence = conf_ds[s] if conf_ds is not None else None
    elif ts_ds is not None:
        all_ts = ts_ds[:]
        mask = (all_ts >= t0) & (all_ts <= t1)
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            data = np.empty((0, *data_ds.shape[1:]), dtype=data_ds.dtype)
            timestamps = np.empty(0)
            confidence = np.empty(0) if conf_ds is not None else None
        else:
            s = slice(int(idx[0]), int(idx[-1]) + 1)
            data = data_ds[s]
            timestamps = all_ts[s]
            confidence = conf_ds[s] if conf_ds is not None else None
    else:
        raise ValueError(f"No timing info for {rec.path}")

    return data, timestamps, confidence


def _labels_for_entry(entry: FeatureEntry) -> list[str]:
    name = entry.display_name
    if entry.columns:
        return [f"{name}_{c}" for c in entry.columns]
    return [name]


def _stack_slices(slices: list[TimeSlice]) -> StackedSlice:
    timestamps = slices[0].timestamps
    labels: list[str] = []
    arrays: list[np.ndarray] = []
    conf_arrays: list[np.ndarray] = []
    has_any_confidence = False

    for s in slices:
        entry_labels = _labels_for_entry(s.feature)
        labels.extend(entry_labels)

        d = s.data
        if d.ndim == 1:
            d = d[:, np.newaxis]
        arrays.append(d)

        if s.confidence is not None:
            has_any_confidence = True
            conf_arrays.append(s.confidence)

    stacked = np.concatenate(arrays, axis=1)

    confidence = None
    if has_any_confidence:
        confidence = np.concatenate(
            [c[:, np.newaxis] if c.ndim == 1 else c for c in conf_arrays], axis=1
        ) if conf_arrays else None

    return StackedSlice(
        data=stacked,
        timestamps=timestamps,
        labels=tuple(labels),
        confidence=confidence,
    )


def _apply_column_sel(stacked: StackedSlice, column_sel: dict[str, str]) -> StackedSlice:
    if not column_sel:
        return stacked
    mask = list(range(len(stacked.labels)))
    for key, val in column_sel.items():
        suffix = f"_{val}"
        candidate = [i for i in mask if stacked.labels[i].endswith(suffix)]
        if candidate:
            mask = candidate
    if len(mask) == len(stacked.labels):
        return stacked
    return StackedSlice(
        data=stacked.data[:, mask],
        timestamps=stacked.timestamps,
        labels=tuple(stacked.labels[i] for i in mask),
        confidence=stacked.confidence,
    )


# ---------------------------------------------------------------------------
# build_combos: NWBCatalog → ComboCatalog
# ---------------------------------------------------------------------------


def build_combos(catalog: NWBCatalog) -> ComboCatalog:
    combos, features = _detect_combos(catalog.timeseries)
    return ComboCatalog(combos=combos, features=tuple(features))


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
        source,
        catalog: DataCatalog,
        combo_catalog: Any | None = None,
    ) -> None:
        self._source = source
        self._catalog = catalog
        self._current_trial_idx = 0
        self._trial_intervals: list[tuple[float, float]] = []

        if combo_catalog is None:
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
        from ethograph.utils.nwb import open_nwb

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

    if "individuals" in ds.coords:
        vals = tuple(ds.coords["individuals"].values.astype(str))
        combos["individuals"] = ComboSpec("individuals", vals)

    features_list = _feature_vars(ds)
    changepoints_list = list(ds.filter_by_attrs(type="changepoints").data_vars)

    if features_list:
        combos["features"] = ComboSpec("features", tuple(features_list))

    for name in find_temporal_dims(ds):
        if name in combos or name.upper() in _HIDDEN_DIMS:
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

    if "individuals" in ds.coords:
        vals = tuple(ds.coords["individuals"].values.astype(str))
        combos["individuals"] = ComboSpec("individuals", vals)

    features_list = _feature_vars(ds)
    changepoints_list = list(ds.filter_by_attrs(type="changepoints").data_vars)

    if features_list:
        combos["features"] = ComboSpec("features", tuple(features_list))

    extra_dims = find_temporal_dims(ds)
    for name in extra_dims:
        if name in combos or name.upper() in _HIDDEN_DIMS:
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
        changepoints=changepoints,
        trial_conditions=[],
    )


def catalog_from_nwb(source: str) -> tuple[DataCatalog, Any]:
    """Build a DataCatalog from an NWB file (local or remote).

    Returns ``(catalog, combo_catalog)`` — the combo_catalog is passed
    to :class:`NWBLoader` to avoid re-scanning the file.
    """
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


# ---------------------------------------------------------------------------
# NWBCatalog: result of scanning an NWB file
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NWBCatalog:
    source: str
    backend: NWBBackend
    timeseries: tuple[TimeSeriesRecord, ...]
    intervals: tuple[TimeIntervalsRecord, ...]

    def __repr__(self) -> str:
        return (
            f"NWBCatalog(source={self.source!r}, "
            f"timeseries={len(self.timeseries)}, "
            f"intervals={len(self.intervals)})"
        )


# ---------------------------------------------------------------------------
# NWB H5 scanning helpers (used by catalog_nwb / read_trial_intervals)
# ---------------------------------------------------------------------------

_TIMESERIES_TYPES = frozenset({
    "TimeSeries",
    "SpatialSeries",
    "IntervalSeries",
    "PoseEstimationSeries",
    "AnnotationSeries",
    "AbstractFeatureSeries",
    "IndexSeries",
    "ImageSeries",
})

_SKIP_MODULES = frozenset({"ecephys", "ophys", "ogen"})

_ALLOWED_ROOTS = ("processing/", "stimulus/", "intervals/", "intervals", "trials", "epochs")

_INTERVAL_TYPES = frozenset({
    "TimeIntervals",
    "DynamicTable",
})


def _read_h5_scalar(ds: Any, default: str = "") -> str:
    if ds is None:
        return default
    try:
        val = ds[()]
    except Exception:
        return default
    if isinstance(val, bytes):
        return val.decode()
    return str(val)


def _read_h5_attr(obj: Any, key: str, default: Any = None) -> Any:
    try:
        return obj.attrs.get(key, default)
    except Exception:
        return default


def _is_h5_group(obj: Any) -> bool:
    import h5py

    if isinstance(obj, h5py.Group):
        return True
    try:
        type_name = type(obj).__name__
        if "Group" in type_name or "File" in type_name:
            return True
    except Exception:
        pass
    return hasattr(obj, "keys") and hasattr(obj, "attrs") and not hasattr(obj, "shape")


def _is_h5_dataset(obj: Any) -> bool:
    return hasattr(obj, "shape") and hasattr(obj, "dtype")


def _should_visit(name: str) -> bool:
    if not any(name.startswith(root) for root in _ALLOWED_ROOTS):
        return False
    parts = name.split("/")
    return not any(p in _SKIP_MODULES for p in parts)


def _collect_timeseries(file: H5Like) -> list[TimeSeriesRecord]:
    """Walk an NWB H5 file and return all TimeSeries-like groups."""
    records: list[TimeSeriesRecord] = []

    def _visitor(name: str, obj: Any) -> None:
        if not _should_visit(name):
            return
        if not _is_h5_group(obj):
            return

        ndt = _read_h5_attr(obj, "neurodata_type", "")
        if isinstance(ndt, bytes):
            ndt = ndt.decode()
        if ndt not in _TIMESERIES_TYPES and "TimeSeries" not in ndt:
            return

        data = obj.get("data")
        if data is None or not _is_h5_dataset(data):
            return

        ts_ds = obj.get("timestamps")
        st_ds = obj.get("starting_time")

        rate_val = _read_h5_attr(obj, "rate")
        if rate_val is None and st_ds is not None:
            rate_val = _read_h5_attr(st_ds, "rate")

        ts_range = None
        n_ts = None
        if ts_ds is not None and _is_h5_dataset(ts_ds) and ts_ds.shape[0] > 0:
            n_ts = ts_ds.shape[0]
            try:
                ts_range = (float(ts_ds[0]), float(ts_ds[-1]))
            except Exception:
                pass

        starting_time = None
        if st_ds is not None:
            try:
                starting_time = float(st_ds[()])
            except Exception:
                pass

        records.append(TimeSeriesRecord(
            path=f"/{name}",
            name=name.rsplit("/", 1)[-1],
            neurodata_type=ndt,
            description=_read_h5_scalar(obj.get("description")),
            shape=tuple(data.shape),
            dtype=str(data.dtype),
            unit=_read_h5_scalar(obj.get("unit"), "unknown"),
            rate=float(rate_val) if rate_val is not None else None,
            starting_time=starting_time,
            n_timestamps=n_ts,
            timestamps_range=ts_range,
        ))

    file.visititems(_visitor)
    return records


def _collect_intervals(file: H5Like) -> list[TimeIntervalsRecord]:
    """Walk an NWB H5 file and return all TimeIntervals groups."""
    records: list[TimeIntervalsRecord] = []

    def _visitor(name: str, obj: Any) -> None:
        if not _should_visit(name):
            return
        if not _is_h5_group(obj):
            return

        ndt = _read_h5_attr(obj, "neurodata_type", "")
        if isinstance(ndt, bytes):
            ndt = ndt.decode()
        if ndt not in _INTERVAL_TYPES:
            return

        start_ds = obj.get("start_time")
        stop_ds = obj.get("stop_time")
        if start_ds is None or not _is_h5_dataset(start_ds):
            return

        n_rows = start_ds.shape[0]

        time_range = None
        if n_rows > 0 and stop_ds is not None and _is_h5_dataset(stop_ds):
            try:
                time_range = (float(start_ds[0]), float(stop_ds[-1]))
            except Exception:
                pass

        col_names: list[str] = []
        for key in obj.keys():
            child = obj.get(key)
            if child is not None and _is_h5_dataset(child):
                col_names.append(key)

        records.append(TimeIntervalsRecord(
            path=f"/{name}",
            name=name.rsplit("/", 1)[-1],
            neurodata_type=ndt,
            description=_read_h5_scalar(obj.get("description")),
            n_rows=n_rows,
            column_names=tuple(sorted(col_names)),
            time_range=time_range,
        ))

    file.visititems(_visitor)
    return records


# ---------------------------------------------------------------------------
# catalog_nwb + read_trial_intervals: scan an NWB file
# ---------------------------------------------------------------------------


def catalog_nwb(
    source,
    backend=None,
) -> NWBCatalog:
    """Scan an NWB file and return an NWBCatalog of its TimeSeries and intervals."""
    from ethograph.utils.nwb import NWBBackend, _infer_backend, open_nwb

    is_path = isinstance(source, (str, Path))
    source_str = str(source) if is_path else repr(source)
    resolved_backend = backend or (
        _infer_backend(source_str) if is_path else NWBBackend.LOCAL
    )

    with open_nwb(source, backend) as f:
        ts_records = _collect_timeseries(f)
        iv_records = _collect_intervals(f)

    return NWBCatalog(
        source=source_str,
        backend=resolved_backend,
        timeseries=tuple(ts_records),
        intervals=tuple(iv_records),
    )


def read_trial_intervals(
    source: str | H5Like,
    backend: NWBBackend | None = None,
) -> list[tuple[float, float]]:
    """Read trial (start, stop) pairs from an NWB file's trials table.

    Looks for ``/intervals/trials`` or the first ``TimeIntervals`` group
    that has ``start_time`` and ``stop_time`` datasets.
    Returns an empty list if no trials table is found.
    """
    from ethograph.utils.nwb import open_nwb

    with open_nwb(source, backend) as f:
        # Try standard trials table first
        for path in ("intervals/trials", "trials"):
            grp = None
            try:
                grp = f[path]
            except (KeyError, Exception):
                continue
            if grp is None:
                continue
            start_ds = grp.get("start_time")
            stop_ds = grp.get("stop_time")
            if start_ds is not None and stop_ds is not None:
                starts = start_ds[:]
                stops = stop_ds[:]
                return [(float(s), float(e)) for s, e in zip(starts, stops)]

        # Fallback: scan for any TimeIntervals with start/stop
        iv_records = _collect_intervals(f)
        for rec in iv_records:
            grp = f[rec.path]
            start_ds = grp.get("start_time")
            stop_ds = grp.get("stop_time")
            if start_ds is not None and stop_ds is not None:
                starts = start_ds[:]
                stops = stop_ds[:]
                return [(float(s), float(e)) for s, e in zip(starts, stops)]

    return []
