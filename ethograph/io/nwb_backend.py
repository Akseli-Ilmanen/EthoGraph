"""NWB backend: lightweight catalog, combo detection, and time-slice loading.

Supports local HDF5, remote via remfile, remote via lindi, and pre-opened
h5py.File / LindiH5pyFile handles.

Public names
------------
H5Like, NWBBackend, TimeSeriesRecord, TimeIntervalsRecord,
open_nwb, catalog_nwb, NWBCatalog,
FeatureEntry, TimeSlice, StackedSlice, ComboCatalog, build_combos
"""

from __future__ import annotations

import enum
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np

from ethograph.io.catalog import ComboSpec

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Protocols: anything that looks like an h5py.File / Group / Dataset
# ---------------------------------------------------------------------------


@runtime_checkable
class H5Like(Protocol):
    def visititems(self, func: Any) -> None: ...
    def __getitem__(self, key: str) -> Any: ...


@runtime_checkable
class H5Group(Protocol):
    attrs: Any
    def get(self, key: str) -> Any: ...
    def keys(self) -> Any: ...


@runtime_checkable
class H5Dataset(Protocol):
    shape: tuple[int, ...]
    dtype: Any
    def __getitem__(self, key: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------


class NWBBackend(enum.Enum):
    LOCAL = "local"
    REMFILE = "remfile"
    LINDI = "lindi"


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeSeriesRecord:
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
    path: str
    name: str
    neurodata_type: str
    description: str
    n_rows: int
    column_names: tuple[str, ...]
    time_range: tuple[float, float] | None


# ---------------------------------------------------------------------------
# File opener: local, remfile, lindi → h5py-like handle
# ---------------------------------------------------------------------------


@contextmanager
def open_nwb(
    source: str | Path | H5Like,
    backend: NWBBackend | None = None,
) -> Iterator[H5Like]:
    if isinstance(source, H5Like):
        yield source
        return

    source = str(source)
    resolved_backend = backend or _infer_backend(source)
    closables: list[Any] = []

    try:
        if resolved_backend == NWBBackend.LINDI:
            import lindi

            if source.endswith((".lindi.json", ".lindi.tar", ".lindi.d")):
                f = lindi.LindiH5pyFile.from_lindi_file(source)
            else:
                f = lindi.LindiH5pyFile.from_hdf5_file(source)
            closables.append(f)
            yield f

        elif resolved_backend == NWBBackend.REMFILE:
            import h5py
            import remfile

            rem = remfile.File(source)
            closables.append(rem)
            h5 = h5py.File(rem, "r")
            closables.append(h5)
            yield h5

        else:
            import h5py

            h5 = h5py.File(source, "r")
            closables.append(h5)
            yield h5

    finally:
        for obj in reversed(closables):
            try:
                obj.close()
            except Exception:
                pass


def _infer_backend(source: str) -> NWBBackend:
    if any(source.endswith(ext) for ext in (".lindi.json", ".lindi.tar", ".lindi.d")):
        return NWBBackend.LINDI
    if source.startswith(("http://", "https://", "s3://")):
        return NWBBackend.LINDI
    return NWBBackend.LOCAL


# ---------------------------------------------------------------------------
# NWB traversal: internal helpers
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


def _read_scalar(ds: Any, default: str = "") -> str:
    if ds is None:
        return default
    try:
        val = ds[()]
    except Exception:
        return default
    if isinstance(val, bytes):
        return val.decode()
    return str(val)


def _read_attr(obj: Any, key: str, default: Any = None) -> Any:
    try:
        return obj.attrs.get(key, default)
    except Exception:
        return default


def _is_group(obj: Any) -> bool:
    import h5py

    if isinstance(obj, h5py.Group):
        return True
    try:
        from lindi import LindiH5pyFile

        type_name = type(obj).__name__
        if "Group" in type_name or "File" in type_name:
            return True
    except ImportError:
        pass
    return hasattr(obj, "keys") and hasattr(obj, "attrs") and not hasattr(obj, "shape")


def _is_dataset(obj: Any) -> bool:
    return hasattr(obj, "shape") and hasattr(obj, "dtype")


def _should_visit(name: str) -> bool:
    if not any(name.startswith(root) for root in _ALLOWED_ROOTS):
        return False
    parts = name.split("/")
    return not any(p in _SKIP_MODULES for p in parts)


def _collect_timeseries(file: H5Like) -> list[TimeSeriesRecord]:
    records: list[TimeSeriesRecord] = []

    def _visitor(name: str, obj: Any) -> None:
        if not _should_visit(name):
            return
        if not _is_group(obj):
            return

        ndt = _read_attr(obj, "neurodata_type", "")
        if isinstance(ndt, bytes):
            ndt = ndt.decode()
        if ndt not in _TIMESERIES_TYPES and "TimeSeries" not in ndt:
            return

        data = obj.get("data")
        if data is None or not _is_dataset(data):
            return

        ts_ds = obj.get("timestamps")
        st_ds = obj.get("starting_time")

        rate_val = _read_attr(obj, "rate")
        if rate_val is None and st_ds is not None:
            rate_val = _read_attr(st_ds, "rate")

        ts_range = None
        n_ts = None
        if ts_ds is not None and _is_dataset(ts_ds) and ts_ds.shape[0] > 0:
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
            description=_read_scalar(obj.get("description")),
            shape=tuple(data.shape),
            dtype=str(data.dtype),
            unit=_read_scalar(obj.get("unit"), "unknown"),
            rate=float(rate_val) if rate_val is not None else None,
            starting_time=starting_time,
            n_timestamps=n_ts,
            timestamps_range=ts_range,
        ))

    file.visititems(_visitor)
    return records


def _collect_intervals(file: H5Like) -> list[TimeIntervalsRecord]:
    records: list[TimeIntervalsRecord] = []

    def _visitor(name: str, obj: Any) -> None:
        if not _should_visit(name):
            return
        if not _is_group(obj):
            return

        ndt = _read_attr(obj, "neurodata_type", "")
        if isinstance(ndt, bytes):
            ndt = ndt.decode()
        if ndt not in _INTERVAL_TYPES:
            return

        start_ds = obj.get("start_time")
        stop_ds = obj.get("stop_time")
        if start_ds is None or not _is_dataset(start_ds):
            return

        n_rows = start_ds.shape[0]

        time_range = None
        if n_rows > 0 and stop_ds is not None and _is_dataset(stop_ds):
            try:
                time_range = (float(start_ds[0]), float(stop_ds[-1]))
            except Exception:
                pass

        col_names: list[str] = []
        for key in obj.keys():
            child = obj.get(key)
            if child is not None and _is_dataset(child):
                col_names.append(key)

        records.append(TimeIntervalsRecord(
            path=f"/{name}",
            name=name.rsplit("/", 1)[-1],
            neurodata_type=ndt,
            description=_read_scalar(obj.get("description")),
            n_rows=n_rows,
            column_names=tuple(sorted(col_names)),
            time_range=time_range,
        ))

    file.visititems(_visitor)
    return records


# ---------------------------------------------------------------------------
# Public API: NWBCatalog + catalog_nwb
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NWBCatalog:
    source: str
    backend: NWBBackend
    timeseries: tuple[TimeSeriesRecord, ...]
    intervals: tuple[TimeIntervalsRecord, ...]

    def to_timeseries_df(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame([
            {
                "path": r.path,
                "type": r.neurodata_type,
                "shape": r.shape,
                "dtype": r.dtype,
                "unit": r.unit,
                "rate": r.rate,
                "time_range": r.time_range,
                "duration": r.duration,
                "regular": r.is_regularly_sampled,
            }
            for r in self.timeseries
        ])

    def to_intervals_df(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame([
            {
                "path": r.path,
                "type": r.neurodata_type,
                "n_rows": r.n_rows,
                "columns": r.column_names,
                "time_range": r.time_range,
            }
            for r in self.intervals
        ])

    def filter_by_type(self, neurodata_type: str) -> list[TimeSeriesRecord]:
        return [r for r in self.timeseries if r.neurodata_type == neurodata_type]

    def filter_by_path(self, prefix: str) -> list[TimeSeriesRecord]:
        return [r for r in self.timeseries if r.path.startswith(prefix)]

    def __repr__(self) -> str:
        return (
            f"NWBCatalog(source={self.source!r}, "
            f"timeseries={len(self.timeseries)}, "
            f"intervals={len(self.intervals)})"
        )


def catalog_nwb(
    source: str | Path | H5Like,
    backend: NWBBackend | None = None,
) -> NWBCatalog:
    source_str = str(source) if not isinstance(source, H5Like) else repr(source)
    resolved_backend = backend or (
        _infer_backend(source_str) if not isinstance(source, H5Like) else NWBBackend.LOCAL
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
    source: str | Path | H5Like,
    backend: NWBBackend | None = None,
) -> list[tuple[float, float]]:
    """Read trial (start, stop) pairs from an NWB file's trials table.

    Looks for ``/intervals/trials`` or the first ``TimeIntervals`` group
    that has ``start_time`` and ``stop_time`` datasets.
    Returns an empty list if no trials table is found.
    """
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


# ---------------------------------------------------------------------------
# Combo detection: FeatureEntry, path parsing, tag inference
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

    def values(self, combo_name: str) -> tuple[str, ...]:
        return self.combos[combo_name].values

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

    def combos_for_module(self, module: str) -> dict[str, ComboSpec]:
        entries = self.filter(module=module)
        result: dict[str, ComboSpec] = {}
        for combo_name in ("group", "keypoint", "feature", "space"):
            vals = sorted({
                e.tags[combo_name]
                for e in entries
                if combo_name in e.tags
            })
            if vals:
                result[combo_name] = ComboSpec(name=combo_name, values=tuple(vals))
        return result

    def load_slice(
        self,
        h5: H5Like,
        t0: float,
        t1: float,
        **combo_sel: str,
    ) -> list[TimeSlice]:
        entries = self._sel_valid_entries(combo_sel)
        results: list[TimeSlice] = []
        for entry in entries:
            data, timestamps, confidence = _read_time_slice(h5, entry, t0, t1)
            results.append(TimeSlice(
                data=data, timestamps=timestamps, feature=entry, confidence=confidence,
            ))
        return results

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
# Public API: build_combos
# ---------------------------------------------------------------------------


def build_combos(catalog: NWBCatalog) -> ComboCatalog:
    combos, features = _detect_combos(catalog.timeseries)
    return ComboCatalog(combos=combos, features=tuple(features))
