"""Concrete TimeSource adapters for xarray, pynapple, and NWB backends.

Each adapter wraps an existing data backend and presents a uniform
:class:`~ethograph.io.time_model.TimeSource` interface with
session-absolute time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ethograph.io.time_model import TimeRange

if TYPE_CHECKING:
    import pynapple as nap
    import xarray as xr

    from ethograph.io.nwb_backend import ComboCatalog, FeatureEntry


# ---------------------------------------------------------------------------
# XarrayTrialSource
# ---------------------------------------------------------------------------


class XarrayTrialSource:
    """TimeSource wrapping one xarray variable across TrialTree trials.

    Works in trial-local time (0-based), matching how TrialTree stores
    per-trial datasets. Call :meth:`set_dataset` on trial change to swap
    the backing dataset.
    """

    def __init__(
        self,
        name: str,
        ds: xr.Dataset,
        time_coord_name: str = "time",
    ) -> None:
        self._name = name
        self._ds = ds
        self._time_coord_name = time_coord_name
        self._update_range()

    def _update_range(self) -> None:
        tc = self._ds.coords.get(self._time_coord_name)
        if tc is not None and len(tc) > 0:
            vals = tc.values
            self._time_range = TimeRange(float(vals[0]), float(vals[-1]))
            if len(vals) > 1:
                self._sampling_rate = 1.0 / float(vals[1] - vals[0])
            else:
                self._sampling_rate = None
        else:
            self._time_range = TimeRange(0.0, 0.0)
            self._sampling_rate = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_range(self) -> TimeRange:
        return self._time_range

    @property
    def sampling_rate(self) -> float | None:
        return self._sampling_rate

    def set_dataset(self, ds: xr.Dataset) -> None:
        """Swap backing dataset on trial change."""
        self._ds = ds
        self._update_range()

    def get_data(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        if self._name not in self._ds.data_vars:
            return np.array([], dtype=np.float64), np.array([])
        var = self._ds[self._name]
        sliced = var.sel({self._time_coord_name: slice(t0, t1)})
        tc = sliced.coords[self._time_coord_name].values.astype(np.float64)
        return tc, sliced.values


# ---------------------------------------------------------------------------
# PynappleSource
# ---------------------------------------------------------------------------


class PynappleSource:
    """TimeSource wrapping a pynapple Tsd/TsdFrame/TsdTensor.

    Uses pynapple's native ``restrict()`` for time slicing.  When trials
    are present, call :meth:`set_trial` to update the active trial.
    """

    def __init__(
        self,
        name: str,
        obj: Any,
        trials_ep: nap.IntervalSet | None = None,
    ) -> None:
        self._name = name
        self._obj = obj
        self._trials_ep = trials_ep
        self._current_trial_idx = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_range(self) -> TimeRange:
        if len(self._obj) == 0:
            return TimeRange(0.0, 0.0)
        return TimeRange(float(self._obj.t[0]), float(self._obj.t[-1]))

    @property
    def sampling_rate(self) -> float | None:
        return float(self._obj.rate) if hasattr(self._obj, "rate") and self._obj.rate else None

    def set_trial(self, trial_idx: int) -> None:
        self._current_trial_idx = trial_idx

    def get_data(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        import pynapple as nap

        ep = nap.IntervalSet(start=t0, end=t1)
        restricted = self._obj.restrict(ep)
        if len(restricted) == 0:
            return np.array([], dtype=np.float64), np.array([])
        return restricted.t.astype(np.float64), np.asarray(restricted.values)


# ---------------------------------------------------------------------------
# NWBTimeSource
# ---------------------------------------------------------------------------


class NWBTimeSource:
    """TimeSource wrapping one NWB TimeSeries via ComboCatalog.

    Uses HDF5 slicing (rate-based or searchsorted on timestamps).
    Compatible with local files and remote access via remfile.
    """

    def __init__(
        self,
        name: str,
        source_path: str,
        entry: FeatureEntry,
        combo_catalog: ComboCatalog,
    ) -> None:
        self._name = name
        self._source_path = source_path
        self._entry = entry
        self._combo_catalog = combo_catalog
        self._time_range = self._compute_range()

    def _compute_range(self) -> TimeRange:
        rec = self._entry.record
        if rec.rate and rec.rate > 0:
            start = rec.starting_time or 0.0
            end = start + rec.shape[0] / rec.rate
            return TimeRange(start, end)
        if rec.timestamps_range:
            return TimeRange(rec.timestamps_range[0], rec.timestamps_range[1])
        return TimeRange(0.0, 0.0)

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_range(self) -> TimeRange:
        return self._time_range

    @property
    def sampling_rate(self) -> float | None:
        return self._entry.record.rate

    def get_data(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        from ethograph.io.nwb_backend import open_nwb

        combo_sel = {"feature": self._entry.display_name}

        with open_nwb(self._source_path) as h5:
            stacked = self._combo_catalog.load_stacked(h5, t0, t1, **combo_sel)

        if stacked.data.size == 0:
            return np.array([], dtype=np.float64), np.array([])

        data = stacked.data
        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]

        return stacked.timestamps.astype(np.float64), data


# ---------------------------------------------------------------------------
# MediaTimeSource — metadata-only source from NWB ImageSeries
# ---------------------------------------------------------------------------


class MediaTimeSource:
    """TimeSource representing an external media stream (video, audio, pose).

    Does not load actual data — only provides time range and rate from
    NWB ImageSeries metadata. Used by SourceCollection for range queries
    so that ``union_range`` includes media file durations.
    """

    def __init__(
        self,
        name: str,
        start_s: float,
        end_s: float,
        rate: float | None,
    ) -> None:
        self._name = name
        self._time_range = TimeRange(start_s, end_s)
        self._sampling_rate = rate

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_range(self) -> TimeRange:
        return self._time_range

    @property
    def sampling_rate(self) -> float | None:
        return self._sampling_rate

    def get_data(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([], dtype=np.float64), np.array([])
