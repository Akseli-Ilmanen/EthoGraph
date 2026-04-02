"""Unified feature data access for xarray and pynapple backends.

FeatureStore provides a backend-agnostic interface so the GUI can
select features and dimensions without knowing whether the data comes
from an xarray Dataset or raw pynapple objects.  Both backends produce
:class:`PlotData` — a pure-numpy container consumed by the rendering
functions in ``plots_lineplot.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

import ethograph as eto
from ethograph.io.validation import find_temporal_dims


# ---------------------------------------------------------------------------
# PlotData — source-agnostic container for rendering
# ---------------------------------------------------------------------------

@dataclass
class PlotData:
    """Prepared data ready for rendering — source-agnostic."""

    time: np.ndarray
    data: np.ndarray  # (T,) or (T, N)
    dim_labels: list[str] | None = None
    title: str = ""
    ylabel: str = ""
    color_data: np.ndarray | None = None
    changepoints: dict[str, np.ndarray] | None = None
    boundary_events: np.ndarray | None = None


# ---------------------------------------------------------------------------
# FeatureStore protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FeatureStore(Protocol):
    """Backend-agnostic interface for feature data access."""

    @property
    def backend(self) -> str: ...

    @property
    def features(self) -> list[str]: ...

    @property
    def dims(self) -> dict[str, np.ndarray]: ...

    @property
    def colors(self) -> list[str]: ...

    @property
    def changepoint_names(self) -> list[str]: ...

    def get_type_vars(self) -> dict:
        """Return type_vars_dict for combo population."""
        ...

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float | None = None,
        t1: float | None = None,
        color_variable: str | None = None,
    ) -> PlotData | None: ...

    def time_range(self, feature: str | None = None) -> tuple[float, float]: ...

    def set_trial(self, trial_idx: int) -> None: ...


# ---------------------------------------------------------------------------
# XarrayStore
# ---------------------------------------------------------------------------

class XarrayStore:
    """Feature access backed by an ``xr.Dataset``.

    Wraps the existing ``sel_valid`` logic.  The dataset reference is
    updated on each trial change via :meth:`update_ds`.
    """

    def __init__(self, ds):
        self._ds = ds

    @property
    def backend(self) -> str:
        return "xarray"

    @property
    def features(self) -> list[str]:
        if self._ds is None:
            return []
        return list(self._ds.filter_by_attrs(type="features").data_vars)

    @property
    def dims(self) -> dict[str, np.ndarray]:
        if self._ds is None:
            return {}
        result: dict[str, np.ndarray] = {}
        dim_names = find_temporal_dims(self._ds)
        for name in dim_names:
            if name in self._ds.coords:
                coord = self._ds.coords[name]
                if coord.dtype.kind in ("U", "S", "O"):
                    result[name] = coord.values.astype(str)
                else:
                    result[name] = coord.values
            else:
                result[name] = np.arange(self._ds.sizes[name])
        if "individuals" in self._ds.coords:
            result["individuals"] = self._ds.coords["individuals"].values.astype(str)
        return result

    @property
    def colors(self) -> list[str]:
        if self._ds is None:
            return []
        return list(self._ds.filter_by_attrs(type="colors").data_vars)

    @property
    def changepoint_names(self) -> list[str]:
        if self._ds is None:
            return []
        return list(self._ds.filter_by_attrs(type="changepoints").data_vars)

    def get_type_vars(self) -> dict:
        """Produce a type_vars_dict compatible with combo population."""
        tvd: dict = {}
        tvd["features"] = self.features
        if self.colors:
            tvd["colors"] = self.colors
        if self.changepoint_names:
            tvd["changepoints"] = self.changepoint_names
        for dim_name, values in self.dims.items():
            tvd[dim_name] = values
        tvd["trial_conditions"] = []
        return tvd

    # -- data access --------------------------------------------------------

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float | None = None,
        t1: float | None = None,
        color_variable: str | None = None,
    ) -> PlotData | None:
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

        # Dim labels for 2-D data
        dim_labels = None
        if data.ndim == 2:
            non_time_dim = next(
                (d for d in var_sel.dims if "time" not in d.lower()), None
            )
            if non_time_dim and non_time_dim in var_sel.coords:
                dim_labels = [str(c) for c in var_sel.coords[non_time_dim].values]
            else:
                dim_labels = [str(i) for i in range(data.shape[1])]

        # Color data (1-D features only)
        color_data = None
        if data.ndim == 1 and color_variable and color_variable in ds.data_vars:
            color_kwargs = {k: v for k, v in selections.items() if k != "RGB"}
            color_data, _ = eto.sel_valid(ds[color_variable], color_kwargs)

        # Changepoints (1-D features only)
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

        # Boundary events
        boundary_events = None
        if "boundary_events" in ds.data_vars:
            raw = ds["boundary_events"].values
            valid = raw[~np.isnan(raw)].astype(int)
            valid = valid[(valid >= 0) & (valid < len(time))]
            if len(valid) > 0:
                boundary_events = time[valid]

        # Title / ylabel
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

    def set_trial(self, trial_idx: int) -> None:
        pass  # Trial switching handled externally via update_ds

    def update_ds(self, ds) -> None:
        """Swap the backing dataset (called on trial change)."""
        self._ds = ds


# ---------------------------------------------------------------------------
# PynappleStore
# ---------------------------------------------------------------------------

class PynappleStore:
    """Lazy feature access backed by raw pynapple objects.

    Data is never copied into xarray.  On each :meth:`select` call the
    pynapple object is ``restrict()``-ed to the current trial and time
    window, then column-selected if needed, and returned as numpy.
    """

    def __init__(self, data: dict, trials_ep=None):
        import pynapple as nap

        self._data = data
        self._trials_ep = trials_ep
        self._current_trial_idx = 0

        self._feature_objs: dict = {
            k: v
            for k, v in data.items()
            if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))
        }

        from ethograph.io.pynapple import _compute_shared_column_dims

        self._dim_map = _compute_shared_column_dims(self._feature_objs)

        # Build dims
        self._dims: dict[str, np.ndarray] = {
            "individuals": np.array(["individual_0"])
        }
        seen_dims: set[str] = set()
        for name, dim_name in self._dim_map.items():
            if dim_name not in seen_dims:
                obj = self._feature_objs[name]
                if isinstance(obj, nap.TsdFrame):
                    self._dims[dim_name] = np.array(list(obj.columns))
                    seen_dims.add(dim_name)

        # Detect color variables
        self._colors: list[str] = []
        for key, obj in self._feature_objs.items():
            if isinstance(obj, nap.TsdFrame):
                cols_lower = [c.lower() for c in obj.columns]
                if "rgb" in key.lower() or set(cols_lower) == {"r", "g", "b"}:
                    self._colors.append(key)

    # -- properties ---------------------------------------------------------

    @property
    def backend(self) -> str:
        return "pynapple"

    @property
    def features(self) -> list[str]:
        return list(self._feature_objs.keys())

    @property
    def dims(self) -> dict[str, np.ndarray]:
        return dict(self._dims)

    @property
    def colors(self) -> list[str]:
        return list(self._colors)

    @property
    def changepoint_names(self) -> list[str]:
        import pynapple as nap

        result: list[str] = []
        for key, obj in self._data.items():
            if isinstance(obj, nap.TsGroup):
                if hasattr(obj, "metadata") and obj.metadata is not None:
                    meta = obj.metadata
                    if "type" in meta.columns and "changepoints" in meta["type"].unique():
                        result.append(key)
        return result

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
        """Absolute start time of the current trial."""
        if self._trials_ep is None:
            times = [obj.t[0] for obj in self._feature_objs.values() if len(obj) > 0]
            return min(times) if times else 0.0
        return float(self._trials_ep.start[self._current_trial_idx])

    @property
    def trial_bounds(self) -> tuple[float, float]:
        """Current trial as (0, duration) in trial-relative time."""
        if self._trials_ep is None:
            all_t = []
            for obj in self._feature_objs.values():
                if len(obj) > 0:
                    all_t.extend([obj.t[0], obj.t[-1]])
            return (0.0, max(all_t) - min(all_t)) if all_t else (0.0, 0.0)
        start = float(self._trials_ep.start[self._current_trial_idx])
        end = float(self._trials_ep.end[self._current_trial_idx])
        return (0.0, end - start)

    # -- type vars for combos -----------------------------------------------

    def get_type_vars(self) -> dict:
        tvd: dict = {}
        tvd["features"] = self.features
        if self._colors:
            tvd["colors"] = self._colors
        cp = self.changepoint_names
        if cp:
            tvd["changepoints"] = cp
        for dim_name, values in self._dims.items():
            tvd[dim_name] = values
        tvd["trial_conditions"] = []
        return tvd

    # -- trial management ---------------------------------------------------

    def set_trial(self, trial_idx: int) -> None:
        self._current_trial_idx = trial_idx

    # -- data access --------------------------------------------------------

    def _restrict(self, obj, t0: float | None = None, t1: float | None = None):
        """Restrict a pynapple object to the current trial + optional window.

        Returns ``(restricted_obj, trial_offset)`` where trial_offset is
        the absolute start time to subtract for trial-relative output.
        """
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

        # Title
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
        )

    def time_range(self, feature: str | None = None) -> tuple[float, float]:
        return self.trial_bounds
