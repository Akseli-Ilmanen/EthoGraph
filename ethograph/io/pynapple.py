"""Pynapple / NWB loading and feature extraction utilities."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pynapple as nap
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from ethograph.features.movement import get_angle_rgb



## IO

NWB_TYPE_MAP = {
    "TsGroup": nap.TsGroup,
    "TsdFrame": nap.TsdFrame,
    "Tsd": nap.Tsd,
    "Ts": nap.Ts,
    "IntervalSet": nap.IntervalSet,
}


def parse_nwb_types(nwb: nap.NWBFile) -> dict[str, type]:
    result = {}
    for line in str(nwb).split("\n"):
        line = line.strip("│┝┕┍ ━┯┑┿┥┷┙")
        if not line or "Keys" in line or "━" in line:
            continue
        parts = [p.strip() for p in line.split("│") if p.strip()]
        if len(parts) == 2:
            result[parts[0]] = NWB_TYPE_MAP.get(parts[1], parts[1])
    return result


def get_metadata(obj) -> dict:
    meta = {"type": type(obj).__name__}
    if isinstance(obj, nap.TsGroup):
        meta["n_units"] = len(obj)
        meta["metadata_columns"] = list(obj.metadata_columns)
    elif isinstance(obj, nap.TsdFrame):
        meta["columns"] = list(obj.columns)
        meta["shape"] = obj.shape
    elif isinstance(obj, (nap.Tsd, nap.Ts)):
        meta["shape"] = obj.shape
    elif isinstance(obj, nap.IntervalSet):
        meta["n_intervals"] = len(obj)
    return meta


def flatten_nap_folder(data, load_metadata: bool = False) -> dict[str, dict]:
    flat = {}
    for key, val in data.items():
        if isinstance(val, nap.NWBFile):
            type_map = parse_nwb_types(val)
            for nwb_key, nwb_type in type_map.items():
                entry = {"type": nwb_type}
                if load_metadata:
                    try:
                        entry.update(get_metadata(val[nwb_key]))
                    except Exception:
                        entry["error"] = "failed to load"
                flat[nwb_key] = entry
        elif isinstance(val, dict):
            flat.update(flatten_nap_folder(val, load_metadata))
        else:
            entry = {"type": type(val)}
            if load_metadata:
                entry.update(get_metadata(val))
            flat[key] = entry
    return flat



### Features


def _iter_units(
    data: nap.Tsd | nap.TsdFrame | nap.TsGroup,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    match data:
        case nap.Tsd():
            return {"0": (data.t, data.values)}
        case nap.TsdFrame():
            return {col: (data.t, data[col].values) for col in data.columns}
        case nap.TsGroup():
            return {uid: (data[uid].t, data[uid].values) for uid in data.index}
        case _:
            raise TypeError(f"Unsupported type: {type(data)}")


def _apply_changepoint_func(
    t: np.ndarray,
    x: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    result = np.asarray(f(x))
    if len(result) == len(x):
        return t[result.astype(bool)]
    return t[result.astype(int)]


def add_changepoints_to_nap(
    data: nap.Tsd | nap.TsdFrame | nap.TsGroup,
    target_feature: str,
    changepoint_func: Callable[[np.ndarray], np.ndarray],
    **func_kwargs,
) -> nap.TsGroup:
    f = partial(changepoint_func, **func_kwargs)
    units = _iter_units(data)

    ts_dict = {}
    for i, (_, (t, x)) in enumerate(units.items()):
        mask = np.zeros(len(t), dtype=np.float32)
        cp_times = _apply_changepoint_func(t, x, f)
        mask[np.isin(t, cp_times)] = 1.0
        ts_dict[i] = nap.Tsd(t=t, d=mask, time_support=data.time_support)

    group = nap.TsGroup(ts_dict, time_support=data.time_support)
    group.set_info(
        source_label=list(units.keys()),
        target_feature=[target_feature] * len(units),
        type=["changepoints"] * len(units),
    )

    if isinstance(data, nap.TsGroup) and data.metadata is not None:
        for col in data.metadata.columns:
            group.set_info(**{col: data.metadata[col].values})

    return group


def add_angle_rgb_to_nap(
    tsdframe: nap.TsdFrame,
    smoothing_params: dict,
    position_key: str = "position",
    xy_columns: list[str] = ["x", "y"],
) -> nap.TsdFrame:
    if tsdframe.shape[1] > 2:
        tsdframe = tsdframe[xy_columns]

    rgb, _ = get_angle_rgb(
        tsdframe.values,
        smooth_func=gaussian_filter1d,
        smoothing_params=smoothing_params,
        input_type=position_key,
    )

    return nap.TsdFrame(
        t=tsdframe.t, d=rgb, columns=["R", "G", "B"],
        time_support=tsdframe.time_support,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

PYNAPPLE_EXTENSIONS = {".nwb", ".npz"}


def detect_trials(data: dict) -> nap.IntervalSet | None:
    """Find a trials IntervalSet in a loaded pynapple data dict."""
    for key in ("trials", "epochs", "intervals"):
        obj = data.get(key)
        if isinstance(obj, nap.IntervalSet):
            return obj
    for key, obj in data.items():
        if isinstance(obj, nap.IntervalSet) and "trial" in key.lower():
            return obj
    return None


def _load_folder_as_dict(folder_path: Path) -> dict:
    """Load all .npz and .nwb files in a folder into a flat dict."""
    data = {}
    for f in sorted(folder_path.iterdir()):
        if f.suffix == ".npz":
            try:
                data[f.stem] = nap.load_file(str(f))
            except Exception:
                pass
        elif f.suffix == ".nwb":
            try:
                nwb = nap.load_file(str(f))
                type_map = parse_nwb_types(nwb)
                for key in type_map:
                    try:
                        data[key] = nwb[key]
                    except Exception:
                        pass
            except Exception:
                pass
    return data


def load_nap_data(path: str) -> tuple[dict, nap.IntervalSet | None]:
    """Load pynapple data from a file or folder.

    Supports ``.nwb``, ``.npz``, or a directory (pynapple folder).
    When loading a single ``.npz``, sibling files in the same directory
    are also loaded (e.g. ``trials.npz`` alongside ``speed.npz``).

    Returns
    -------
    data : dict
        Loaded pynapple objects keyed by name.
    trials_ep : IntervalSet or None
        Trial intervals if found in the data.
    """
    p = Path(path)
    if p.is_dir():
        data = _load_folder_as_dict(p)
    elif p.suffix == ".nwb":
        nwb = nap.load_file(str(p))
        type_map = parse_nwb_types(nwb)
        data = {}
        for key in type_map:
            try:
                data[key] = nwb[key]
            except Exception:
                pass
    elif p.suffix == ".npz":
        # Load all sibling npz files from the same directory
        data = _load_folder_as_dict(p.parent)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    trials_ep = detect_trials(data)
    return data, trials_ep


def _compute_shared_column_dims(
    feature_objs: dict[str, nap.Tsd | nap.TsdFrame | nap.TsdTensor],
) -> dict[str, str]:
    """Map each TsdFrame variable to a shared column dimension name.

    TsdFrame objects with identical column values share one xarray
    dimension, so the GUI shows a single combo instead of duplicates.
    """
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


def nap_to_metadata_trialtree(
    data: dict,
    trials_ep: nap.IntervalSet | None = None,
):
    """Create a lightweight TrialTree with time coords only (no data).

    Used alongside :class:`~ethograph.io.feature_store.PynappleStore`
    which handles lazy data access.  The TrialTree is still needed for
    trial navigation, session table, and label storage.
    """
    from ethograph.io.trialtree import TrialTree

    feature_objs = {
        k: v for k, v in data.items()
        if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))
    }

    def _time_coord_for_trial(ep_start: float, ep_end: float):
        """Build a representative time coordinate from the densest feature."""
        best_t = None
        for obj in feature_objs.values():
            if trials_ep is not None:
                restricted = obj.restrict(nap.IntervalSet(start=ep_start, end=ep_end))
            else:
                restricted = obj
            if len(restricted) == 0:
                continue
            t = restricted.t - ep_start
            if best_t is None or len(t) > len(best_t):
                best_t = t
        if best_t is None:
            best_t = np.array([0.0, ep_end - ep_start])
        return best_t

    if trials_ep is None or len(trials_ep) == 0:
        t_min = min(obj.t[0] for obj in feature_objs.values() if len(obj) > 0)
        t_max = max(obj.t[-1] for obj in feature_objs.values() if len(obj) > 0)
        t_coord = _time_coord_for_trial(t_min, t_max)
        ds = xr.Dataset(coords={"time": t_coord, "individuals": ["individual_0"]})
        ds.attrs["trial"] = 1
        return TrialTree.from_datasets([ds], validate=False)

    datasets = []
    for i in range(len(trials_ep)):
        start = float(trials_ep.start[i])
        end = float(trials_ep.end[i])
        t_coord = _time_coord_for_trial(start, end)
        ds = xr.Dataset(coords={"time": t_coord, "individuals": ["individual_0"]})
        ds.attrs["trial"] = i + 1
        datasets.append(ds)

    dt = TrialTree.from_datasets(datasets, validate=False)

    session_data = {
        "start_time": ("trial", list(trials_ep.start)),
        "stop_time": ("trial", list(trials_ep.end)),
    }
    session_ds = xr.Dataset(
        session_data,
        coords={"trial": list(range(1, len(trials_ep) + 1))},
    )
    dt.set_session_table(session_ds)

    return dt


def extract_type_vars_pynapple(data: dict) -> dict:
    """Build a type-variable catalogue from pynapple objects.

    Returns the same dict structure as
    :func:`ethograph.io.validation.extract_type_vars` so the GUI can
    populate combo boxes identically for both backends.
    """
    type_vars: dict = {}
    type_vars["individuals"] = np.array(["individual_0"])

    feature_objs = {
        k: v for k, v in data.items()
        if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))
    }
    col_dim_map = _compute_shared_column_dims(feature_objs)

    features = []
    colors = []
    changepoints = []
    extra_dims: dict[str, list] = {}

    for key, obj in data.items():
        if isinstance(obj, nap.IntervalSet):
            continue

        if isinstance(obj, nap.TsGroup):
            if hasattr(obj, "metadata") and obj.metadata is not None:
                meta = obj.metadata
                if "type" in meta.columns:
                    type_vals = meta["type"].unique()
                    if "changepoints" in type_vals:
                        changepoints.append(key)
                        continue
            continue

        if isinstance(obj, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
            features.append(key)

            if isinstance(obj, nap.TsdFrame):
                cols_lower = [c.lower() for c in obj.columns]
                if "rgb" in key.lower() or set(cols_lower) == {"r", "g", "b"}:
                    colors.append(key)

                dim_name = col_dim_map.get(key, f"{key}_columns")
                if dim_name not in extra_dims:
                    extra_dims[dim_name] = list(obj.columns)

    type_vars["features"] = features
    if colors:
        type_vars["colors"] = colors
    if changepoints:
        type_vars["changepoints"] = changepoints
    type_vars["trial_conditions"] = []

    for dim_name, values in extra_dims.items():
        type_vars[dim_name] = np.array(values)

    return type_vars

