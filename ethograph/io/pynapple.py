from functools import partial
from collections.abc import Callable

import numpy as np
import pynapple as nap
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
    
    
