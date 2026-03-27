from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from ethograph.features.movement import get_angle_rgb, extract_video_motion

from ethograph.utils.xr_utils import get_time_coord
from ethograph.utils.label_intervals import INTERVAL_COLUMNS, empty_intervals, intervals_to_xr
from ethograph.utils.trialtree import TrialTree


def dataset_to_basic_trialtree(ds, video_path: Optional[str] = None, video_motion: bool = False) -> TrialTree:
    """Converts single dataset (single trial only) into data tree with empty labels, video motion and attributes"""

    if "labels" not in ds.data_vars and "onset_s" not in ds.data_vars:
        interval_ds = intervals_to_xr(empty_intervals())
        for var_name in interval_ds.data_vars:
            ds[var_name] = interval_ds[var_name]

    if video_motion and video_path is not None:
        ds["video_motion"] = extract_video_motion(video_path, fps=ds.fps, time_coord_name="time_video")

    for feat in list(ds.data_vars):
        if feat not in INTERVAL_COLUMNS and feat != "confidence":
            ds[feat].attrs["type"] = "features"

    ds.attrs["trial"] = 1
    return TrialTree.from_datasets([ds])




def get_project_root(start: Path | None = None) -> Path:
    if start is not None:
        path = start.resolve()
    else:
        path = Path.cwd().resolve()
    for parent in [path] + list(path.parents):
        if (parent / "pyproject.toml").exists():
            if parent.parent.name != "deps":
                return parent
            continue
    fallback = Path(__file__).resolve()
    for parent in fallback.parents:
        if (parent / "pyproject.toml").exists():
            if parent.parent.name != "deps":
                return parent
            continue
    raise FileNotFoundError(
        f"Could not find project root starting from {path}"
    )


def downsample_trialtree(dt: TrialTree, factor: int) -> TrialTree:
    """Downsample all trials in a TrialTree using min-max envelope."""
    return dt.map_trials(lambda ds: _downsample_dataset(ds, factor))


def _minmax_envelope(values: np.ndarray, n_segments: int, factor: int) -> np.ndarray:
    shape_suffix = values.shape[1:]
    reshaped = values[:n_segments * factor].reshape(n_segments, factor, *shape_suffix)
    interleaved = np.empty((n_segments * 2, *shape_suffix), dtype=values.dtype)
    interleaved[0::2] = reshaped.min(axis=1)
    interleaved[1::2] = reshaped.max(axis=1)
    return interleaved


def _find_time_dims(ds: xr.Dataset) -> set[str]:
    """Find all time-like dimension names in a dataset."""
    return {dim for dim in ds.dims if "time" in dim.lower()}


def _downsample_along_time_dim(
    ds: xr.Dataset, time_dim: str, factor: int,
) -> tuple[dict, dict]:
    """Downsample variables along a single time dimension.

    Returns (new_coord_entry, downsampled_vars) for the given time_dim.
    """
    n_time = ds.sizes[time_dim]
    n_segments = n_time // factor
    if n_segments < 2:
        return {}, {}

    usable_len = n_segments * factor
    time_vals = ds.coords[time_dim].values[:usable_len]
    time_downsampled = time_vals[::factor][:n_segments]
    step = (time_vals[-1] - time_vals[0]) / len(time_vals) if len(time_vals) > 1 else 1.0
    half_step = step * factor / 2

    time_interleaved = np.empty(n_segments * 2)
    time_interleaved[0::2] = time_downsampled
    time_interleaved[1::2] = time_downsampled + half_step

    coord_entry = {time_dim: time_interleaved}
    data_vars = {}

    for var_name, var_data in ds.data_vars.items():
        if time_dim not in var_data.dims:
            continue
        time_axis = var_data.dims.index(time_dim)
        values = var_data.values
        if time_axis != 0:
            values = np.moveaxis(values, time_axis, 0)

        other_dims = [d for d in var_data.dims if d != time_dim]
        interleaved = _minmax_envelope(values, n_segments, factor)
        data_vars[var_name] = xr.DataArray(
            interleaved, dims=[time_dim] + other_dims, attrs=var_data.attrs
        )

    return coord_entry, data_vars


def _downsample_dataset(ds: xr.Dataset, factor: int) -> xr.Dataset:
    time_dims = _find_time_dims(ds)
    if not time_dims:
        return ds

    new_coords: dict = {}
    all_data_vars: dict = {}
    downsampled_var_names: set[str] = set()

    for time_dim in sorted(time_dims):
        coord_entry, data_vars = _downsample_along_time_dim(ds, time_dim, factor)
        new_coords.update(coord_entry)
        all_data_vars.update(data_vars)
        downsampled_var_names.update(data_vars.keys())

    for coord_name, coord_val in ds.coords.items():
        if coord_name not in new_coords:
            new_coords[coord_name] = coord_val

    for var_name, var_data in ds.data_vars.items():
        if var_name not in downsampled_var_names:
            all_data_vars[var_name] = var_data

    new_attrs = ds.attrs.copy()
    new_attrs['downsample_factor'] = factor
    new_attrs['downsample_method'] = 'minmax_envelope'
    return xr.Dataset(all_data_vars, coords=new_coords, attrs=new_attrs)


def add_changepoints_to_ds(ds, target_feature, changepoint_name, changepoint_func, **func_kwargs):
    """Compute changepoints for a feature and add to dataset.

    Parameters:
        ds: xarray Dataset
        target_feature: name of the feature variable
        changepoint_name: name of the changepoint variable
        changepoint_func: 1D changepoint detection function
        **func_kwargs: additional arguments to pass to changepoint_func

    Returns:
        xarray Dataset with added changepoints.
    """
    feature_data = ds[target_feature]
    func = partial(changepoint_func, **func_kwargs)

    time_dim = get_time_coord(feature_data).dims[0]
    changepoints = xr.apply_ufunc(
        func,
        feature_data,
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int8]
    )

    changepoints.attrs.update({
        "type": "changepoints",
        "target_feature": target_feature,
    })

    ds[f"{target_feature}_{changepoint_name}"] = changepoints
    return ds


def add_angle_rgb_to_ds(ds, smoothing_params):
    """Apply angle RGB with Gaussian smoothing across all individuals and trials.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with position data.
    smoothing_params : dict
        Parameters for Gaussian smoothing.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with added angle_rgb variable.
    """
    xy_pos = ds.position.sel(space=["x", "y"])
    time_dim = get_time_coord(xy_pos).dims[0]

    def process_angles(xy):
        _, angles = get_angle_rgb(
            xy,
            smooth_func=gaussian_filter1d,
            smoothing_params=smoothing_params
        )
        return angles

    angles = xr.apply_ufunc(
        process_angles,
        xy_pos,
        input_core_dims=[[time_dim, "space"]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    ds["angles"] = angles

    def process_rgb(xy):
        rgb, _ = get_angle_rgb(
            xy,
            smooth_func=gaussian_filter1d,
            smoothing_params=smoothing_params
        )
        return rgb

    angle_rgb = xr.apply_ufunc(
        process_rgb,
        xy_pos,
        input_core_dims=[[time_dim, "space"]],
        output_core_dims=[[time_dim, "RGB"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"RGB": 3}}
    )

    ds["angle_rgb"] = angle_rgb
    ds["angle_rgb"].attrs["type"] = "colors"

    return ds
