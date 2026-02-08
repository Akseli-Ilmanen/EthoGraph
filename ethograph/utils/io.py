from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from ethograph.features.mov_features import get_angle_rgb
from ethograph.utils.validation import validate_datatree




class TrialTree(xr.DataTree):
    """DataTree subclass with trial-specific functionality."""

    TRIAL_PREFIX: str = "trial_"

    @classmethod
    def trial_key(cls, trial_num: int | str) -> str:
        return f"{cls.TRIAL_PREFIX}{trial_num}"

    @classmethod
    def trial_id(cls, key: str) -> str:
        return key.removeprefix(cls.TRIAL_PREFIX)

    def __init__(self, data=None, children=None, name=None):
        """Initialize TrialTree from DataTree or other arguments."""
        if isinstance(data, xr.DataTree):
            super().__init__(dataset=data.ds, children=children, name=name)
            for child_name, child_node in data.children.items():
                self[child_name] = child_node
        else:
            super().__init__(dataset=data, children=children, name=name)

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def open(cls, path: str) -> "TrialTree":
        """Open TrialTree from a NetCDF file."""
        tree = xr.open_datatree(path)
        tree.__class__ = cls
        tree._source_path = path
        return tree

    @classmethod
    def from_datasets(cls, datasets: List[xr.Dataset]) -> "TrialTree":
        """Create from list of datasets."""
        tree = cls()
        trials = []
        for ds in datasets:
            trial_num = ds.attrs.get('trial')
            if trial_num is None:
                raise ValueError("Each dataset must have 'trial' attribute")
            if trial_num in trials:
                raise ValueError(f"Duplicate trial number: {trial_num}")
            trials.append(trial_num)
            tree[cls.trial_key(trial_num)] = xr.DataTree(ds)
        tree._validate_tree()
        return tree

    @classmethod
    def from_datatree(cls, dt: xr.DataTree, attrs: dict | None = None) -> "TrialTree":
        tree = cls()
        for name, child in dt.children.items():
            tree[name] = child
        if dt.ds is not None:
            tree.ds = dt.ds
        tree.attrs = (attrs if attrs is not None else dt.attrs).copy()
        return tree

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        import gc
        import shutil

        source_path = getattr(self, '_source_path', None)

        if path is None and source_path is None:
            raise ValueError("No path provided and no source path stored.")

        path = Path(path) if path else Path(source_path)

        self.load()
        self.close()
        gc.collect()

        if path.exists():
            temp_path = path.with_suffix('.tmp.nc')
            self.to_netcdf(temp_path, mode='w')
            try:
                path.unlink()
            except PermissionError:
                gc.collect()
                path.unlink()
            shutil.move(str(temp_path), str(path))
        else:
            self.to_netcdf(path, mode='w')

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def trials(self) -> List[int | str]:
        """Get list of trial numbers."""
        raw = [node.ds.attrs["trial"] for node in self.children.values() if node.ds is not None and "trial" in node.ds.attrs]
        trials = [val.item() if hasattr(val, 'item') else val for val in raw]
        if not trials:
            raise ValueError("No datasets with 'trial' attribute found in the tree.")
        return trials

    # -------------------------------------------------------------------------
    # Trial access
    # -------------------------------------------------------------------------

    def trial(self, trial) -> xr.Dataset:
        ds = self[self.trial_key(trial)].ds
        if ds is None:
            raise ValueError(f"Trial {trial} has no dataset")
        return ds

    def itrial(self, trial_idx) -> xr.Dataset:
        """Index select from a specific trial dataset."""
        trial_nodes = sorted(
            k for k in self.children
            if k.startswith(self.TRIAL_PREFIX)
        )
        if trial_idx >= len(trial_nodes):
            raise IndexError(f"Trial index {trial_idx} out of range")
        ds = self[trial_nodes[trial_idx]].ds
        if ds is None:
            raise ValueError(f"Trial at index {trial_idx} has no dataset")
        return ds

    def get_all_trials(self) -> Dict[int, xr.Dataset]:
        """Get all trials as a dictionary."""
        return {num: self.trial(num) for num in self.trials}

    def get_common_attrs(self) -> Dict[str, Any]:
        """Extract attributes common to all trial datasets."""
        trials_dict = self.get_all_trials()
        if not trials_dict:
            return {}
        common = dict(next(iter(trials_dict.values())).attrs)
        for ds in trials_dict.values():
            common = {
                k: v for k, v in common.items()
                if k in ds.attrs and ds.attrs[k] == v
            }
        return common

    # -------------------------------------------------------------------------
    # Data manipulation
    # -------------------------------------------------------------------------

    def new_var_like(self, trial: int, new_var: str, new_val: np.ndarray, template_var: str, kwargs: dict = None):
        """
        Update multiple variables for a trial at once.

        Args:
            trial: Trial number
            new_var: Name of new variable
            new_value: Data of new variable
            template_var: Variable name to use as template for shape and coords
            kwargs: e.g. kwargs = {'individuals': 'bird1'} [Optional]
        """
        trial_node = self.trial_key(trial)
        trial_ds = self[trial_node].ds.copy()
        trial_ds[new_var] = xr.full_like(trial_ds[template_var], np.nan)
        if kwargs:
            trial_ds[new_var].loc[kwargs] = new_val
        else:
            trial_ds[new_var][:] = new_val
        self[trial_node] = xr.DataTree(trial_ds)

    def apply_1d(
        self,
        var_name: str,
        func: Callable[[np.ndarray], np.ndarray],
        dim: str = 'time',
        **kwargs
    ) -> "TrialTree":
        """Apply a 1D function to a variable along a dimension across all trials."""
        def transform_ds(ds):
            if ds is None or var_name not in ds:
                return ds
            var_attrs = ds[var_name].attrs.copy()
            transformed = xr.apply_ufunc(
                func,
                ds[var_name],
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                kwargs=kwargs,
                vectorize=True,
                dask='parallelized'
            )
            transformed.attrs = var_attrs
            result = ds.copy()
            result[var_name] = transformed
            return result
        return self.from_datatree(self.map_over_datasets(transform_ds), attrs=self.attrs)

    def apply_2d(
        self,
        var1_name: str,
        var2_name: str,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        output_name: str | None = None,
        dim: str = 'time',
        **kwargs
    ) -> "TrialTree":
        """Apply a function taking two arrays to variables across all trials."""
        def transform_ds(ds):
            if ds is None or var1_name not in ds or var2_name not in ds:
                return ds
            var1_attrs = ds[var1_name].attrs.copy()
            transformed = xr.apply_ufunc(
                func,
                ds[var1_name],
                ds[var2_name],
                input_core_dims=[[dim], [dim]],
                output_core_dims=[[dim]],
                kwargs=kwargs,
                vectorize=True,
                dask='parallelized'
            )
            transformed.attrs = var1_attrs
            result = ds.copy()
            result[output_name or var1_name] = transformed
            return result
        return self.from_datatree(self.map_over_datasets(transform_ds), attrs=self.attrs)

    # -------------------------------------------------------------------------
    # Label operations
    # -------------------------------------------------------------------------

    def get_label_dt(self, empty: bool = False) -> "TrialTree":
        def filter_node(ds):
            if ds is None or "labels" not in ds.data_vars:
                return xr.Dataset()
            vars_to_extract = ["labels"]
            if "labels_confidence" in ds.data_vars:
                vars_to_extract.append("labels_confidence")
            filtered = ds[vars_to_extract]
            return xr.zeros_like(filtered) if empty else filtered
        return self.from_datatree(self.map_over_datasets(filter_node), attrs=self.attrs)

    def overwrite_with_attrs(self, labels_tree: xr.DataTree) -> "TrialTree":
        """Overwrite attrs in this tree with that from another tree."""
        def merge_func(self_ds, labels_ds):
            self_ds.attrs.update(labels_ds.attrs)
            return self_ds
        tree = self.map_over_datasets(merge_func, labels_tree)
        tree.attrs = labels_tree.attrs.copy()
        return TrialTree(tree)

    def overwrite_with_labels(self, labels_tree: xr.DataTree) -> "TrialTree":
        """Overwrite labels/labels confidence and attrs in this tree with labels from another tree."""
        def merge_func(data_ds, labels_ds):
            if labels_ds is not None and data_ds is not None:
                tree = data_ds.copy()
                for var_name in labels_ds.data_vars:
                    tree[var_name] = labels_ds[var_name]
                tree.attrs.update(labels_ds.attrs)
                return tree
            return data_ds
        tree = self.map_over_datasets(merge_func, labels_tree)
        tree.attrs = labels_tree.attrs.copy()
        return TrialTree(tree)

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    def filter_by_attr(self, attr_name: str, attr_value: Any) -> "TrialTree":
        """Filter trials by attribute value with type conversion."""
        new_tree = xr.DataTree()

        def values_match(stored: Any, target: Any) -> bool:
            if stored == target:
                return True
            for coerce in (str, int, float):
                try:
                    return coerce(stored) == coerce(target)
                except (ValueError, TypeError):
                    continue
            return False

        for name, node in self.children.items():
            if node.ds and attr_name in node.ds.attrs:
                if values_match(node.ds.attrs[attr_name], attr_value):
                    new_tree[name] = node
        return TrialTree(new_tree)

    # -------------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------------

    def _validate_tree(self) -> List[str]:
        errors = validate_datatree(self)
        if errors:
            error_msg = "Dataset validation failed:\n"
            error_msg += "\n".join(f"• {e}" for e in errors)
            raise ValueError("TrialTree validation failed:\n" + error_msg)
    
        
    

def set_media_attrs(
    ds: xr.Dataset,
    cameras: Optional[List[str]] = None,
    mics: Optional[List[str]] = None,
    tracking: Optional[List[str]] = None,
    tracking_prefix: str = "dlc",
) -> xr.Dataset:
    """
    Set media file attributes with consistent keys.

    Creates both the file type list (e.g., ds.attrs["cameras"] = ["cam1", "cam2"])
    and individual file path attrs (e.g., ds.attrs["cam1"] = "video.mp4").

    Args:
        ds: xarray Dataset to modify
        cameras: List of camera file paths, keys auto-generated as cam1, cam2, ...
        mics: List of microphone file paths, keys auto-generated as mic1, mic2, ...
        tracking: List of tracking file paths, keys use tracking_prefix
        tracking_prefix: Prefix for tracking keys (default "dlc", could be "sleap", "anipose")

    Returns:
        Modified dataset with file attributes set

    Example:
        ds = set_media_attrs(
            ds,
            cameras=["video-cam-1.mp4", "video-cam-2.mp4"],
            tracking=["dlc-cam-1.csv", "dlc-cam-2.csv"],
        )
        # Result:
        # ds.attrs["cameras"] = ["cam1", "cam2"]
        # ds.attrs["cam1"] = "video-cam-1.mp4"
        # ds.attrs["cam2"] = "video-cam-2.mp4"
        # ds.attrs["tracking"] = ["dlc1", "dlc2"]
        # ds.attrs["dlc1"] = "dlc-cam-1.csv"
    """
    file_configs = [
        ("cameras", cameras, "cam"),
        ("mics", mics, "mic"),
        ("tracking", tracking, tracking_prefix),
    ]

    for file_type, files, prefix in file_configs:
        if files is None:
            continue

        keys = [f"{prefix}{i+1}" for i in range(len(files))]

        ds.attrs[file_type] = keys
        for key, filepath in zip(keys, files):
            ds.attrs[key] = filepath

    return ds





def downsample_trialtree(dt: "TrialTree", factor: int) -> "TrialTree":
    """Downsample all trials in a TrialTree using min-max envelope.

    Args:
        dt: TrialTree to downsample
        factor: Downsample factor

    Returns:
        New TrialTree with downsampled data
    """
    return TrialTree.from_datatree(
        dt.map_over_datasets(lambda ds: _downsample_dataset(ds, factor) if ds is not None else ds),
        attrs=dt.attrs
    )

def _downsample_dataset(ds: xr.Dataset, factor: int) -> xr.Dataset:
    """Downsample a dataset along the time dimension using min-max envelope.

    Args:
        ds: xarray Dataset with 'time' dimension
        factor: Downsample factor (e.g., 100 = keep ~2% of samples as min/max pairs)

    Returns:
        Downsampled dataset preserving peaks via min-max envelope
    """
    if 'time' not in ds.dims:
        return ds

    n_time = ds.sizes['time']
    n_segments = n_time // factor
    if n_segments < 2:
        return ds

    usable_len = n_segments * factor
    time_vals = ds.time.values[:usable_len]

    time_downsampled = time_vals[::factor][:n_segments]
    dt = (time_vals[-1] - time_vals[0]) / len(time_vals) if len(time_vals) > 1 else 1.0
    half_step = dt * factor / 2

    time_interleaved = np.empty(n_segments * 2)
    time_interleaved[0::2] = time_downsampled
    time_interleaved[1::2] = time_downsampled + half_step

    new_coords = {'time': time_interleaved}
    for coord_name, coord_val in ds.coords.items():
        if coord_name != 'time':
            new_coords[coord_name] = coord_val

    data_vars = {}
    for var_name, var_data in ds.data_vars.items():
        if 'time' not in var_data.dims:
            data_vars[var_name] = var_data
            continue

        var_attrs = var_data.attrs.copy()
        values = var_data.values[:usable_len] if var_data.dims[0] == 'time' else var_data.values

        time_axis = var_data.dims.index('time')
        other_dims = [d for d in var_data.dims if d != 'time']
        new_dims = ['time'] + other_dims

        if time_axis == 0:
            shape_suffix = values.shape[1:] if len(values.shape) > 1 else ()
            reshaped = values.reshape(n_segments, factor, *shape_suffix)
            mins = reshaped.min(axis=1)
            maxs = reshaped.max(axis=1)
        else:
            values_t = np.moveaxis(values, time_axis, 0)[:usable_len]
            shape_suffix = values_t.shape[1:] if len(values_t.shape) > 1 else ()
            reshaped = values_t.reshape(n_segments, factor, *shape_suffix)
            mins = reshaped.min(axis=1)
            maxs = reshaped.max(axis=1)

        interleaved_shape = (n_segments * 2,) + shape_suffix
        interleaved = np.empty(interleaved_shape, dtype=values.dtype)
        interleaved[0::2] = mins
        interleaved[1::2] = maxs

        data_vars[var_name] = xr.DataArray(interleaved, dims=new_dims, attrs=var_attrs)

    new_attrs = ds.attrs.copy()
    if 'audio_sr' in new_attrs:
        new_attrs['original_audio_sr'] = new_attrs['audio_sr']
        new_attrs['audio_sr'] = new_attrs['audio_sr'] / factor * 2
    new_attrs['downsample_factor'] = factor
    new_attrs['downsample_method'] = 'minmax_envelope'

    return xr.Dataset(data_vars, coords=new_coords, attrs=new_attrs)



def add_changepoints_to_ds(ds, target_feature, changepoint_name, changepoint_func, **func_kwargs):
    """
    Generalized function to compute changepoints for any feature across selected dimensions and add to ds.

    Parameters:
    - ds: xarray Dataset
    - target_feature: name of the feature variable
    - changepoint_name: name of the changepoint variable
    - changepoint_func: 1D changepoint detection function
    - kwargs: additional arguments to pass to changepoint_func
   
    Returns:
    - xarray DataSet with added changepoints.
    """

    feature_data = ds[target_feature]
    func = partial(changepoint_func, **func_kwargs)
    
    # Core dimension = time. 
    # This dimension is not broadcast during the operation.
    changepoints = xr.apply_ufunc(
        func,
        feature_data,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
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
    """
    Apply angle RGB with Gaussian smoothing across all individuals and trials.
    
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
    # Select x-y data
    xy_pos = ds.position.sel(space=["x", "y"])


    def process_angles(xy):
        
        # Get angle RGB with smoothing
        _, angles = get_angle_rgb(
            xy,
            smooth_func=gaussian_filter1d,
            smoothing_params=smoothing_params
        )
        return angles

    # Apply transformation
    angles = xr.apply_ufunc(
        process_angles,
        xy_pos,
        input_core_dims=[["time", "space"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    ds["angles"] = angles


    def process_rgb(xy):
        
        # Get angle RGB with smoothing
        rgb, _ = get_angle_rgb(
            xy,
            smooth_func=gaussian_filter1d,
            smoothing_params=smoothing_params
        )
        return rgb


    # Apply transformation
    angle_rgb = xr.apply_ufunc(
        process_rgb,
        xy_pos,
        input_core_dims=[["time", "space"]],
        output_core_dims=[["time", "RGB"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"RGB": 3}}
    )

    
    ds["angle_rgb"] = angle_rgb
    ds["angle_rgb"].attrs["type"] = "colors"
    

    return ds








