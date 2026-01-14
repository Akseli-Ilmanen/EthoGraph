import numpy as np
import scipy.io as sio
from ethograph.features.changepoints import more_changepoint_features, merge_changepoints
from ethograph.utils.labels import fix_endings
import xarray as xr
from functools import partial
import itertools
from scipy.ndimage import gaussian_filter1d
from ethograph.features.mov_features import get_angle_rgb
import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, List, Set, get_args
from ethograph.utils.validation import validate_datatree
from movement.kinematics import compute_velocity, compute_speed, compute_acceleration, compute_pairwise_distances
from movement.io import load_poses
from ethograph.features.audio_features import get_synced_envelope






class TrialTree(xr.DataTree):
    """DataTree subclass with trial-specific functionality."""

    TRIAL_PREFIX: str = "trial_"

    def __init__(self, data=None, children=None, name=None):
        """Initialize TrialTree from DataTree or other arguments."""
        if isinstance(data, xr.DataTree):
            # Initialize with the root dataset from the DataTree
            super().__init__(dataset=data.ds, children=children, name=name)
            # Copy all children from the source DataTree
            for child_name, child_node in data.children.items():
                self[child_name] = child_node
        else:
            # Standard initialization (data can be a Dataset)
            super().__init__(dataset=data, children=children, name=name)

    @property
    def trials(self) -> List[int | str]:
        """Get list of trial numbers."""

        raw = [node.ds.attrs["trial"] for node in self.children.values() if node.ds is not None and "trial" in node.ds.attrs]
        trials = [val.item() if hasattr(val, 'item') else val for val in raw]
        if not trials:
            raise ValueError("No datasets with 'trial' attribute found in the tree.")
        return trials
    
    # https://docs.xarray.dev/en/stable/generated/xarray.DataTree.sel.html#xarray.DataTree.sel
    # Given xarray has native .sel, and they are quite useful! I should reorganize my own methods.
    
    
    def trial(self, trial) -> xr.Dataset:
        
        ds = self[f'{self.TRIAL_PREFIX}{trial}'].ds
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
        
        # Keep only attrs that match across all trials
        for ds in trials_dict.values():
            common = {
                k: v for k, v in common.items()
                if k in ds.attrs and ds.attrs[k] == v
            }
        return common

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
        trial_node = f'{self.TRIAL_PREFIX}{trial}'
        trial_ds = self[trial_node].ds.copy()
        
        trial_ds[new_var] = xr.full_like(trial_ds[template_var], np.nan)
            
        # Set value
        if kwargs:
            trial_ds[new_var].loc[kwargs] = new_val
        else:
            trial_ds[new_var][:] = new_val
    
        # Replace node once with all updates
        self[trial_node] = xr.DataTree(trial_ds)
    
    @classmethod
    def load(cls, path: str) -> "TrialTree":
        """Load TrialTree from a NetCDF file."""
        tree = xr.open_datatree(path)
        tree.__class__ = cls  # Convert xr.DataTree to TrialTree
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
            tree[f'{cls.TRIAL_PREFIX}{trial_num}'] = xr.DataTree(ds)
            
        
        tree._validate_tree()
        return tree
    

    @classmethod
    def from_datatree(cls, dt: xr.DataTree, attrs: dict | None = None) -> "TrialTree":
        tree = cls()
        for name, child in dt.children.items():
            tree[name] = child
        if dt.ds is not None: # handle root node
            tree.ds = dt.ds
        tree.attrs = (attrs if attrs is not None else dt.attrs).copy()
        
        
        return tree


    def _validate_tree(self) -> List[str]:
        inconsistencies, errors = validate_datatree(self)
    
        if inconsistencies or errors:
            error_msg = ""
            
            if inconsistencies:
                error_msg += "Inconsistent structure across trials:\n"
                for category, items in inconsistencies.items():
                    error_msg += f"• {category}: {items}\n"
            
            if errors:
                if error_msg:
                    error_msg += "\n"
                error_msg += "Dataset validation failed:\n"
                error_msg += "\n".join(f"• {e}" for e in errors)
            
            
            raise ValueError("TrialTree validation failed: \n" + error_msg)
    
    

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
        
    
    
    def overwrite_with_attrs(self, labels_tree: xr.DataTree) -> "TrialTree":
        """
        Overwrite attrs in this tree with that from another tree.
        """
        def merge_func(self_ds, labels_ds):
            self_ds.attrs.update(labels_ds.attrs)
            return self_ds

        result = self.map_over_datasets(merge_func, labels_tree)
        result.attrs = labels_tree.attrs.copy()
        return result
        
    
    
    
    def overwrite_with_labels(self, labels_tree: xr.DataTree) -> "TrialTree":
        """
        Overwrite labels/labels confidence and attrs in this tree with labels from another tree.
        """
        def merge_func(data_ds, labels_ds):
            if labels_ds is not None and data_ds is not None:
                result = data_ds.copy()

                for var_name in labels_ds.data_vars:
                    result[var_name] = labels_ds[var_name]

                # Copy individual dataset attributes from labels_ds
                result.attrs.update(labels_ds.attrs)

                return result
            return data_ds

        result = self.map_over_datasets(merge_func, labels_tree)

        # Copy global attributes from labels_tree (not from self)
        result.attrs = labels_tree.attrs.copy()
        return result
    
        
        
    def filter_by_attr(self, attr_name: str, attr_value: Any) -> "TrialTree":
        """Filter trials by attribute value with type conversion."""
        new_tree = xr.DataTree()
        
        def values_match(stored: Any, target: Any) -> bool:
            """Check if values match, attempting type coercion if needed."""
            if stored == target:
                return True
            
            # Try coercing both to common types
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
    
        
    
def _minimal_basics(ds):
    
    if "labels" not in ds.data_vars:
        ds["labels"] = xr.DataArray(
                np.zeros((ds.dims["time"], ds.dims["individuals"])),
                dims=["time", "individuals"],
        )
        
    for feat in list(ds.data_vars):
        if feat != "labels":
            ds[feat].attrs["type"] = "features"

    ds.attrs["trial"] = "sample_trial"
    dt = TrialTree.from_datasets([ds])
    
    return dt
   

   
def minimal_dt_from_pose(video_path, fps, tracking_path, source_software):
    """
    Create a minimal TrialTree from pose data.
    
    Args:
        video_path: Path to video file
        fps: Frames per second of the video
        tracking_path: Path to tracking file (e.g. poses.csv/poses.h5)
        source_software: Software used for tracking (e.g., 'DeepLabCut')
        
        
    Returns:
        TrialTree with minimal structure
    """
    # Validate inputs: must provide either ds OR (source_software + fps)

    ds = load_poses.from_file(
        file_path=tracking_path, 
        fps=fps, 
        source_software=source_software
    )


    ds["velocity"] = compute_velocity(ds.position)
    ds["speed"] = compute_speed(ds.position)
    ds["acceleration"] = compute_acceleration(ds.position)
    
    if len(ds.keypoints) > 1:
        compute_pairwise_distances(ds.position, dim='keypoints', pairs='all')
    
    if len(ds.individuals) > 1:
        # Not sure how this looks like with individuals > 2
        compute_pairwise_distances(ds.position, dim='individuals', pairs='all')
    


    ds = set_media_attrs(
        ds,
        cameras=[Path(video_path).name],
        tracking=[Path(tracking_path).name],
        tracking_prefix=f"{ds.attrs['source_software']}_1"
    )            
    dt = _minimal_basics(ds)


    return dt


def minimal_dt_from_ds(video_path, ds: xr.Dataset):
    
    # No public function from movement to validate that this
    # is a proper poses dataset -> add later
    
    ds = set_media_attrs(
        ds,
        cameras=[Path(video_path).name],
    )  
    dt = _minimal_basics(ds)
    
    return dt


def minimal_dt_from_audio(video_path, fps, audio_path, sr, individuals=None):

    if individuals is None:
        individuals = ["individual 1", "individual 2", "individual 3", "individual 4"]

    envelope, gen_wav_path = get_synced_envelope(audio_path, sr, fps)

    if gen_wav_path:
        audio_path = gen_wav_path
    
    n_frames = len(envelope)
    time_coords = np.arange(n_frames) / fps
    
    ds = xr.Dataset(
        data_vars={
            "envelope": ("time", envelope),
            "labels": (["time", "individuals"], np.zeros((n_frames, len(individuals))))
        },
        coords={
            "time": time_coords,
            "individuals": individuals  
        }
    )    

    ds.attrs["sr"] = sr
    ds.attrs["fps"] = fps

    
    ds = set_media_attrs(
        ds,
        cameras=[Path(video_path).name],
        mics=[Path(audio_path).name],
    )  
    
    dt = _minimal_basics(ds)
    
    return dt
 


    
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


def extract_variable_flat(
    nc_paths: List[str],
    variable: str,
    selection: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Extract and flatten a variable from multiple TrialTree files.
    
    Args:
        nc_paths: List of paths to NetCDF files
        variable: Name of the variable to extract
        selection: Optional selection dict for the variable
        
    Returns:
        Flattened 1D numpy array of the variable
    """
    arrays = []
    
    for path in nc_paths:
        dt = TrialTree.load(path)
        
        for trial_num in dt.trials:
            trial_ds = dt.trial(trial_num)
            
            data = trial_ds[variable]
            if selection:
                data = data.sel(**selection)
            arrays.append(data.squeeze().values.flatten())
    
    return np.concatenate(arrays)
   


def set_file_types_attrs(ds, cameras=None, tracking=None, mics=None):
    """Deprecated: Use set_media_attrs instead."""
    import warnings
    warnings.warn(
        "set_file_types_attrs is deprecated, use set_media_attrs instead",
        DeprecationWarning,
        stacklevel=2
    )
    if cameras is not None:
        ds.attrs["cameras"] = cameras
    if tracking is not None:
        ds.attrs["tracking"] = tracking
    if mics is not None:
        ds.attrs["mics"] = mics
    return ds

def extract_trial_info_from_filename(path):
    """
    Extract session_date, trial_num, and bird from a DLC filename.
    Expected filename format: YYYY-MM-DD_NNN_Bird_...
    """
    filename = os.path.basename(path)
    parts = filename.split('_')
    if len(parts) >= 3:
        session_date = parts[0]
        trial_num = int(parts[1])
        bird = parts[2]
        return session_date, trial_num, bird
    else:
        raise ValueError(f"Filename format not recognized: {filename}")


# TO DO, figure out smart way to specify individual in feat_kwargs, else. 
def get_feature_names(ds, all_params):
    changepoint_sigmas = all_params["changepoint_params"]["sigmas"]
    changepoint_names = []

    for var in ds.filter_by_attrs(type="changepoints").data_vars:
        changepoint_names.extend([f"{var}_binary"])
        changepoint_names.extend([f"{var}_σ={sigma}" for sigma in changepoint_sigmas])
        changepoint_names.extend([f"{var}_segIDs"])

    feat_kwargs = all_params["feat_kwargs"]
    feat_da = ds.sel(**feat_kwargs).squeeze().filter_by_attrs(type="features")

    
    feature_var_names = []
    for var_name in feat_da.data_vars:
        var_data = feat_da[var_name]
        non_time_dims = [dim for dim in var_data.dims if dim != 'time' and dim != 'trials']
        if len(non_time_dims) == 0:
            feature_var_names.append(var_name)
        elif len(non_time_dims) == 1:
            dim_name = non_time_dims[0]
            dim_coords = var_data[dim_name].values
            for coord in dim_coords:
                feature_var_names.append(f"{var_name}_{coord}")
        else:
            dim_coords_lists = [var_data[dim].values for dim in non_time_dims]
            for coord_combo in itertools.product(*dim_coords_lists):
                name_parts = [var_name] + [str(c) for c in coord_combo]
                feature_var_names.append("_".join(name_parts))
    return changepoint_names + feature_var_names



def extract_features_per_trial(ds, all_params):
    """
    Extracts and concatenates changepoint and feature data for a single trial from a dataset.
    This function selects the data corresponding to the specified trial, removes any padding.
    ----------
    ds : xarray.Dataset (single trial)
    Returns
    -------
    tuple of np.ndarray
        changepoint_feats: 2D array of shape (time, num_changepoint_features)
        features: 2D array of shape (time, num_features)
    """
    
    changepoint_sigmas = all_params["changepoint_params"]["sigmas"]
    feat_kwargs = all_params["feat_kwargs"]
    cp_kwargs = all_params["cp_kwargs"]
    good_s3d_feats = all_params["good_s3d_feats"]
    
    

    if all_params["changepoint_params"]["merge_changepoints"]:
        ds, target_feature = merge_changepoints(ds)
    

    cp_ds = ds.sel(**cp_kwargs).filter_by_attrs(type="changepoints")

    
    

    cp_list = []
    for var in cp_ds.data_vars:
        
        if not all_params["changepoint_params"]["merge_changepoints"]:
            target_feature = cp_ds[var].attrs["target_feature"]
            
        targ_feat_vals = ds[target_feature].sel(**cp_kwargs).values
        cp_data = cp_ds[var].squeeze().values
        
        output = more_changepoint_features(cp_data, sigmas=changepoint_sigmas, targ_feat_vals=targ_feat_vals)
        cp_list.append(output)
    cp_feats = np.hstack(cp_list)

    if good_s3d_feats is None: # or all_params["split_5"]["feature_ablation_condition"] == "all_s3d":
        s3d = ds.s3d.values
    else:
        s3d = ds.s3d.sel(s3d_dims=good_s3d_feats).values
    
    
    # s3d = ds.s3d.values
    ds = ds.drop_vars("s3d")
    
    feat_ds = ds.sel(**feat_kwargs).squeeze().filter_by_attrs(type="features")
    features = feat_ds.to_stacked_array('features', sample_dims=['time']).values # flatten across non-time dimensions
    shape1 = features.shape
    features = features[:, ~np.all(np.isnan(features), axis=0)]
    shape2 = features.shape
    
    # if shape1[1] != shape2[1]:
    #     print(f"\nWarning: Dropped {shape1[1]-shape2[1]} all-NaN feature columns.")

    return cp_feats, features, s3d








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





def get_session_path(user: str, datatype: str, bird: str, session: str, data_folder_type: str):
    """
    Args:
        user (str): e.g. 'Akseli_right' or 'Alice_home'.
        datatype (str): Type of data (e.g., 'rawdata' or 'derivatives').
        bird (str): Name of the bird (e.g., 'Ivy', 'Poppy', or 'Freddy').
        session (str): Date of the session in 'YYYYMMDD_XX' format.
        data_folder_type (str): 'rigid_local', 'working_local', or 'working_backup'

    Returns:
        subject_folder (str): Path to the subject folder
        session_path (str): Path to the rawdata/derivatives session folder
        data_folder (str): Path to parent data folder
    """
    breakpoint()
    # Desktop path (Windows default, swap for Linux/mac if needed)
    desktop_path = os.path.join(os.environ.get("USERPROFILE", os.environ.get("HOME")), "Desktop")
    
    # Load user_paths.json
    with open(os.path.join(desktop_path, "user_paths.json"), "r") as f:
        paths = json.load(f)
        

    # Select the data folder
    if data_folder_type == "rigid_local":
        data_folder = paths[user]["rigid_local_data_folder"]
    elif data_folder_type == "working_local":
        data_folder = paths[user]["working_local_data_folder"]
    elif data_folder_type == "working_backup":
        data_folder = paths[user]["working_backup_data_folder"]
    else:
        raise ValueError("Unknown data folder type.")

    # Bird mapping
    if bird == "Ivy":
        sub_name = "sub-01_id-Ivy"
    elif bird == "Poppy":
        sub_name = "sub-02_id-Poppy"
    elif bird == "Freddy":
        sub_name = "sub-03_id-Freddy"
    else:
        raise ValueError("Unknown bird type.")

    # Subject folder
    subject_folder = os.path.join(data_folder, datatype, sub_name)
    print(f"Subject folder: {subject_folder}")

    # Find session folder
    matches = [d for d in os.listdir(subject_folder) if session in d]

    if len(matches) != 1:
        raise RuntimeError(
            "Likely causes:\n1) Multiple or no folders found containing the session date."
            "\n2) Paths wrong in Desktop/user_paths.json."
        )

    session_path = os.path.join(subject_folder, matches[0])

    return subject_folder, session_path, data_folder



