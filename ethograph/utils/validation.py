"""Validation utilities for TrialTree datasets."""


import numpy as np
import xarray as xr
from typing import Dict, List, Set, TYPE_CHECKING
from numbers import Number
from pathlib import Path



if TYPE_CHECKING:
    from ethograph.utils.io import TrialTree


TRIAL_PREFIX = "trial_"


def is_integer_array(arr: np.ndarray) -> bool:
    """Check if array contains only integer values (no fractional part)."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.all(np.mod(arr, 1) == 0)
    return np.issubdtype(arr.dtype, np.integer)


def validate_required_attrs(ds: xr.Dataset) -> List[str]:
    """Validate required dataset attributes."""
    errors = []

    if "fps" not in ds.attrs:
        errors.append("Xarray dataset ('ds') must have 'fps' attribute")
    elif not isinstance(ds.attrs["fps"], Number) or ds.attrs["fps"] <= 0:
        errors.append("'fps' must be a positive number")

    if "trial" not in ds.attrs:
        errors.append("Xarray dataset ('ds') must have 'trial' attribute")
        
    if "cameras" not in ds.attrs:
        errors.append("Xarray dataset ('ds') must have 'cameras' attribute")

    return errors


def validate_media_files(ds: xr.Dataset, file_type: str) -> List[str]:
    """Validate media file attributes consistency.

    Checks that each key in the file type list has a corresponding file path attribute.
    E.g., if ds.attrs["cameras"] = ["cam1", "cam2"], then ds.attrs["cam1"] and
    ds.attrs["cam2"] must exist.
    """
    errors = []

    keys = ds.attrs.get(file_type)
    if keys is None:
        return errors

    keys = np.atleast_1d(keys)
    # for key in keys:
    #     if key not in ds.attrs:
    #         errors.append(
    #             f"Missing file path for '{file_type}': ds.attrs['{key}'] not found. "
    #             f"Use set_media_attrs() to set both key list and file paths together."
    #         )

    return errors


def validate_changepoints(ds: xr.Dataset) -> List[str]:
    """Validate changepoint variables."""
    errors = []
    cp_ds = ds.filter_by_attrs(type='changepoints')

    for var_name, var in cp_ds.data_vars.items():
        arr = var.values

        if not is_integer_array(arr):
            errors.append(
                f"Changepoint '{var_name}' must contain only integer values"
            )

        if arr.min() < 0 or arr.max() > 1:
            errors.append(
                f"Changepoint '{var_name}' must have values in range [0, 1]"
            )

        target = var.attrs.get("target_feature")
        if target and target not in ds.data_vars:
            errors.append(
                f"Changepoint '{var_name}' references non-existent target_feature '{target}'"
            )

    return errors


def validate_colors(ds: xr.Dataset) -> List[str]:
    """Validate color variables."""
    errors = []
    color_ds = ds.filter_by_attrs(type='colors')

    for var_name, data_array in color_ds.data_vars.items():
        if 'RGB' not in data_array.dims:
            errors.append(f"Color variable '{var_name}' must have 'RGB' dimension")
            continue

        flat = data_array.transpose(..., 'RGB').values.reshape(-1, 3)

        is_valid_rgb = (
            flat.shape[1] == 3 and
            ((0 <= flat.min() <= flat.max() <= 1) or
            (0 <= flat.min() <= flat.max() <= 255))
        )
        if not is_valid_rgb:
            errors.append(
                f"Color variable '{var_name}' must have RGB values in [0,1] or [0,255]"
            )

    return errors


def validate_dataset(ds: xr.Dataset, type_vars_dict: Dict) -> List[str]:
    """Validate dataset structure and data types.

    Args:
        ds: The xarray Dataset to validate
        type_vars_dict: Dictionary containing categorized variables/coordinates

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Required attributes
    errors.extend(validate_required_attrs(ds))
    

    # Required dimensions and coordinates
    if "time" not in ds.dims or len(ds.coords["time"]) == 0:
        errors.append("Xarray dataset ('ds') must have 'time' dimension")

    if "individuals" not in ds.coords or len(ds.coords["individuals"]) == 0:
        errors.append("Xarray dataset ('ds') must have 'individuals' coordinate")

    # Required variables
    if "labels" not in ds.data_vars:
        errors.append("Xarray dataset ('ds') must contain 'labels' variable")
    elif not is_integer_array(ds['labels'].values):
        errors.append("Variable 'labels' must contain integer values")

    if "features" not in type_vars_dict or len(type_vars_dict["features"]) == 0:
        errors.append("Xarray dataset ('ds') must contain at least one variable with attribute type='features'")

        
    for cam in type_vars_dict["cameras"]:
        if cam not in ds.attrs:
            errors.append(f"Xarray dataset ('ds') missing file name attribute for camera '{cam}'")

    # Audio requires sample rate
    if "mics" in type_vars_dict and "audio_sr" not in ds.attrs:
        errors.append("Xarray dataset ('ds') with 'mics' must have 'audio_sr' (sample rate) attribute")
    
    if "mics" in type_vars_dict:
        for mic in type_vars_dict["mics"]:
            if mic not in ds.attrs:
                errors.append(f"Xarray dataset ('ds') missing file name attribute for mic '{mic}'")
                
    if "tracking" in type_vars_dict:
        # e.g. dlc1, dlc2 for cam1, cam2
        for track in type_vars_dict["tracking"]:
            if track not in ds.attrs:
                errors.append(f"Xarray dataset ('ds') missing file name attribute for pose data: '{track}'")

    # Media file consistency
    for file_type in ["cameras", "mics", "tracking"]:
        errors.extend(validate_media_files(ds, file_type))

    # Feature variables must be arrays
    feat_ds = ds.filter_by_attrs(type='features')
    for var_name, var in feat_ds.data_vars.items():
        if not isinstance(var.values, np.ndarray):
            errors.append(f"Feature '{var_name}' must be an array")

    # Changepoints validation
    if "changepoints" in type_vars_dict:
        errors.extend(validate_changepoints(ds))

    # Colors validation
    if "colors" in type_vars_dict:
        errors.extend(validate_colors(ds))

    return errors


def _extract_trial_datasets(dt: "TrialTree") -> List[xr.Dataset]:
    """Extract all trial datasets from a TrialTree."""
    datasets = []
    for name, node in dt.children.items():
        if name.startswith(TRIAL_PREFIX) and node.ds is not None:
            datasets.append(node.ds)
    return datasets


# Removed temporarily, probably not needed long-term


# def _check_cross_trial_consistency(
#     datasets: List[xr.Dataset],
# ) -> Dict[str, Set[str]]:
#     """Check that coords, data_vars, and attrs keys are consistent across trials.

#     Returns:
#         Dict with 'coords', 'data_vars', 'attrs' keys mapping to sets of
#         inconsistent items (empty dict if all consistent).
#     """
#     if len(datasets) < 2:
#         return {}

#     all_coords = [set(ds.coords.keys()) for ds in datasets]
#     all_vars = [set(ds.data_vars.keys()) for ds in datasets]
#     all_attrs = [set(ds.attrs.keys()) for ds in datasets]

#     inconsistencies = {}

#     union_coords = set.union(*all_coords)
#     intersect_coords = set.intersection(*all_coords)
#     if diff := (union_coords - intersect_coords):
#         inconsistencies["coords"] = diff

#     union_vars = set.union(*all_vars)
#     intersect_vars = set.intersection(*all_vars)
#     if diff := (union_vars - intersect_vars):
#         inconsistencies["data_vars"] = diff

#     union_attrs = set.union(*all_attrs)
#     intersect_attrs = set.intersection(*all_attrs)
#     if diff := (union_attrs - intersect_attrs):
#         diff.discard("human_verified")
#         if diff:
#             inconsistencies["attrs"] = diff





    return inconsistencies


def _possible_trial_conditions(ds: xr.Dataset, dt: "TrialTree") -> List[str]:
    """Identify possible trial condition attributes."""
    common_extensions = {
        '.csv', '.mp4', '.avi', '.mov', '.h5', '.hdf5',
        '.wav', '.mp3', '.npy',
    }

    common_attrs = dt.get_common_attrs().keys()

    cond_attrs = []
    for key, value in ds.attrs.items():
        if key in ['trial'] or key in common_attrs:
            continue

        if isinstance(value, str):
            if Path(value).suffix.lower() in common_extensions:
                continue

        cond_attrs.append(key)

    return cond_attrs



    
def extract_type_vars(ds: xr.Dataset, dt: "TrialTree") -> dict:
    """Extract type variables dictionary from dataset."""
    type_vars_dict = {}

    type_vars_dict['individuals'] = ds.coords['individuals'].values.astype(str)

    feat_ds = ds.filter_by_attrs(type='features')
    type_vars_dict['features'] = list(feat_ds.data_vars)

    type_vars_dict['cameras'] = np.atleast_1d(ds.attrs.get('cameras')).astype(str)

    # Optional attributes
    mics = ds.attrs.get('mics')
    if mics:
        type_vars_dict['mics'] = np.atleast_1d(mics).astype(str)

    tracking = ds.attrs.get('tracking')
    if tracking:
        type_vars_dict['tracking'] = np.atleast_1d(tracking).astype(str)

    if 'keypoints' in ds.coords:
        type_vars_dict['keypoints'] = ds.coords['keypoints'].values.astype(str)

    color_ds = ds.filter_by_attrs(type='colors')
    if color_ds.data_vars:
        type_vars_dict['colors'] = list(color_ds.data_vars)

    cp_ds = ds.filter_by_attrs(type='changepoints')
    if cp_ds.data_vars:
        type_vars_dict['changepoints'] = list(cp_ds.data_vars)

    type_vars_dict["trial_conditions"] = _possible_trial_conditions(ds, dt)
    

    return type_vars_dict




def validate_datatree(
    dt: "TrialTree"
) -> tuple[Dict[str, Set[str]], List[str]]:
    """Validate a TrialTree for consistency and data integrity.

    Performs two levels of validation:
    1. Cross-trial consistency: Ensures all trials have the same structure
       (coords, data_vars, attrs keys and optionally values)
    2. Single-dataset validation: Validates data content on first trial
       (array types, RGBA format, changepoints, etc.)

    Args:
        dt: TrialTree to validate
    Returns:
        Tuple of:
        - inconsistencies: Dict mapping category to set of inconsistent items
        - errors: List of validation error messages
    """
    ds = dt.itrial(0) # sample
    type_vars_dict = extract_type_vars(ds, dt)
    
    print("Extracted type_vars_dict:", type_vars_dict)
            
    datasets = _extract_trial_datasets(dt)

    if not datasets:
        return {}, ["No trial datasets found in TrialTree"]


    errors = []
    # Validate data content on random sample of trials
    sample_size = min(5, len(datasets))
    sample_indices = np.random.choice(len(datasets), size=sample_size, replace=False)
    for idx in sample_indices:
        errors.extend(validate_dataset(datasets[idx], type_vars_dict))
        
    return list(set(errors))