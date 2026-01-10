"""Data loading utilities for the moveseg GUI."""


import xarray as xr
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from napari.utils.notifications import show_error
from moveseg.utils.io import TrialTree



# rewrite this code by simplfying,
# use filter attrs xarray fucnction
# and place lots of statements like these to throw error if something doesn't occur:
        # assert(feature.shape[1] == event_seq_raw.shape[0])
        # assert(feature.shape[1] == boundary_seq_raw.shape[0])


def load_dataset(file_path: str) -> Tuple[Optional[xr.Dataset], Optional[dict]]:
    """Load dataset from file path and cache metadata on the instance.

    Returns:
        Tuple of (dataset, info_dict)
    """

    # Check file extension
    if Path(file_path).suffix != '.nc':
        error_msg = (
            f"Unsupported file type: {Path(file_path).suffix}. Expected .nc file.\n"
            "See documentation:\n"
            "https://movement.neuroinformatics.dev/user_guide/input_output.html#native-saving-and-loading-with-netcdf"
        )
        show_error(error_msg)
        return None, None

    dt = TrialTree.load(file_path)

    

    label_dt = dt.get_label_dt()
    
    
    
    ds = dt.isel(trials=0)


    # Initialize info dictionary to categorize variables by type
    type_vars_dict = {}

    # We validated that same coords, data_vars, and atttr keys across trials. 
    # So we just use first.
    


    type_vars_dict['individuals'] = ds.coords['individuals'].values.astype(str)


    feat_ds = ds.filter_by_attrs(type='features')
    type_vars_dict['features'] = list(feat_ds.data_vars) 

    type_vars_dict['cameras'] = np.atleast_1d(ds.attrs.get('cameras')).astype(str)



    # Optional
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


    type_vars_dict["trial_conditions"] = possible_trial_conditions(dt)


    if not validate_dataset(ds, type_vars_dict):
        show_error("Dataset validation failed - see error messages above")
        raise ValueError("Stopping...")

    return dt, label_dt, type_vars_dict



def possible_trial_conditions(dt: xr.DataTree) -> List[str]:
    """
    Identify possible trial conditions.
    """
    ds = dt.isel(trials=0)
    
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


def validate_dataset(ds: xr.Dataset, type_vars_dict: dict) -> bool:
    """Validate data types in the dataset according to specifications.
    
    Args:
        ds: The xarray Dataset to validate
        info: Dictionary containing categorized variables/coordinates
        
    Returns:
        True if all validations pass, False otherwise
    """
    validation_errors = []
    
    
    if "features" not in type_vars_dict:
        validation_errors.append(f"Dataset must contain at least one variable with type 'features'")

    if "cameras" not in type_vars_dict:
        validation_errors.append(f"Dataset must contain at least one attribute with file_type 'cameras'")

    if "labels" not in ds.data_vars:
        validation_errors.append("Dataset must contain 'labels' variable")
          
    if "mics" in type_vars_dict and not hasattr(ds, "sr"):
        validation_errors.append("Dataset must have 'sr' (for sampling rate) attribute for microphone processing.")



    labels = ds['labels'].values 
    
    def is_integer_labels(arr: np.ndarray) -> bool:
        """Check if the array contains only integer values (no fractional part)."""
        if np.issubdtype(arr.dtype, np.floating):
            return np.all(np.mod(arr, 1) == 0)
        return np.issubdtype(arr.dtype, np.integer)

    if not is_integer_labels(labels):
        validation_errors.append("Variable 'labels' must contain integer values (no fractional part)")
    
    
    
    feat_ds = ds.filter_by_attrs(type='features')
    for var_name, var in feat_ds.data_vars.items():
        if not isinstance(var.values, np.ndarray):
            validation_errors.append(f"Variable '{var.name}' with type 'features' must be an array")
    
    
    if "changepoints" in type_vars_dict:
        cp_ds = ds.filter_by_attrs(type='changepoints')

        for var_name, var in cp_ds.data_vars.items():
            arr = var.values

            if not is_integer_labels(arr):
                validation_errors.append(
                    f"Data variable '{var.name}' with type 'changepoints' must contain only integer values"
                )

            if arr.min() < 0 or arr.max() > 1:
                validation_errors.append(
                    f"Data variable '{var.name}' with type 'changepoints' must have values in range [0, 1]"
                )
 
    
    if 'colors' in type_vars_dict:
        cp_ds = ds.filter_by_attrs(type='colors')

        for var_name, data_array in cp_ds.data_vars.items():
            flat = data_array.transpose(..., 'RGB').values.reshape(-1, 3)
            
            is_valid_rgb = (
                flat.shape[1] == 3 and
                ((0 <= flat.min() <= flat.max() <= 1) or 
                (0 <= flat.min() <= flat.max() <= 255))
            )
            if not is_valid_rgb:
                validation_errors.append(
                    f"Data variable '{var_name}' with type 'colors' must be an array of RGB values with shape (..., 3) "
                    "and values in range [0, 1] or [0, 255]"
                )

  

    # Report validation errors
    if validation_errors:
        error_msg = "Dataset validation failed:\n" + "\n".join(f"• {error}" for error in validation_errors)
        show_error(error_msg)
        return False
    
    return True

