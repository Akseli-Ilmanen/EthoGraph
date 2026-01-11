"""Data loading utilities for the ethograph GUI."""

import numpy as np
import xarray as xr
from typing import List, Optional, Tuple
from pathlib import Path
from qtpy.QtWidgets import QMessageBox
from napari import current_viewer
from ethograph.utils.io import TrialTree
from ethograph.utils.validation import validate_datatree

def show_error_dialog(message: str, title: str = ".nc File Error") -> None:
    QMessageBox.critical(current_viewer().window._qt_window, title, message)

def load_dataset(file_path: str) -> Tuple[Optional[xr.Dataset], Optional[dict]]:
    """Load dataset from file path and cache metadata on the instance.

    Returns:
        Tuple of (dt, label_dt, type_vars_dict) or (None, None, None) on error
    """
    if Path(file_path).suffix != '.nc':
        error_msg = (
            f"Unsupported file type: {Path(file_path).suffix}. Expected .nc file.\n"
            "See documentation:\n"
            "https://movement.neuroinformatics.dev/user_guide/input_output.html#native-saving-and-loading-with-netcdf"
        )
        raise ValueError(error_msg)

    dt = TrialTree.load(file_path)
    label_dt = dt.get_label_dt()
    ds = dt.isel(trials=0)

    # Build type_vars_dict from first trial
    type_vars_dict = _extract_type_vars(ds, dt)

    inconsistencies, errors = validate_datatree(dt, type_vars_dict)
    
    if inconsistencies or errors:
        error_msg = ""
        
        if inconsistencies:
            error_msg += "Inconsistent structure across trials:\n"
            for category, items in inconsistencies.items():
                error_msg += f"• {category}: {items}\n"
        
        if errors:
            if error_msg:
                error_msg += "\n"
            error_msg += "\n".join(f"• {e}" for e in errors)
            
        suffix = "\n\n See documentation: XXX"
        
        # Display twice to ensure visibility in napari
        show_error_dialog("Validation failed: \n" + error_msg + suffix)
        raise ValueError("Validation failed: \n" + error_msg + suffix)


    return dt, label_dt, type_vars_dict






def _extract_type_vars(ds: xr.Dataset, dt: TrialTree) -> dict:
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


def _possible_trial_conditions(ds: xr.Dataset, dt: TrialTree) -> List[str]:
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
