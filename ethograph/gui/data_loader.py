"""Data loading utilities for the ethograph GUI."""

import numpy as np
import xarray as xr
from typing import List, Optional, Tuple
from pathlib import Path
from qtpy.QtWidgets import QMessageBox
from napari import current_viewer
from ethograph.utils.io import TrialTree
from ethograph.utils.validation import validate_datatree, extract_type_vars

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


    inconsistencies, errors = validate_datatree(dt)
    
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


    # Build type_vars_dict from first trial
    type_vars_dict = extract_type_vars(ds, dt)
    

    return dt, label_dt, type_vars_dict






