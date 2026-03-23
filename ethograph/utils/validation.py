"""Validation utilities for TrialTree datasets."""

from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set

import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from ethograph.utils.trialtree import TrialTree


# ---------------------------------------------------------------------------
# Supported file extensions (single source of truth)
# ---------------------------------------------------------------------------

# Not all tested 
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".mp3", ".aac",
    ".mp4", ".avi", ".mov",
}

POSE_EXTENSIONS = {".h5", ".hdf5", ".csv", ".slp", ".nwb"}

EPHYS_EXTENSIONS = {
    ".abf", ".axgd", ".axgx", ".bdf", ".ccf", ".continuous",
    ".edr", ".edf", ".events", ".medd", ".meta", ".ncs", ".nev",
    ".nrd", ".nse", ".ns1", ".ns2", ".ns3", ".ns4", ".ns5", ".ns6",
    ".ntt", ".nvt", ".nwb", ".oebin", ".openephys", ".pl2", ".plx",
    ".rdat", ".rec", ".rhd", ".rhs", ".ridx", ".sev", ".sif", ".smr",
    ".smrx", ".spikes", ".tbk", ".tdx", ".tev", ".tin", ".tnt", ".trc",
    ".tsq", ".vhdr", ".wcp", ".xdat",
}


def _fmt_extensions(exts: set[str]) -> str:
    return ", ".join(sorted(exts))


VIDEO_EXTENSIONS_STR = _fmt_extensions(VIDEO_EXTENSIONS)
AUDIO_EXTENSIONS_STR = _fmt_extensions(AUDIO_EXTENSIONS)
POSE_EXTENSIONS_STR = _fmt_extensions(POSE_EXTENSIONS)
EPHYS_EXTENSIONS_STR = _fmt_extensions(EPHYS_EXTENSIONS)


def _qt_filter(label: str, exts: set[str]) -> str:
    globs = " ".join(f"*{e}" for e in sorted(exts))
    return f"{label} ({globs});;All files (*)"


VIDEO_FILE_FILTER = _qt_filter("Video files", VIDEO_EXTENSIONS)
AUDIO_FILE_FILTER = _qt_filter("Audio files", AUDIO_EXTENSIONS)
POSE_FILE_FILTER = _qt_filter("Pose files", POSE_EXTENSIONS)
EPHYS_FILE_FILTER = _qt_filter("Ephys files", EPHYS_EXTENSIONS)




def find_temporal_dims(ds: xr.Dataset) -> set[str]:
    """Identify dims that co-occur with any time-like dim in at least one data var."""
    temporal = set()
    time_dims = set()

    for var in ds.data_vars.values():
        var_time_dims = {d for d in var.dims if 'time' in d}
        if var_time_dims:
            time_dims.update(var_time_dims)
            temporal.update(var.dims)

    temporal -= time_dims
    return temporal


def is_integer_array(arr: np.ndarray) -> bool:
    """Check if array contains only integer values (no fractional part)."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.all(np.mod(arr, 1) == 0)
    return np.issubdtype(arr.dtype, np.integer)


def validate_required_attrs(
    ds: xr.Dataset,
    require_fps: bool = True,
) -> List[str]:
    """Validate required dataset attributes.

    Args:
        ds: Dataset to validate.
        require_fps: When False, missing fps is not an error (audio-only mode).
    """
    errors = []

    if "fps" in ds.attrs:
        if not isinstance(ds.attrs["fps"], Number) or ds.attrs["fps"] <= 0:
            errors.append("'fps' must be a positive number")
    elif require_fps:
        errors.append("Xarray dataset ('ds') must have 'fps' attribute")

    if "trial" not in ds.attrs:
        errors.append("Xarray dataset ('ds') must have 'trial' attribute")

    return errors


def validate_media_files_session(dt: "TrialTree") -> List[str]:
    """Validate session-level media file entries.

    Checks that video/audio/pose arrays in the session table contain strings.
    """
    errors = []
    if dt.session is None:
        return errors
    for var_name in ("video", "audio", "pose"):
        if var_name not in dt.session:
            continue
        vals = dt.session[var_name].values.flat
        for v in vals:
            v_str = str(v)
            if v_str and v_str not in ("", "nan"):
                continue
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


def validate_dataset(
    ds: xr.Dataset,
    type_vars_dict: Dict,
    require_fps: bool = True,
) -> List[str]:
    """Validate dataset structure and data types.

    Args:
        ds: The xarray Dataset to validate
        type_vars_dict: Dictionary containing categorized variables/coordinates
        require_fps: When False, missing fps is not an error.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Required attributes
    errors.extend(validate_required_attrs(ds, require_fps=require_fps))
    

    # Required dimensions and coordinates
    

    if "individuals" not in ds.coords or len(ds.coords["individuals"]) == 0:
        errors.append("Xarray dataset ('ds') must have 'individuals' coordinate")

    # TODO: add something about new interval based label validation?

    if "features" in type_vars_dict and len(type_vars_dict["features"]) > 0:
        for feat_name in type_vars_dict["features"]:
            if feat_name not in ds.data_vars:
                errors.append(f"Feature variable '{feat_name}' missing from trial '{ds.attrs.get('trial', '?')}'")
                continue
            feat_var = ds[feat_name]
            has_time_coord = any('time' in str(dim).lower() for dim in feat_var.dims)
            if not has_time_coord:
                errors.append(f"Feature variable '{feat_name}' must have a coordinate containing 'time'. E.g. 'time', 'time_labels', 'time_aux', etc.")

        

    # Media files are validated at the TrialTree level, not per-dataset

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
    return [ds for _, ds in dt.trial_items()]




def _possible_trial_conditions(ds: xr.Dataset, dt: "TrialTree") -> List[str]:
    """Identify possible trial condition attributes."""
    common_extensions = (
        VIDEO_EXTENSIONS
        | AUDIO_EXTENSIONS
        | POSE_EXTENSIONS
        | EPHYS_EXTENSIONS
        | {'.dat', '.bin', '.raw', '.mda'}
        | {'.csv', '.h5', '.hdf5', '.npy'}
    )

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
    type_vars_dict = {}

    if "individuals" in ds.coords:
        type_vars_dict["individuals"] = ds.coords["individuals"].values.astype(str)

    # Cameras and mics come from the session table
    if dt.cameras:
        type_vars_dict["cameras"] = np.array(dt.cameras, dtype=str)
    if dt.mics:
        type_vars_dict["mics"] = np.array(dt.mics, dtype=str)
            
    

    # Filter by type attribute
    type_filters = ['features', 'colors', 'changepoints']
    for type_name in type_filters:
        filtered = ds.filter_by_attrs(type=type_name)
        if filtered.data_vars:
            type_vars_dict[type_name] = list(filtered.data_vars)
            
    # Custom user coords/dims
    dims = find_temporal_dims(ds)
    for name in dims:
        if name in type_vars_dict:
            continue
        if name in ds.coords:
            coord = ds.coords[name]
            if coord.dtype.kind in ('U', 'S', 'O'):
                type_vars_dict[name] = coord.values.astype(str)
            elif coord.dtype.kind in ('i', 'u', 'f'):  # int, unsigned int, float
                type_vars_dict[name] = coord.values
        else:
            # Dim without coord - generate integer range
            type_vars_dict[name] = np.arange(ds.sizes[name])
            
            
    type_vars_dict["trial_conditions"] = _possible_trial_conditions(ds, dt)

    return type_vars_dict




def validate_datatree(
    dt: "TrialTree",
    require_fps: bool = True,
) -> list[str]:
    """Validate a TrialTree for consistency and data integrity.

    Performs two levels of validation:
    1. Cross-trial consistency: Ensures all trials have the same structure
       (coords, data_vars, attrs keys and optionally values)
    2. Single-dataset validation: Validates data content on first trial
       (array types, RGBA format, changepoints, etc.)

    Args:
        dt: TrialTree to validate
        require_fps: When False, missing fps is not an error.

    Returns:
        List of validation error messages (empty if valid)
    """
    ds = dt.itrial(0)
    type_vars_dict = extract_type_vars(ds, dt)
    datasets = _extract_trial_datasets(dt)

    if not datasets:
        return ["No trial datasets found in TrialTree"]

    errors = []
    errors.extend(validate_media_files_session(dt))

    sample_size = min(5, len(datasets))
    sample_indices = np.random.choice(len(datasets), size=sample_size, replace=False)
    for idx in sample_indices:
        errors.extend(validate_dataset(
            datasets[idx], type_vars_dict,
            require_fps=require_fps,
        ))

    return list(set(errors))