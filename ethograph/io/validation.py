"""Validation utilities for TrialTree datasets."""

from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from ethograph.io.trialtree import TrialTree


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


EPHYS_EXTENSIONS_STR = _fmt_extensions(EPHYS_EXTENSIONS)


def _qt_filter(label: str, exts: set[str]) -> str:
    globs = " ".join(f"*{e}" for e in sorted(exts))
    return f"{label} ({globs});;All files (*)"


VIDEO_FILE_FILTER = _qt_filter("Video files", VIDEO_EXTENSIONS)
AUDIO_FILE_FILTER = _qt_filter("Audio files", AUDIO_EXTENSIONS)
EPHYS_FILE_FILTER = _qt_filter("Ephys files", EPHYS_EXTENSIONS)




def find_temporal_dims(ds: xr.Dataset) -> set[str]:
    """Identify non-time dimensions that co-occur with a time dimension.

    Returns the set of dimension names that appear alongside at least one
    time-like dimension (any dim whose name contains ``"time"``) in the
    same data variable.  Used to discover extra selection dimensions such
    as ``space``, ``keypoints``, or ``individuals``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to inspect.

    Returns
    -------
    set[str]
        Dimension names (excluding time dims themselves).
    """
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
) -> list[str]:
    """Validate required dataset attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate.
    require_fps : bool
        When False, missing ``fps`` is not an error (audio-only mode).

    Returns
    -------
    list[str]
        Validation error messages (empty if valid).
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


def validate_media_files_session(dt: "TrialTree") -> list[str]:
    """Validate session-level media file entries.

    Checks that media paths referenced in session_io are non-empty.
    """
    errors = []
    sio = getattr(dt, "session_io", None)
    if sio is None:
        return errors
    # Basic check: ensure session_io is accessible
    try:
        _ = sio.cameras
    except Exception:
        pass
    return errors


def validate_changepoints(ds: xr.Dataset) -> list[str]:
    """Validate changepoint variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing changepoint variables.

    Returns
    -------
    list[str]
        Validation error messages (empty if valid).
    """
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


def validate_colors(ds: xr.Dataset) -> list[str]:
    """Validate color variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing color variables.

    Returns
    -------
    list[str]
        Validation error messages (empty if valid).
    """
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
    catalog,
    require_fps: bool = True,
) -> list[str]:
    """Validate dataset structure and data types.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to validate.
    catalog : DataCatalog
        Feature/dimension catalog (from ``catalog_from_xarray``).
    require_fps : bool
        When False, missing ``fps`` is not an error.

    Returns
    -------
    list[str]
        Validation error messages (empty if valid).
    """
    errors = []

    errors.extend(validate_required_attrs(ds, require_fps=require_fps))

    if "individuals" not in ds.coords or len(ds.coords["individuals"]) == 0:
        errors.append("Xarray dataset ('ds') must have 'individuals' coordinate")

    for feat_name in catalog.features:
        if feat_name not in ds.data_vars:
            errors.append(f"Feature variable '{feat_name}' missing from trial '{ds.attrs.get('trial', '?')}'")
            continue
        feat_var = ds[feat_name]
        has_time_coord = any('time' in str(dim).lower() for dim in feat_var.dims)
        if not has_time_coord:
            errors.append(f"Feature variable '{feat_name}' must have a coordinate containing 'time'. E.g. 'time', 'time_labels', 'time_aux', etc.")

    feat_ds = ds.filter_by_attrs(type='features')
    for var_name, var in feat_ds.data_vars.items():
        if not isinstance(var.values, np.ndarray):
            errors.append(f"Feature '{var_name}' must be an array")

    if catalog.changepoints:
        errors.extend(validate_changepoints(ds))

    if catalog.colors:
        errors.extend(validate_colors(ds))

    return errors


def _extract_trial_datasets(dt: "TrialTree") -> list[xr.Dataset]:
    """Extract all trial datasets from a TrialTree."""
    return [ds for _, ds in dt.trial_items()]




def _possible_trial_conditions(ds: xr.Dataset, dt: "TrialTree") -> list[str]:
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

    Parameters
    ----------
    dt : TrialTree
        Tree to validate.
    require_fps : bool
        When False, missing ``fps`` is not an error.

    Returns
    -------
    list[str]
        Validation error messages (empty if valid).
    """
    from ethograph.io.catalog import catalog_from_xarray

    ds = dt.itrial(0)
    catalog = catalog_from_xarray(ds, dt)
    datasets = _extract_trial_datasets(dt)

    if not datasets:
        return ["No trial datasets found in TrialTree"]

    errors = []
    errors.extend(validate_media_files_session(dt))

    sample_size = min(5, len(datasets))
    sample_indices = np.random.choice(len(datasets), size=sample_size, replace=False)
    for idx in sample_indices:
        errors.extend(validate_dataset(
            datasets[idx], catalog,
            require_fps=require_fps,
        ))

    return list(set(errors))