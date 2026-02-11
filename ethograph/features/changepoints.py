from typing import List, Literal

import numpy as np
import xarray as xr
from scipy.signal import find_peaks

from ethograph.features.preprocessing import z_normalize
from ethograph.utils.labels import find_blocks, fix_endings, purge_small_blocks, stitch_gaps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nan_boundary_indices(arr: np.ndarray) -> np.ndarray:
    """Return indices where NaN transitions occur (NaN->valid and valid->NaN)."""
    arr = np.asarray(arr)
    is_valid = ~np.isnan(arr)
    transitions = np.diff(np.concatenate(([0], is_valid.astype(int), [0])))
    nan_to_val = np.where(transitions == 1)[0]
    val_to_nan = np.where(transitions == -1)[0] - 1
    return np.concatenate((nan_to_val, val_to_nan)).astype(int)


def _to_binary(indices: np.ndarray, length: int) -> np.ndarray:
    """Convert sparse index array to dense binary mask.

    If the only marked positions are the first and last sample, returns all zeros
    (boundary-only case treated as empty).
    """
    mask = np.zeros(length, dtype=np.int8)
    if len(indices) == 0:
        return mask
    valid = indices[(indices >= 0) & (indices < length)]
    mask[valid] = 1
    if np.sum(mask) == 2 and mask[0] == 1 and mask[-1] == 1:
        mask[:] = 0
    return mask


# ---------------------------------------------------------------------------
# NaN boundary handling (refactored to use helpers)
# ---------------------------------------------------------------------------

def add_NaN_boundaries(arr, changepoints):
    """Merge NaN-transition boundaries with other changepoints -> binary mask."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("add_NaN_boundaries only supports 1D (N,) input arrays.")
    nan_bounds = _nan_boundary_indices(arr)
    merged = np.unique(np.concatenate((np.asarray(changepoints, dtype=int), nan_bounds)))
    return _to_binary(merged, len(arr))


# ---------------------------------------------------------------------------
# Binary detection (dense binary masks for apply_ufunc / dataset storage)
# ---------------------------------------------------------------------------

def find_peaks_binary(x, **kwargs):
    """scipy.signal.find_peaks + NaN boundaries -> binary mask."""
    peaks, _ = find_peaks(np.asarray(x), **kwargs)
    return add_NaN_boundaries(x, peaks)


def find_troughs_binary(x, **kwargs):
    """Find troughs (local minima) + NaN boundaries -> binary mask."""
    troughs, _ = find_peaks(-np.asarray(x), **kwargs)
    return add_NaN_boundaries(x, troughs)


def find_nearest_turning_points_binary(x, threshold=1, max_value=None, **kwargs):
    """Nearest turning points around peaks + NaN boundaries -> binary mask."""
    x = np.asarray(x, dtype=float)
    grad = np.gradient(x)
    turning_points = np.where((grad > -threshold) & (grad < threshold))[0]

    if max_value is not None:
        turning_points = turning_points[x[turning_points] < max_value]

    peaks, _ = find_peaks(x, **kwargs)
    turning_points = np.setdiff1d(turning_points, peaks)

    nearest = []
    for peak in peaks:
        left = turning_points[turning_points < peak]
        right = turning_points[turning_points > peak]
        if len(left) > 0:
            nearest.append(left[-1])
        if len(right) > 0:
            nearest.append(right[0])

    return add_NaN_boundaries(x, np.array(nearest, dtype=int))


# ---------------------------------------------------------------------------
# Changepoint time extraction
# ---------------------------------------------------------------------------

def extract_cp_times(ds: xr.Dataset, time_coord: np.ndarray, **cp_kwargs) -> np.ndarray:
    """Extract merged changepoint times from dataset.

    Replaces the inline pattern: merge_changepoints -> binary -> np.where -> times.
    Returns empty array if no CP variables exist.
    """
    filtered = ds.sel(**cp_kwargs) if cp_kwargs else ds
    cp_ds = filtered.filter_by_attrs(type="changepoints")
    if len(cp_ds.data_vars) == 0:
        return np.array([], dtype=np.float64)

    try:
        ds_merged, _ = merge_changepoints(filtered)
    except (ValueError, KeyError):
        return np.array([], dtype=np.float64)

    cp_binary = ds_merged["changepoints"].values
    cp_indices = np.where(cp_binary)[0]
    if len(cp_indices) == 0:
        return np.array([], dtype=np.float64)

    valid = cp_indices[cp_indices < len(time_coord)]
    return time_coord[valid].astype(np.float64)


def snap_to_nearest_changepoint_time(
    t_clicked: float,
    ds: xr.Dataset,
    feature_sel: str,
    time_coord: np.ndarray,
    **ds_kwargs,
) -> float:
    """Snap a clicked time (seconds) to the nearest changepoint time.

    Works entirely in the time domain — no index conversion needed.
    Also filters changepoints by target_feature matching feature_sel.
    """
    filtered = ds.sel(**ds_kwargs) if ds_kwargs else ds
    cp_ds = filtered.filter_by_attrs(type="changepoints")
    cp_ds = cp_ds.filter_by_attrs(target_feature=feature_sel)

    if len(cp_ds.data_vars) == 0:
        return t_clicked

    cp_indices = np.concatenate([
        np.where(cp_ds[var].values)[0] for var in cp_ds.data_vars
    ])
    cp_indices = np.unique(cp_indices)
    if len(cp_indices) == 0:
        return t_clicked

    valid = cp_indices[cp_indices < len(time_coord)]
    if len(valid) == 0:
        return t_clicked

    cp_times = time_coord[valid]
    nearest_idx = np.argmin(np.abs(cp_times - t_clicked))
    return float(cp_times[nearest_idx])


# ---------------------------------------------------------------------------
# Legacy index-domain snap (kept for backward compat)
# ---------------------------------------------------------------------------

def snap_to_nearest_changepoint(x_clicked, ds, feature_sel, **ds_kwargs):
    """Snap index to nearest changepoint index (legacy, index-domain)."""
    cp_ds = ds.sel(**ds_kwargs).filter_by_attrs(type="changepoints")
    cp_ds = cp_ds.filter_by_attrs(target_feature=feature_sel)

    if len(cp_ds.data_vars) == 0:
        return x_clicked

    changepoint_indices = np.concatenate([
        np.where(cp_ds[var].values)[0] for var in cp_ds.data_vars
    ])
    changepoint_indices = np.unique(changepoint_indices)

    if len(changepoint_indices) == 0:
        return x_clicked

    snapped_idx = np.argmin(np.abs(changepoint_indices - x_clicked))
    snapped_val = int(round(changepoint_indices[snapped_idx]))
    return snapped_val


# ---------------------------------------------------------------------------
# Dense correction (legacy — kept for ML pipeline)
# ---------------------------------------------------------------------------

def correct_changepoints_one_trial(labels, ds, all_params):
    """Correct labels with changepoints (dense array, legacy pipeline)."""
    cp_kwargs = all_params["cp_kwargs"]

    min_label_length = all_params.get("min_label_length")
    label_thresholds = all_params.get("label_thresholds", {})
    stitch_gap_len = all_params.get("stitch_gap_len")
    max_expansion = all_params["changepoint_params"]["max_expansion"]
    max_shrink = all_params["changepoint_params"]["max_shrink"]

    ds = ds.sel(**cp_kwargs)
    ds_merged, _ = merge_changepoints(ds)
    changepoints_binary = ds_merged["changepoints"].values

    assert changepoints_binary.ndim == 1

    changepoint_idxs = np.where(changepoints_binary)[0]
    corrected_labels = np.zeros_like(labels, dtype=np.int8)

    labels = purge_small_blocks(labels, min_label_length, label_thresholds)
    labels = stitch_gaps(labels, stitch_gap_len)

    for label in np.unique(labels):
        if label == 0:
            continue

        label_mask = labels == label
        starts, ends = find_blocks(label_mask)

        for block_start, block_end in zip(starts, ends):
            snap_start = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_start))]
            snap_end = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_end))]

            start_expansion = block_start - snap_start
            start_shrink = snap_start - block_start

            if start_expansion > max_expansion or start_shrink > max_shrink:
                snap_start = block_start

            end_expansion = snap_end - block_end
            end_shrink = block_end - snap_end

            if end_expansion > max_expansion or end_shrink > max_shrink:
                snap_end = block_end

            if snap_start > snap_end:
                snap_start = block_start
                snap_end = block_end

            if snap_end < len(corrected_labels):
                if corrected_labels[snap_end] != 0 and corrected_labels[snap_end] != label:
                    snap_end = snap_end - 1

            corrected_labels[block_start:block_end+1] = 0
            if snap_start < snap_end:
                corrected_labels[snap_start:snap_end+1] = label

    corrected_labels = purge_small_blocks(corrected_labels, min_label_length)
    corrected_labels = fix_endings(corrected_labels, changepoints_binary)

    return corrected_labels


# ---------------------------------------------------------------------------
# Merge changepoints
# ---------------------------------------------------------------------------

def merge_changepoints(ds):
    """Merge all changepoint variables into a single combined mask."""
    ds = ds.copy()
    cp_ds = ds.filter_by_attrs(type="changepoints")

    target_feature = []
    for var in cp_ds.data_vars:
        target_feature.append(cp_ds[var].attrs["target_feature"])

    if np.unique(target_feature).size > 1:
        raise ValueError(f"Not allowed to merge changepoints for different target features: {np.unique(target_feature)}")

    dims = [dim for dim in cp_ds.dims if dim not in ["trials", "time"]]

    ds["changepoints"] = (cp_ds
                                .to_array()
                                .any(dim=["variable"] + dims)
                                .astype(float))
    ds["changepoints"].attrs["type"] = "changepoints"

    ds = ds.drop_vars(list(cp_ds.data_vars))

    return ds, target_feature[0]


# ---------------------------------------------------------------------------
# ML feature engineering
# ---------------------------------------------------------------------------

def more_changepoint_features(
    changepoint_binary: np.ndarray,
    targ_feat_vals: np.ndarray,
    sigmas: List[float],
    distribution: Literal["gaussian", "laplacian"] = "laplacian",
) -> np.ndarray:
    """Create changepoint-based features from binary changepoint array."""
    features = [changepoint_binary]
    seq_length = len(changepoint_binary)
    changepoint_indices = np.where(changepoint_binary)[0]

    x = np.arange(seq_length)
    for sigma in sigmas:
        peak = np.zeros(seq_length)
        for idx in changepoint_indices:
            if distribution == "gaussian":
                peak += np.exp(-0.5 * ((x - idx) / sigma) ** 2)
            else:
                peak += np.exp(-np.abs(x - idx) / sigma)

        if peak.max() > 0:
            peak /= peak.max()
        features.append(peak)

    cp_binary_peak = np.column_stack(features)

    multiplier = np.exp(-targ_feat_vals / (np.nanmean(targ_feat_vals) + 1e-8))
    weighted_cps = cp_binary_peak * multiplier[:, np.newaxis]
    weighted_cps = np.nan_to_num(weighted_cps, nan=0.0)
    weighted_cps = z_normalize(weighted_cps)

    segment_ids = np.zeros(seq_length)
    if len(changepoint_indices) > 0:
        boundaries = np.unique(np.concatenate([[0], changepoint_indices, [seq_length]]))
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            segment_ids[start:end] = i
        if segment_ids.max() > 0:
            segment_ids /= segment_ids.max()

    return np.column_stack([cp_binary_peak, weighted_cps, segment_ids])
