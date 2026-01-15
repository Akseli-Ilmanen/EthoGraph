import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Sequence
from scipy.signal import find_peaks
import xarray as xr
from ethograph.utils.labels import stitch_gaps, fix_endings, remove_small_blocks, find_blocks, purge_small_motifs
import os

from typing import List, Literal
from ethograph.features.preprocessing import z_normalize, interpolate_nans
from pathlib import Path



def add_NaN_boundaries(arr, changepoints):
    """
    Finds the boundaries where NaN values transition to valid values and vice versa.

    Inputs:
        arr - 1D array of values (may contain NaN values)
        changepoints - other type of changepoint (e.g., peaks, troughs).

    Outputs:
        Binary mask of changepoint locations.
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("add_NaN_boundaries only supports 1D (N,) input arrays.")

    is_valid = ~np.isnan(arr)
    transitions = np.diff(np.concatenate(([0], is_valid.astype(int), [0])))

    nan_to_val_idx = np.where(transitions == 1)[0]      # (NaN → value) - rising edge
    val_to_nan_idx = np.where(transitions == -1)[0] - 1 # (value → NaN) - falling edge

    NaN_boundaries = np.concatenate((nan_to_val_idx, val_to_nan_idx)).astype(int)
    changepoints = np.unique(np.concatenate((changepoints, NaN_boundaries)))
    
    # Binarize
    mask = np.zeros_like(arr, dtype=np.int8)
    mask[changepoints] = 1
    
    # If mask is all zeros, except 1 at beginning and end, set to all zeros
    if np.sum(mask) == 2 and mask[0] == 1 and mask[-1] == 1:
        mask[:] = 0
        
    return mask


def find_peaks_binary(x, **kwargs):
    """
    scipy.signal.find_peaks + NaN boundaries -> binary mask
    """
    peaks, _ = find_peaks(np.asarray(x), **kwargs)
    
    changepoints_mask = add_NaN_boundaries(x, peaks)

    return changepoints_mask



def find_troughs_binary(x, **kwargs):
    """
    Mimics scipy.signal.find_peaks, but finds troughs (local minima) instead. Otherwise, it's the same. + NaN boundaries
    -> binary mask
    """
    troughs, _ = find_peaks(-np.asarray(x), **kwargs)
    
    changepoints_mask = add_NaN_boundaries(x, troughs)
    
    return changepoints_mask


def find_nearest_turning_points_binary(x, threshold=1, max_value=None, **kwargs):
    """
    For each peak (found via scipy.signal.find_peaks), find the nearest left and right turning points where the gradient
    is within the (-threshold, threshold) and x value is less than max_value (if provided).

    Inputs:
    x - 1D array of values (e.g., any variable profile)
    threshold - gradient threshold (default = 1)
    max_value - maximum value to qualify as a turning point (default = None, no filtering)
    **kwargs - additional keyword arguments for scipy.signal.find_peaks

    Outputs:
    nearest_turning_points - array of nearest left/right turning point indices
    """
 
        
    grad = np.gradient(x)

    # Find turning points where gradient is within threshold
    turning_points = np.where((grad > -threshold) & (grad < threshold))[0]

    # Optionally filter turning points where x value is less than max_value
    if max_value is not None:
        turning_points = turning_points[x[turning_points] < max_value]
        

    peaks, _ = find_peaks(np.asarray(x), **kwargs)

    nearest_turning_points = []

    for peak in peaks:
        # Left candidates (points before peak)
        left_candidates = turning_points[turning_points < peak]

        # Right candidates (points after peak)
        right_candidates = turning_points[turning_points > peak]

        # Add nearest left turning point (largest index < peak)
        if len(left_candidates) > 0:
            nearest_turning_points.append(left_candidates[-1])

        # Add nearest right turning point (smallest index > peak)
        if len(right_candidates) > 0:
            nearest_turning_points.append(right_candidates[0])

    nearest_turning_points = np.array(nearest_turning_points, dtype=int)

    return add_NaN_boundaries(x, nearest_turning_points)




def snap_to_nearest_changepoint(x_clicked, ds, feature_sel, **ds_kwargs):
    """
    Args:
        x_clicked: The x-coordinate value to snap.
        ds: xarray.Dataset
        feature_sel: The feature to filter changepoints by (e.g., 'speed').
        ds_kwargs: Additional selection arguments for the dataset (e.g., trials, individuals, keypoints).
        
    Returns:
        snapped_val, snapped_idx
        
    Example:
        ds_kwargs = {individuals:"Freddy", keypoints: "beakTip"}
        snapped_val, _ = snap_to_nearest_changepoint(x_clicked, cds, 'speed', **ds_kwargs)
    """
    
    
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




def correct_changepoints_one_trial(labels, ds, all_params, speed_correction=True):
    """
    Correct labels (or predictions of labels) with changepoints.
    """
    cp_kwargs = all_params["cp_kwargs"]
    
    min_motif_len = all_params.get("min_motif_len")
    stitch_gap_len = all_params.get("stitch_gap_len")
    max_expansion = all_params["changepoint_params"]["max_expansion"]
    max_shrink = all_params["changepoint_params"]["max_shrink"]
    


    ds = ds.sel(**cp_kwargs)
    ds_merged, _ = merge_changepoints(ds)
    changepoints_binary = ds_merged["changepoints"].values
    
    assert changepoints_binary.ndim == 1

    
    # Missing some of the checks from below
    if not speed_correction:
        # Simple correction without speed
        changepoint_idxs = np.where(changepoints_binary)[0]
        corrected_labels = np.zeros_like(labels, dtype=np.int8)
        
        
        changepoint_idxs = np.where(changepoints_binary)[0]
        labels = purge_small_motifs(labels, min_motif_len)
        labels = stitch_gaps(labels, stitch_gap_len)

        
        # Step two - Changepoint correction
        for label in np.unique(labels):
            if label == 0:
                continue
            
            label_mask = labels == label
            starts, ends = find_blocks(label_mask)
            
            for block_start, block_end in zip(starts, ends):
                snap_start = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_start))]
                snap_end = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_end))]



                start_expansion = block_start - snap_start  # Positive = expanding left
                start_shrink = snap_start - block_start      # Positive = shrinking from left
                
                if start_expansion > max_expansion or start_shrink > max_shrink:                    
                    snap_start = block_start
                
                end_expansion = snap_end - block_end  # Positive = expanding right
                end_shrink = block_end - snap_end      # Positive = shrinking from right
                
                if end_expansion > max_expansion or end_shrink > max_shrink:
                    snap_end = block_end
                
                # Ensure valid boundaries
                if snap_start > snap_end:
                    # Keep original boundaries if snapping creates invalid range
                    snap_start = block_start
                    snap_end = block_end
                

                if snap_end < len(corrected_labels):
                    if corrected_labels[snap_end] != 0 and not corrected_labels[snap_end] == label:
                        snap_end = snap_end - 1
                        
                

                # Clear original block and set corrected boundaries
                corrected_labels[block_start:block_end+1] = 0
                if snap_start < snap_end:
                    corrected_labels[snap_start:snap_end+1] = label
                    
                    
                
    # else:
        
    #     # Speed-based correction
    #     corrected_labels = labels.copy()
        
    #     repo_root = Path(__file__).resolve().parents[2]
    #     speed_stats_path = repo_root / "configs" / f"{all_params['target_individual']}_speed_stats.npy"
        
    #     speed_stats = np.load(speed_stats_path, allow_pickle=True).item()
        
    #     speed = ds["speed"].values
        

    #     cp_ds = ds.filter_by_attrs(type="changepoints")
        
    #     speed_da_list = [da_candidate
    #             for da_candidate in cp_ds.data_vars.values()
    #             if da_candidate.attrs.get('target_feature') == 'speed'
    #     ]

    #     changepoint_idxs = np.concatenate([
    #         np.where(da.values)[0] for da in speed_da_list
    #     ])
    #     changepoint_idxs = np.unique(changepoint_idxs)
        
        
    #     assert speed.ndim == 1
    #     assert changepoint_idxs.ndim == 1
        
        

    #     # Process each label class
    #     for label in np.unique(labels):
    #         if label == 0:
    #             continue
            
    #         if label not in speed_stats:
    #             max_start = float('inf')
    #             max_end = float('inf')
      
    #         else:
    #             max_start = speed_stats[label]["max_starts"]
    #             max_end = speed_stats[label]["max_ends"]


            

    #         # Find contiguous blocks of this label
    #         label_mask = labels == label
    #         diff = np.diff(np.concatenate(([0], label_mask.astype(int), [0])))
    #         starts = np.where(diff == 1)[0]
    #         ends = np.where(diff == -1)[0] - 1
            
    #         for block_start, block_end in zip(starts, ends):
    #             # Skip small blocks
    #             if (block_end - block_start + 1) < min_motif_len:
    #                 corrected_labels[block_start:block_end+1] = 0
    #                 continue
                
                

    #             min_prominence = all_params["changepoint_params"]["min_prominence_peaks"]
                
    #             if label in all_params["changepoint_params"].get("prominence_exceptions", []):
    #                 min_prominence = 2.0
                
                
    #             peaks, _ = find_peaks(speed, prominence=min_prominence)
                
                
    #             peaks_in_range = peaks[(peaks >= block_start) & (peaks <= block_end)]
                
    #             if len(peaks_in_range) == 0:
    #                 # No peaks, just snap to nearest changepoints
    #                 snap_start = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_start))]
    #                 snap_end = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_end))]
    #             else:
    #                 # Use peaks to find changepoints
    #                 first_peak = peaks_in_range.min()
    #                 last_peak = peaks_in_range.max()
                    
                    
    #                 # Find left changepoint (before first peak, speed < max_start)
    #                 left_cps = changepoint_idxs[changepoint_idxs < first_peak]
    #                 valid_left = left_cps[speed[left_cps] < max_start]
    #                 snap_start = valid_left[-1] if len(valid_left) > 0 else left_cps[-1]
                    
                    
    #                 # Find right changepoint (after last peak, speed < max_end)
    #                 right_cps = changepoint_idxs[changepoint_idxs > last_peak]
    #                 valid_right = right_cps[speed[right_cps] < max_end]
    #                 snap_end = valid_right[0] if len(valid_right) > 0 else right_cps[0]


     

    #             if (block_start - snap_start) > max_expansion or (snap_start - block_start) > max_shrink:
    #                 snap_start = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_start))]

    #                 if (block_start - snap_start) > max_expansion or (snap_start - block_start) > max_shrink:
    #                     snap_start = block_start


                  
    #             if (snap_end - block_end) > max_expansion or (block_end - snap_end) > max_shrink:
    #                 snap_end = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_end))]

    #                 if (snap_end - block_end) > max_expansion or (block_end - snap_end) > max_shrink:
    #                     snap_end = block_end
                        

    #             if snap_end < len(corrected_labels):
    #                 if corrected_labels[snap_end] != 0 and not corrected_labels[snap_end] == label:
    #                     snap_end = snap_end - 1


    #             # Clear original block and set corrected boundaries
    #             mask =  (corrected_labels[block_start:block_end+1] == label)
    #             corrected_labels[block_start:block_end+1][mask] = 0
    #             if snap_start < snap_end:
    #                 corrected_labels[snap_start:snap_end+1] = label


    corrected_labels = remove_small_blocks(corrected_labels, min_motif_len)
    corrected_labels = fix_endings(corrected_labels, changepoints_binary)
        
    
    
    
    return corrected_labels




def merge_changepoints(ds):
    """Merge all changepoint variables into a single combined mask."""

    # Make a copy to ensure mutability
    ds = ds.copy()
    cp_ds = ds.filter_by_attrs(type="changepoints")

    target_feature = []
    for var in cp_ds.data_vars:
        target_feature.append(cp_ds[var].attrs["target_feature"])
        
    if np.unique(target_feature).size > 1:
        raise ValueError(f"Not allowed to merge changepoints for different target features: {np.unique(target_feature)}")

    # This will merge all changepoint variables into one.
    # If ds has not been filtered, e.g. by:
    # cp_kwargs = {"individuals": "Poppy", "keypoints": "beakTip"}
    # ds.sel(**cp_kwargs)
    # Then this will also merge across individuals and keypoints, thus
    # maybe overspecified changepoints.
    dims = [dim for dim in cp_ds.dims if dim not in ["trials", "time"]]

    ds["changepoints"] = (cp_ds
                                .to_array()
                                .any(dim=["variable"] + dims)
                                .astype(float))
    ds["changepoints"].attrs["type"] = "changepoints"

    # Remove individual changepoint variables
    ds = ds.drop_vars(list(cp_ds.data_vars))

    return ds, target_feature[0]








def more_changepoint_features(
    changepoint_binary: np.ndarray,
    targ_feat_vals: np.ndarray,
    sigmas: List[float],
    distribution: Literal["gaussian", "laplacian"] = "laplacian",
) -> np.ndarray:
    """Create changepoint-based features from binary changepoint array.
    
    Args:
        changepoint_binary: Binary array where 1 indicates changepoints
        targ_feat_vals: Target feature values (e.g., speed) associated with changepoints
        sigmas: Scale parameters for distribution peaks
        distribution: Type of distribution ("gaussian" or "laplacian")
        
    Returns:
        Changepoint features with distribution peaks, weighted versions, and segment IDs
    """
    features = [changepoint_binary]
    seq_length = len(changepoint_binary)
    changepoint_indices = np.where(changepoint_binary)[0]
    
    print(sigmas)
    
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
            
    print(cp_binary_peak.shape, weighted_cps.shape, segment_ids.shape)

    return np.column_stack([cp_binary_peak, weighted_cps, segment_ids])














    # add later with speed above to make feature explanation 

    # changepoints = [0, 20, 22, 35, 79, 89]
    # seq_len = 100

    # default_features = make_changepoint_features(changepoints, seq_len)
    
    
    # fig, axes = plt.subplots(5, 1, figsize=(4, 3), sharex=True)

    # x = np.arange(seq_len)
    # row_labels = ['binary', 'σ=3', 'σ=5', 'σ=7', 'segIDs']

    # for i in range(5):
    #     axes[i].plot(x, default_features[:, i], color='k' if i == 0 else ['r', 'b', 'g', 'm'][i-1], linewidth=2)
    #     axes[i].set_yticks([])
    #     axes[i].text(seq_len + 0.5, 0.5, row_labels[i], va='center', ha='left', fontsize=10)
    #     axes[i].set_xlim([0, 100])
    # axes[-1].set_xlabel('Sequence Index')
    # plt.subplots_adjust(hspace=0)  # tighter vertical spacing
    # for ax in axes:
    #     ax.get_yaxis().set_visible(False)
    #     ax.get_xaxis().set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.spines['left'].set_visible(False)