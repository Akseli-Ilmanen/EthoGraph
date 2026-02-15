from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from ethograph import TrialTree

def sel_valid(da, sel_kwargs):
    """
    Selects data from an xarray DataArray using only valid dimension keys.
    This function filters the selection keyword arguments to include only those
    keys that are present in the DataArray's dimensions. Uses .sel() for
    dimensions with coordinates and .isel() for dimensions without.
    Parameters
    ----------
    da : xarray.DataArray
        The DataArray from which to select data.
    sel_kwargs : dict
        Dictionary of selection arguments, where keys are coordinate names and
        values are the labels or slices to select.
    Returns
    -------
    numpy.ndarray
        The selected data as a numpy array. Where 'time' is the first dimension.
    dict
        The filtered selection arguments that were actually used.
    """

    valid_keys = set(da.dims)
    coord_keys = set(da.coords.keys())

    sel_kwargs_filtered = {}
    isel_kwargs = {}

    for k, v in sel_kwargs.items():
        if k not in valid_keys:
            continue
        if k in coord_keys:
            sel_kwargs_filtered[k] = v
        else:
            isel_kwargs[k] = int(v) if isinstance(v, str) else v

    # Only return sel-compatible kwargs (those with coordinates)
    # isel kwargs are applied but not returned since .sel() can't use them
    filt_kwargs = dict(sel_kwargs_filtered)

    if sel_kwargs_filtered:
        da = da.sel(**sel_kwargs_filtered)
    if isel_kwargs:
        da = da.isel(**isel_kwargs)
    da = da.squeeze()
    
    time_dim = next((dim for dim in da.dims if 'time' in dim), None)
    if time_dim:
        da = da.transpose(time_dim, ...)

    data = da.values
    assert data.ndim in [1, 2] # either (time,) or (time, space)/ (time, RGB), ...

    return data, filt_kwargs

def get_time_coord(da: xr.DataArray) -> xr.DataArray | None:
    """Select whichever time coord is available for a given data array."""
    coords = da.coords
    time_coord = next((c for c in coords if 'time' in c.lower()), None)
    return coords[time_coord]


def stack_trials(
    trees: Dict[str, TrialTree],
    keep_vars: List[str],
    keep_attrs: List[str],
) -> xr.Dataset:
    """Stack trials with preserved condition attributes as coordinates.

    Parameters
    ----------
    trees : Dict[str, TrialTree]
        Dictionary mapping dt keys to TrialTree objects.
    keep_vars : List[str]
        Variables to keep from each trial dataset.
    keep_attrs : List[str]
        Attributes to preserve as variables (e.g. 'session', 'poscat').

    Returns
    -------
    xr.Dataset
        Stacked dataset with all trials.
    """
    datasets = []

    for dt in trees.values():
        for trial in dt.children:
            if not trial.startswith(TrialTree.TRIAL_PREFIX):
                continue
            print(dt[trial].ds.data_vars)
            trial_ds = dt[trial].ds[keep_vars].copy()

            for attr in keep_attrs:
                if attr in dt[trial].ds.attrs:
                    trial_ds[attr] = dt[trial].ds.attrs[attr]

            datasets.append(trial_ds)

    return xr.concat(datasets, dim="global_trials", join="outer")



def trees_to_df(
    trees: Dict[str, TrialTree],
    keep_attrs: List[str],
    sr: int = 30000,
) -> pd.DataFrame:
    """Collect interval labels from trees into a segment DataFrame.

    Reads interval-format labels (onset_s, offset_s, labels, individual)
    directly from each trial's label_dt, avoiding unnecessary xarray stacking.

    Parameters
    ----------
    trees : Dict[str, TrialTree] / TrialTree
        Dictionary mapping dt keys to TrialTree objects or single TrialTree.
    keep_attrs : List[str]
        Trial-level attributes to include as columns (e.g. 'session', 'poscat').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: session, trial, label, start, stop, duration,
        event_times, sequence, plus any columns from keep_attrs.
    """
    from ethograph.utils.label_intervals import xr_to_intervals

    if isinstance(trees, TrialTree):
        trees = {"_single": trees}
    elif isinstance(trees, (str, Path)):
        trees = {"tree_0": TrialTree.open(Path(trees))}
    elif isinstance(trees, list):
        if trees and isinstance(trees[0], (str, Path)):
            trees = {f"tree_{i}": TrialTree.open(Path(p)) for i, p in enumerate(trees)}
        else:
            trees = {str(i): t for i, t in enumerate(trees)}

    rows = []
    

    for dt in trees.values():        
        for trial_name in dt.children:
            if not trial_name.startswith(TrialTree.TRIAL_PREFIX):
                continue

            trial_id = TrialTree.trial_id(trial_name)
            ds = dt[trial_name].ds
            intervals = xr_to_intervals(ds)

            attrs = {}
            for attr in keep_attrs:
                if attr in ds.attrs:
                    attrs[attr] = ds.attrs[attr]

            valid = intervals[intervals["labels"] > 0].sort_values("onset_s")

            sequence = valid["labels"].tolist()

            for pos, (_, seg) in enumerate(valid.iterrows(), start=1):
                row = {
                    'session': ds.attrs.get('session', ''), # optional
                    'trial': trial_id,
                    'individual': seg["individual"],
                    'labels': int(seg["labels"]),
                    'onset_s': seg["onset_s"],
                    'offset_s': seg["offset_s"],
                }
                if 'pulse_onsets' in ds:
                    trial_onset = float(ds.pulse_onsets.values[0]) / sr
                    row['trial_onset'] = round(trial_onset, 4)
                    row['onset_global'] = round(trial_onset + seg["onset_s"], 4)
                    row['offset_global'] = round(trial_onset + seg["offset_s"], 4)
                
                row.update({
                    'duration': round(seg["offset_s"] - seg["onset_s"], 4),
                    'sequence_idx': pos,
                    'sequence': "-".join(str(s) for s in sequence),
                })
                row.update(attrs)
                rows.append(row)
                    
    return pd.DataFrame(rows)




