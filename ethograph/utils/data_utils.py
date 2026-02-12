from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

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
        Dictionary mapping tree keys to TrialTree objects.
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

    for tree in trees.values():
        for trial in tree.children:
            if not trial.startswith(TrialTree.TRIAL_PREFIX):
                continue
            print(tree[trial].ds.data_vars)
            trial_ds = tree[trial].ds[keep_vars].copy()

            for attr in keep_attrs:
                if attr in tree[trial].ds.attrs:
                    trial_ds[attr] = tree[trial].ds.attrs[attr]

            datasets.append(trial_ds)

    return xr.concat(datasets, dim="global_trials", join="outer")






def ds_to_df_legacy(ds):
    """Convert dense-label stacked Dataset to segment DataFrame.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'global_trials' and 'time' dimensions.
        Expected variables: labels (dense), poscat, num_pellets.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: session, trial, label, start, stop, duration,
        poscat, num_pellets, event_times, sequence.
    """
    df = []
    times = ds.time.values

    for trial_idx in range(ds.sizes['global_trials']):
        session = ds.session.values[trial_idx]
        raw_trial_id = ds.trial.values[trial_idx]
        labels = ds.labels.isel(global_trials=trial_idx).values



        if isinstance(raw_trial_id, str) and raw_trial_id.startswith(TrialTree.TRIAL_PREFIX):
            trial_id = TrialTree.trial_id(raw_trial_id)
        else:
            trial_id = raw_trial_id




        poscat_value = float(ds.poscat.isel(global_trials=trial_idx).item())
        num_pellets_value = float(ds.num_pellets.isel(global_trials=trial_idx).item())

        trial_events = []
        label_sequence = []


        for label, group in groupby(enumerate(labels), key=lambda x: x[1]):

            if label > 0 and not np.isnan(label):
                indices = [i for i, _ in group]
                start_time = float(times[indices[0]])
                stop_time = float(times[indices[-1]])


                trial_events.extend([start_time, stop_time])
                label_sequence.append(int(label))

                df.append({
                    'session': session,
                    'trial': trial_id,
                    'label': int(label),
                    'start': start_time,
                    'stop': stop_time,
                    'duration': stop_time - start_time,
                    'poscat': poscat_value,
                    'num_pellets': num_pellets_value,
                })




        for record in df:
            if record['session'] == session and record['trial'] == trial_id:
                record['event_times'] = trial_events
                record['sequence'] = label_sequence

    return pd.DataFrame(df)


def trees_to_df(
    trees: Dict[str, TrialTree],
    keep_attrs: List[str],
) -> pd.DataFrame:
    """Collect interval labels from trees into a segment DataFrame.

    Reads interval-format labels (onset_s, offset_s, labels, individual)
    directly from each trial's label_dt, avoiding unnecessary xarray stacking.

    Parameters
    ----------
    trees : Dict[str, TrialTree]
        Dictionary mapping tree keys to TrialTree objects.
    keep_attrs : List[str]
        Trial-level attributes to include as columns (e.g. 'session', 'poscat').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: session, trial, label, start, stop, duration,
        event_times, sequence, plus any columns from keep_attrs.
    """
    from ethograph.utils.label_intervals import xr_to_intervals

    rows = []

    for tree in trees.values():
        label_dt = tree.get_label_dt()

        for trial_name in tree.children:
            if not trial_name.startswith(TrialTree.TRIAL_PREFIX):
                continue

            trial_id = TrialTree.trial_id(trial_name)
            trial_ds = tree[trial_name].ds
            intervals = xr_to_intervals(label_dt[trial_name].ds)

            attrs = {}
            for attr in keep_attrs:
                if attr in trial_ds.attrs:
                    attrs[attr] = trial_ds.attrs[attr]

            valid = intervals[intervals["labels"] > 0].sort_values("onset_s")

            event_times = [t for _, r in valid.iterrows() for t in (r["onset_s"], r["offset_s"])]
            sequence = valid["labels"].tolist()

            for _, seg in valid.iterrows():
                row = {
                    'session': attrs.get('session', ''),
                    'trial': trial_id,
                    'label': int(seg["labels"]),
                    'start': seg["onset_s"],
                    'stop': seg["offset_s"],
                    'duration': seg["offset_s"] - seg["onset_s"],
                    'event_times': event_times,
                    'sequence': sequence,
                }
                row.update(attrs)
                rows.append(row)

    return pd.DataFrame(rows)




