import xarray as xr
import numpy as np
import pandas as pd
from itertools import groupby
from ethograph.utils.io import TrialTree
from typing import Dict, List

def sel_valid(da, sel_kwargs):
    """
    Selects data from an xarray DataArray using only valid dimension keys.
    This function filters the selection keyword arguments to include only those
    keys that are present in the DataArray's dimensions. It then performs the
    selection using the filtered arguments.
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
    filt_kwargs = {k: v for k, v in sel_kwargs.items() if k in valid_keys}
    da = da.sel(**filt_kwargs).squeeze()
    da = da.transpose('time', ...)

    data = da.values
    assert data.ndim in [1, 2] # either (time,) or (time, space)/ (time, RGB), ...

    return data, filt_kwargs





def stack_trials(trees: Dict[str, TrialTree], keep_vars: List[str], keep_attrs: List[str]) -> xr.Dataset:
    """Stack trials with preserved condition attributes as coordinates."""
    datasets = []
    
    for tree in trees.values():
        for trial in tree.children:
            if not trial.startswith("trial_"):
                continue
            trial_ds = tree[trial].ds[keep_vars].copy()
            
            for attr in keep_attrs:
                if attr in tree[trial].ds.attrs:
                    trial_ds[attr] = tree[trial].ds.attrs[attr]
            
            datasets.append(trial_ds)
    
    return xr.concat(datasets, dim="trials", join="outer")



def ds_to_df(ds):
    """Convert xarray Dataset to segment DataFrame with time info."""
    
    
    df = []
    times = ds.time.values
    
    
    for trial_idx in range(ds.sizes['trials']):
        labels = ds.labels.isel(trials=trial_idx).values
        trial_id = ds.trials.values[trial_idx]
        
        # Exclude eat label (13) as often cut off
        valid_mask = (labels > 0) & ~np.isnan(labels) # &  (labels != 13)
        
        if not np.any(valid_mask):
            continue
        
        valid_indices = np.where(valid_mask)[0]
        first_idx = valid_indices[0]
        last_idx = valid_indices[-1]
        
      
        labels_cropped = labels[first_idx:last_idx + 1]
        times_cropped = times[first_idx:last_idx + 1]
        

        time_offset = times_cropped[0]
        times_adjusted = times_cropped - time_offset
        actual_trial_length = times_adjusted[-1]
        
        poscat_value = float(ds.poscat.isel(trials=trial_idx).item())
        num_pellets_value = float(ds.num_pellets.isel(trials=trial_idx).item())
        
        trial_events = []
        label_sequence = []
        

        for label, group in groupby(enumerate(labels_cropped), key=lambda x: x[1]):

            if label > 0 and not np.isnan(label):
                indices = [i for i, _ in group]
                start_time = float(times_adjusted[indices[0]])
                stop_time = float(times_adjusted[indices[-1]])
    
                
        
                trial_events.extend([start_time, stop_time])
                label_sequence.append(int(label))
                
                df.append({
                    'trial': trial_id,
                    'label': int(label),
                    'start': start_time,
                    'stop': stop_time,
                    'duration': stop_time - start_time,
                    'sequence_len': actual_trial_length,
                    'poscat': poscat_value,
                    'num_pellets': num_pellets_value,
                })
        
     
    
        
        # Add event_times to all records of this trial
        trial_records = [r for r in df if r['trial'] == trial_id]
        for record in trial_records:
            record['event_times'] = trial_events
            record['sequence'] = label_sequence
        
    return pd.DataFrame(df)