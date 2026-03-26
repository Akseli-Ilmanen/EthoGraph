from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple
import pandas as pd
import xarray as xr


from ethograph.utils.labels import load_label_mapping, correct_offsets

if TYPE_CHECKING:
    from ethograph.utils.trialtree import TrialTree
    

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
    
    if time_dim is None:
        raise ValueError("No dimension containing 'time' found in the DataArray.")
    
    da = da.transpose(time_dim, ...)

    data = da.values
    assert data.ndim in [1, 2] # either (time,) or (time, space)/ (time, RGB), ...

    return data, filt_kwargs

def get_time_coord(da: xr.DataArray) -> xr.DataArray | None:
    """Select whichever time coord is available for a given data array.

    Dimension coordinates are preferred over non-dimension coordinates so
    that auxiliary coords (e.g. ``time_labels``, ``time_aux``) with fewer
    samples do not shadow the primary time axis.
    """
    time_dims = [d for d in da.dims if 'time' in d.lower()]
    if time_dims:
        return da.coords[time_dims[0]]
    time_coord = next((c for c in da.coords if 'time' in c.lower()), None)
    if time_coord is None:
        return None
    return da.coords[time_coord]


def trees_to_df(
    trees: Dict[str, TrialTree],
    keep_attrs: List[str],
) -> pd.DataFrame:
    """Convert labels from single or multiple TrialTrees into DataFrame format.

    Reads interval-format labels (onset_s, offset_s, labels, individual)
    from each trial's label data via ``xr_to_intervals``, filters out
    background (label == 0), and assembles one row per valid segment.

    Parameters
    ----------
    trees : Dict[str, TrialTree] | TrialTree | Path | str | list
        One or more trial trees. Accepts a single ``TrialTree``, a dict
        mapping keys to trees, a path (or list of paths) to saved trees,
        or a list of ``TrialTree`` objects.
    keep_attrs : List[str]
        Trial-level ``ds.attrs`` keys to propagate as columns
        (e.g. ``'session'``, ``'poscat'``).

    Returns
    -------
    pd.DataFrame
        One row per non-background segment with columns:

        - **session** – from ``ds.attrs['session']`` (empty string if absent)
        - **trial** – trial identifier
        - **individual** – subject identifier from the interval
        - **labels** – integer action label
        - **onset_s / offset_s** – segment boundaries in trial-relative seconds
        - **trial_onset** – absolute trial start (seconds) for ephys alignment.
        - **onset_global / offset_global** – absolute segment boundaries in ephys time.
        - **duration** – segment length in seconds
        - **sequence_idx** – zero-based position within the trial's label sequence
        - **sequence** – dash-joined string of all non-background labels in
          the trial (e.g. ``"1-3-2"``)
        - any additional columns specified by *keep_attrs*
    """
    from ethograph.utils.trialtree import TrialTree
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
        for trial_id, ds in dt.trial_items():
            intervals = xr_to_intervals(ds)

            attrs = {}
            for attr in keep_attrs:
                if attr in ds.attrs:
                    attrs[attr] = ds.attrs[attr]

            valid = intervals[intervals["labels"] > 0].sort_values("onset_s")

            sequence = valid["labels"].tolist()

            for idx, (_, seg) in enumerate(valid.iterrows()):
                row = {
                    'session': ds.attrs.get('session', ''), # optional
                    'trial': trial_id,
                    'session_trial': f"{ds.attrs.get('session', '')}_{trial_id}",
                    'individual': seg["individual"],
                    'labels': int(seg["labels"]),
                    'onset_s': seg["onset_s"],
                    'offset_s': seg["offset_s"],
                }
                t_start = None
                if hasattr(dt, 'session') and dt.session is not None and "start_time" in dt.session:
                    try:
                        t_start = float(dt.session.start_time.sel(trial=trial_id))
                    except (KeyError, ValueError):
                        pass
                elif 'pulse_onsets' in ds:
                    t_start = float(ds.pulse_onsets.values[0]) / 30_000  # Legacy crow lab

                if t_start is not None:
                    row['trial_onset'] =  t_start
                    row['onset_global'] = t_start + seg["onset_s"]
                    row['offset_global'] = t_start + seg["offset_s"]
                    
                    
                
                row.update({
                    'duration': seg["offset_s"] - seg["onset_s"],
                    'sequence_idx': idx, # zero-indexing
                    'sequence': "-".join(str(s) for s in sequence),
                })
                row.update(attrs)
                rows.append(row)
                
    df = pd.DataFrame(rows)      
    
    # Correction of legacy label system that was frame-wise. 
    # Unless you have offset, onset exactly 5ms, thsi correction shouldnt affect your data.
    
    
    
    corrected_df = correct_offsets(df)
    
    
                    
    return corrected_df




