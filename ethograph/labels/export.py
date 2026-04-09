import numpy as np
import pandas as pd
from ethograph import TrialTree
from pathlib import Path
from ethograph.labels.intervals import empty_intervals
import xarray as xr





def correct_offsets_trial(df: pd.DataFrame) -> pd.DataFrame:
    """Apply gap correction to a single trial's interval DataFrame.

    For each individual, pulls back ``offset_s`` when the gap to the next onset
    is smaller than ``eps`` so pynapple can resolve all intervals.

    Works on the per-trial format (columns: trial, onset_s, offset_s, labels,
    individual) returned by ``app_state.get_trial_intervals()``.
    """
    if df.empty:
        return df
    eps = 1e-4
    df = df.copy().sort_values(["individual", "onset_s"]).reset_index(drop=True)
    
    counter = 0
    for _, group in df.groupby("individual"):
        idx = group.index.tolist()
        for i in range(len(idx) - 1):
            gap = df.loc[idx[i + 1], "onset_s"] - df.loc[idx[i], "offset_s"]
            
            if gap < 0:
                raise ValueError(f"Negative gap of {gap:.3f} seconds between intervals for individual {group['individual'].iloc[0]} at index {idx[i]} and {idx[i + 1]}. Check your data for overlapping intervals.")
            
            if gap < eps:
                counter += 1
                
                df.loc[idx[i], "offset_s"] = df.loc[idx[i + 1], "onset_s"] - eps
                df.loc[idx[i], "duration"] = df.loc[idx[i], "offset_s"] - df.loc[idx[i], "onset_s"]

            
                if "offset_global" in df.columns and "onset_global" in df.columns:
                    df.loc[idx[i], "offset_global"] = df.loc[idx[i + 1], "onset_global"] - eps
                    df.loc[idx[i], "duration"] = df.loc[idx[i], "offset_s"] - df.loc[idx[i], "onset_s"]
                
                
                
    print(f"Corrected {counter} offsets with gap smaller than {eps:.3f} seconds.")
                        
            
    if "onset_global" in df.columns:
        df.sort_values(["individual", "onset_global"], inplace=True)
                
    return df




def enrich_labels_df(
    all_labels_df: pd.DataFrame,
    nwb_alignment=None,
    keep_attrs: list[str] | None = None,
    dt=None,
) -> pd.DataFrame:
    """Enrich a raw labels DataFrame with computed columns for analysis export.

    Takes the in-memory ``_all_labels_df`` (with columns ``onset_s``, ``offset_s``,
    ``labels``, ``individual``, ``trial``) and adds session timing, duration,
    sequence info, and trial attributes.

    Parameters
    ----------
    all_labels_df : pd.DataFrame
        Raw labels with required columns: onset_s, offset_s, labels, individual, trial.
    nwb_alignment
        Session metadata (for trial timing).
    keep_attrs : list[str], optional
        Trial-level ``ds.attrs`` keys to include as extra columns (xarray only).
    dt : TrialTree, optional
        Xarray data tree (only needed for ``keep_attrs`` and session name).

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with one row per non-background segment.
    """
    if all_labels_df is None or all_labels_df.empty:
        return pd.DataFrame()

    if keep_attrs is None:
        keep_attrs = []

    valid = all_labels_df[all_labels_df["labels"] > 0].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["duration"] = valid["offset_s"] - valid["onset_s"]

    # Session info (only when session attr exists)
    session_name = getattr(dt, "attrs", {}).get("session", None)
    if session_name is not None:
        valid["session"] = session_name
        valid["session_trial"] = valid["trial"].apply(lambda t: f"{session_name}_{t}")

    # Per-trial: sequence, sequence_idx, timing, attrs
    enriched_rows = []
    for trial_id, group in valid.groupby("trial", sort=False):
        group = group.sort_values("onset_s").reset_index(drop=True)
        sequence = group["labels"].tolist()
        group["sequence_idx"] = range(len(group))
        group["sequence"] = "-".join(str(s) for s in sequence)

        # Session timing
        t_start, t_stop = None, None
        try:
            t_start = nwb_alignment.start_time(trial_id)
        except (AttributeError, KeyError, ValueError):
            pass
        try:
            t_stop = nwb_alignment.stop_time(trial_id)
        except (AttributeError, KeyError, ValueError):
            pass

        if t_start is not None:
            group["trial_onset"] = t_start
            group["onset_global"] = t_start + group["onset_s"]
            group["offset_global"] = t_start + group["offset_s"]
        if t_stop is not None:
            group["trial_offset"] = t_stop

        # Trial attrs
        try:
            ds = dt.trial(trial_id)
            for attr in keep_attrs:
                if attr in ds.attrs:
                    group[attr] = ds.attrs[attr]
        except (KeyError, ValueError):
            pass

        enriched_rows.append(group)

    return pd.concat(enriched_rows, ignore_index=True)