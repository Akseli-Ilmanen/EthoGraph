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

    Works on the compact per-trial format (columns: onset_s, offset_s, labels,
    individual) returned by ``app_state.get_trial_intervals()``.
    """
    if df.empty:
        return df
    eps = 1e-3
    df = df.copy().sort_values(["individual", "onset_s"]).reset_index(drop=True)
    for _, group in df.groupby("individual"):
        idx = group.index.tolist()
        for i in range(len(idx) - 1):
            gap = df.loc[idx[i + 1], "onset_s"] - df.loc[idx[i], "offset_s"]
            if abs(gap) < eps:
                df.loc[idx[i], "offset_s"] = df.loc[idx[i + 1], "onset_s"] - eps
                
            
            
            
                

    # Internal to crow lab, we had a legacy labeling system that was frame-wise(200 Hz), and this correction should fix those labels.
    dt = 1 / 200  # 5 ms frame rate
    group_cols = ["trial", "individual"]
    if "session" in df.columns:
        group_cols.insert(0, "session")
    for _, group in df.groupby(group_cols):
        individual = group["individual"].iloc[0]
        
        # Won't affect other users.
        if not any(name in individual for name in ["Ivy", "Freddy"]):
            continue

        print(f"Processing session {group['session'].iloc[0]}, trial {group['trial'].iloc[0]}, individual {individual}")
        
        idx = group.index
        
        for i in range(len(idx) - 1):
            current = idx[i]
            next_row = idx[i + 1]
            
            gap = df.loc[next_row, "onset_s"] - df.loc[current, "offset_s"]
            
            if abs(gap - dt) < eps:
                df.loc[current, "offset_s"] = df.loc[next_row, "onset_s"] - eps
                df.loc[current, "offset_global"] = df.loc[next_row, "onset_global"] - eps
                df.loc[current, "duration"] = (
                    df.loc[current, "offset_s"] - df.loc[current, "onset_s"]
                )

                
    return df




def enrich_labels_df(
    all_labels_df: pd.DataFrame,
    dt: "TrialTree",
    keep_attrs: list[str] | None = None,
) -> pd.DataFrame:
    """Enrich a raw labels DataFrame with computed columns for analysis export.

    Takes the in-memory ``_all_labels_df`` (with columns ``onset_s``, ``offset_s``,
    ``labels``, ``individual``, ``trial``) and adds session timing, duration,
    sequence info, and trial attributes from ``dt``.

    Parameters
    ----------
    all_labels_df : pd.DataFrame
        Raw labels with required columns: onset_s, offset_s, labels, individual, trial.
    dt : TrialTree
        The data tree (for session timing and trial attributes).
    keep_attrs : list[str], optional
        Trial-level ``ds.attrs`` keys to include as extra columns.

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
            t_start = dt.start_time(trial_id)
        except (AttributeError, KeyError, ValueError):
            pass
        try:
            t_stop = dt.stop_time(trial_id)
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