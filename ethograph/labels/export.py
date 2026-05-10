from __future__ import annotations

import pandas as pd


def correct_offsets_trial(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Apply gap correction to a single trial's interval DataFrame.

    For each individual, pulls back ``offset_s`` when the gap to the next onset
    is smaller than ``eps`` so pynapple can resolve all intervals.

    Works on the per-trial format (columns: trial, onset_s, offset_s, labels,
    individual) returned by ``app_state.get_trial_intervals()``.

    Returns
    -------
    tuple[pd.DataFrame, int, int]
        Corrected DataFrame, number of offsets corrected, number of negative gaps found.
    """
    if df.empty:
        return df, 0, 0
    eps = 1e-4
    df = df.copy().sort_values(["individual", "onset_s"]).reset_index(drop=True)

    corrected = 0
    negative_gaps = 0
    for _, group in df.groupby("individual"):
        idx = group.index.tolist()
        for i in range(len(idx) - 1):
            gap = df.loc[idx[i + 1], "onset_s"] - df.loc[idx[i], "offset_s"]

            if gap < 0:
                negative_gaps += 1

            if gap < eps:
                corrected += 1
                df.loc[idx[i], "offset_s"] = df.loc[idx[i + 1], "onset_s"] - eps
                df.loc[idx[i], "duration"] = df.loc[idx[i], "offset_s"] - df.loc[idx[i], "onset_s"]

                if "offset_global" in df.columns and "onset_global" in df.columns:
                    df.loc[idx[i], "offset_global"] = df.loc[idx[i + 1], "onset_global"] - eps
                    df.loc[idx[i], "duration"] = df.loc[idx[i], "offset_s"] - df.loc[idx[i], "onset_s"]

    if "onset_global" in df.columns:
        df.sort_values(["individual", "onset_global"], inplace=True)

    return df, corrected, negative_gaps


def enrich_labels_df(
    all_labels_df: pd.DataFrame,
    nwb_alignment=None,
    keep_attrs: list[str] | None = None,
    dt=None,
    metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Enrich a raw labels DataFrame with computed columns for analysis export.

    Takes the in-memory ``_all_labels_df`` (with columns ``onset_s``, ``offset_s``,
    ``labels``, ``individual``, ``trial``) and adds session timing, duration,
    sequence info, and trial attributes from metadata_df and ds.attrs.

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
    metadata_df : pd.DataFrame, optional
        Trial metadata table. Columns are merged per trial into the enriched output.
        If metadata_df contains "poscat" or "num_pellets", those are used instead
        of falling back to ds.attrs.

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

    # Determine which legacy attrs to fetch from ds (only if not in metadata_df)
    legacy_fallback_attrs = []
    if metadata_df is None or "poscat" not in metadata_df.columns:
        legacy_fallback_attrs.append("poscat")
    if metadata_df is None or "num_pellets" not in metadata_df.columns:
        legacy_fallback_attrs.append("num_pellets")

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

        # Trial attrs from metadata_df (primary source)
        if metadata_df is not None and not metadata_df.empty:
            # Support both "trial" column or index
            if "trial" in metadata_df.columns:
                trial_meta = metadata_df[metadata_df["trial"] == trial_id]
            else:
                # Try by index
                try:
                    trial_meta = metadata_df.loc[[trial_id]]
                except KeyError:
                    trial_meta = pd.DataFrame()

            if not trial_meta.empty:
                # Skip computed/derived columns that shouldn't be exported
                skip_cols = {
                    "trial",
                    "offsets_corrected",
                    "small_labels_purged",
                    "model_confidence",
                    "model_confidence_level",
                }
                for col in trial_meta.columns:
                    if col not in skip_cols:
                        group[col] = trial_meta[col].iloc[0]

        # Legacy fallback: poscat and num_pellets from ds.attrs only if not in metadata_df
        if legacy_fallback_attrs:
            try:
                ds = dt.trial(trial_id)
                for attr in legacy_fallback_attrs:
                    if attr in ds.attrs:
                        group[attr] = ds.attrs[attr]
            except (KeyError, ValueError, AttributeError):
                pass

        enriched_rows.append(group)

    return pd.concat(enriched_rows, ignore_index=True)
