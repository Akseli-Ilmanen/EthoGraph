import pandas as pd
from ethograph import TrialTree
from pathlib import Path


def correct_offsets(df: pd.DataFrame, dt: float = 0.005) -> pd.DataFrame:
    """Fix off-by-one-frame gaps between consecutive action syllables.

    During dense labeling at a fixed frame rate (default 200 Hz, dt=5ms),
    each frame receives exactly one label. When two syllables are adjacent,
    the offset of syllable N and the onset of syllable N+1 compete for the
    same frame, and the onset wins. This leaves a spurious 1-frame gap
    where offset[N] + dt == onset[N+1], even though the syllables are
    truly contiguous. This function detects those gaps and snaps offset[N]
    forward to onset[N+1], correcting offset_s, offset_global, and duration.
    """
    df = df.copy().sort_values(["session", "trial", "individual", "sequence_idx"])

    # Pynapple can resolve up to 1e-6 intervals, so we must set lower. 
    eps = 1e-3
    
    for _, group in df.groupby(["session", "trial", "individual"]):
        print(f"Processing session {group['session'].iloc[0]}, trial {group['trial'].iloc[0]}")
        
        idx = group.index
        # Pass 1: fix 1-frame gaps
        for i in range(len(idx) - 1):
            current = idx[i]
            next_row = idx[i + 1]
            gap = df.loc[next_row, "onset_s"] - df.loc[current, "offset_s"]
            if abs(gap - dt) < eps:
                df.loc[current, "offset_s"] = df.loc[next_row, "onset_s"] - eps
                df.loc[current, "offset_global"] = df.loc[next_row, "onset_global"] - eps
                df.loc[current, "duration"] = df.loc[current, "offset_s"] - df.loc[current, "onset_s"]



        # Pass 2: check all pairwise combinations for offset == onset
        for i in range(len(idx)):
            for j in range(len(idx)):
                if i == j:
                    continue
                row_i = idx[i]
                row_j = idx[j]
                if abs(df.loc[row_i, "offset_s"] - df.loc[row_j, "onset_s"]) < eps:
                    print(f"Corrected gap (size: {abs(df.loc[row_i, 'offset_s'] - df.loc[row_j, 'onset_s'])}), at labels: {df.loc[row_i, 'labels']}, {df.loc[row_j, 'labels']}")
                    df.loc[row_i, "offset_s"] = df.loc[row_j, "onset_s"] - eps
                    df.loc[row_i, "offset_global"] = df.loc[row_j, "onset_global"] - eps
                    df.loc[row_i, "duration"] = df.loc[row_i, "offset_s"] - df.loc[row_i, "onset_s"]

    return df





def trees_to_df(
    trees: dict[str, "TrialTree"],
    keep_attrs: list[str],
) -> pd.DataFrame:
    """Flatten labelled segments from one or more TrialTrees into a tidy DataFrame.

    Each non-background interval (``labels > 0``) becomes one row. This is
    the standard way to export ethograph labels for analysis or ML
    pipelines.

    Parameters
    ----------
    trees : dict[str, TrialTree] | TrialTree | Path | str | list
        Flexible input — pass any of:

        * a single :class:`~ethograph.io.trialtree.TrialTree`
        * a dict mapping arbitrary keys to TrialTrees
        * a file path (or list of paths) to saved ``.nc`` files
        * a list of TrialTree objects
    keep_attrs : list[str]
        Trial-level ``ds.attrs`` keys to carry over as extra columns
        (e.g. ``["stimulus", "num_pellets"]``).

    Returns
    -------
    pandas.DataFrame
        One row per labelled segment.  Always-present columns:

        ============== =================================================
        Column         Description
        ============== =================================================
        session        ``ds.attrs["session"]`` (empty string if absent)
        trial          trial identifier
        session_trial  ``"{session}_{trial}"`` for grouping
        individual     subject identifier
        labels         integer action-label class
        onset_s        segment start in trial-relative seconds
        offset_s       segment end in trial-relative seconds
        duration       ``offset_s - onset_s``
        sequence_idx   zero-based position in the trial's label order
        sequence       dash-joined label IDs (e.g. ``"1-3-2"``)
        ============== =================================================

        Columns present only when a session table with ``start_time`` exists:

        =============== ================================================
        trial_onset     absolute trial start (seconds)
        onset_global    ``trial_onset + onset_s``
        offset_global   ``trial_onset + offset_s``
        trial_offset    absolute trial end (if ``stop_time`` available)
        =============== ================================================

    Examples
    --------
    >>> import ethograph as eto
    >>> dt = eto.open("experiment.nc")
    >>> df = eto.trees_to_df(dt, keep_attrs=["stimulus"])
    >>> df[["trial", "labels", "onset_s", "offset_s", "duration"]].head()
       trial  labels  onset_s  offset_s  duration  stimulus 
    0      1       2     0.50      1.23      0.73    tone_A
    1      1       1     1.23      2.10      0.87    tone_A
    2      2       3     0.00      0.95      0.95    tone_B

    Multiple files at once:

    >>> df = eto.trees_to_df(
    ...     ["bird_A.nc", "bird_B.nc"],
    ...     keep_attrs=["session"],
    ... )
    """
    from ethograph.io.trialtree import TrialTree
    from ethograph.labels.intervals import xr_to_intervals

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
                t_stop = None
                if hasattr(dt, 'session') and dt.session is not None and "start_time" in dt.session:
                    try:
                        t_start = float(dt.session.start_time.sel(trial=trial_id))
                    except (KeyError, ValueError):
                        pass
                    if "stop_time" in dt.session:
                        try:
                            t_stop = float(dt.session.stop_time.sel(trial=trial_id))
                        except (KeyError, ValueError):
                            pass
                    
                    
                elif 'pulse_onsets' in ds:
                    t_start = float(ds.pulse_onsets.values[0]) / 30_000  # Legacy crow lab

                if t_start is not None:
                    row['trial_onset'] =  t_start
                    row['onset_global'] = t_start + seg["onset_s"]
                    row['offset_global'] = t_start + seg["offset_s"]
                if t_stop is not None:
                    row['trial_offset'] = t_stop
                    
                    
                
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