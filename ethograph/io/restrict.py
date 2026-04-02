"""Restriction logic for building display windows from trials, labels, or sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pynapple as nap
import xarray as xr

from ethograph.gui.plots_timeseriessource import RestrictionWindow, TimeRange

if TYPE_CHECKING:
    from ethograph.gui.plots_timeseriessource import TrialAlignment
    from ethograph.io.trialtree import TrialTree


def build_trial_window(
    trial_alignment: TrialAlignment,
    trial_id: int | str,
    extra_t0: float = 0.0,
    extra_t1: float = 0.0,
) -> RestrictionWindow:
    """Build a restriction window covering an entire trial."""
    core = trial_alignment.trial_range
    time_range = TimeRange(
        max(0.0, core.start_s - extra_t0),
        core.end_s + extra_t1,
    )
    return RestrictionWindow(
        mode="trial",
        time_range=time_range,
        core_range=core,
        trial_id=trial_id,
    )


def build_label_window(
    labels_df: pd.DataFrame,
    label_idx: int,
    trial_bounds: TimeRange,
    extra_t0: float = 0.0,
    extra_t1: float = 0.0,
) -> RestrictionWindow:
    """Build a restriction window around a single label instance.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Full labels DataFrame (must have onset_s, offset_s, labels, trial columns).
    label_idx : int
        Row index into *labels_df*.
    trial_bounds : TimeRange
        Bounds of the trial this label belongs to (for clamping).
    extra_t0, extra_t1 : float
        Extra padding before/after the label interval.
    """
    row = labels_df.iloc[label_idx]
    onset = float(row["onset_s"])
    offset = float(row["offset_s"])
    core = TimeRange(onset, offset)
    time_range = TimeRange(
        max(trial_bounds.start_s, onset - extra_t0),
        min(trial_bounds.end_s, offset + extra_t1),
    )
    return RestrictionWindow(
        mode="label",
        time_range=time_range,
        core_range=core,
        trial_id=row.get("trial"),
        label_info={
            "label_id": int(row["labels"]),
            "individual": row.get("individual"),
            "onset_s": onset,
            "offset_s": offset,
            "row_idx": label_idx,
        },
    )


def build_sequence_window(
    match: dict,
    trial_bounds: TimeRange,
    extra_t0: float = 0.0,
    extra_t1: float = 0.0,
) -> RestrictionWindow:
    """Build a restriction window spanning a matched label sequence.

    Parameters
    ----------
    match : dict
        From :func:`ethograph.utils.sequences.match_sequences`.
        Must contain ``trial``, ``onset_s``, ``offset_s``, ``pattern``.
    trial_bounds : TimeRange
        Bounds of the trial for clamping.
    extra_t0, extra_t1 : float
        Extra padding before/after the sequence span.
    """
    onset = float(match["onset_s"])
    offset = float(match["offset_s"])
    core = TimeRange(onset, offset)
    time_range = TimeRange(
        max(trial_bounds.start_s, onset - extra_t0),
        min(trial_bounds.end_s, offset + extra_t1),
    )
    return RestrictionWindow(
        mode="sequence",
        time_range=time_range,
        core_range=core,
        trial_id=match["trial"],
        sequence_info={
            "pattern": match.get("pattern"),
            "match_rows": match.get("match_rows"),
        },
    )


def find_closest_trial(dt: TrialTree, global_time: float) -> tuple[int | str, float]:
    """Given a session-absolute time, find the enclosing trial.

    Returns
    -------
    trial_id : int or str
        Identifier of the trial containing *global_time*.
    trial_relative_t : float
        Time relative to the trial's start.

    Raises
    ------
    ValueError
        If *global_time* falls outside all trial epochs.
    """
    ep = dt.trials_ep
    if ep is None:
        raise ValueError("No trial timing information available")

    starts = np.asarray(ep.start)
    ends = np.asarray(ep.end)
    mask = (starts <= global_time) & (global_time <= ends)

    if mask.any():
        idx = int(np.argmax(mask))
    else:
        dists = np.minimum(
            np.abs(starts - global_time),
            np.abs(ends - global_time),
        )
        idx = int(np.argmin(dists))

    trial_id = dt.trials[idx]
    trial_start = float(starts[idx])
    return trial_id, global_time - trial_start


def restrict_xarray(
    ds: xr.Dataset,
    time_range: TimeRange,
    time_coord_name: str = "time",
) -> xr.Dataset:
    """Slice an xarray Dataset to a time range."""
    return ds.sel({time_coord_name: slice(time_range.start_s, time_range.end_s)})


def restrict_pynapple(
    obj: nap.Tsd | nap.TsdFrame | nap.TsdTensor | nap.Ts | nap.TsGroup,
    time_range: TimeRange,
) -> nap.Tsd | nap.TsdFrame | nap.TsdTensor | nap.Ts | nap.TsGroup:
    """Restrict a pynapple object to a time range."""
    ep = nap.IntervalSet(start=time_range.start_s, end=time_range.end_s)
    return obj.restrict(ep)
