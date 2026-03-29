"""TSV-based label storage for EthoGraph.

File format:
    onset_s                 - start time in seconds (trial-relative)
    offset_s                - end time in seconds (trial-relative)
    labels                  - integer label class ID
    individual              - individual identifier
    trial                   - trial identifier
    human_verified          - per-trial flag (0/1), repeated per row
    changepoint_corrected   - per-trial flag (0/1), repeated per row
    prediction_source       - path to prediction file that produced this label (empty if human)
    n_samples               - per-trial sample count for dense conversion (int, 0 if unknown)

Label names are managed centrally in mapping.txt.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.labels.intervals import (
    INTERVAL_COLUMNS,
    INTERVAL_DTYPES,
    empty_intervals,
)

logger = logging.getLogger(__name__)

TSV_COLUMNS = [
    "onset_s", "offset_s", "labels", "individual", "trial",
    "human_verified", "changepoint_corrected", "prediction_source", "n_samples",
]

# Per-trial metadata columns (same value for all rows in a trial)
TRIAL_META_COLUMNS = ["human_verified", "changepoint_corrected", "prediction_source", "n_samples"]

TRIAL_META_DEFAULTS = {
    "human_verified": 0,
    "changepoint_corrected": 0,
    "prediction_source": "",
    "n_samples": 0,
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def labels_tsv_path(nc_path: str | Path, suffix: str = "") -> Path:
    """Derive the labels TSV path from the .nc file path.

    Examples
    --------
    >>> labels_tsv_path("experiment/data.nc")
    PosixPath('experiment/data_labels.tsv')
    >>> labels_tsv_path("experiment/data.nc", suffix="_downsampled_100x")
    PosixPath('experiment/data_downsampled_100x_labels.tsv')
    """
    nc_path = Path(nc_path)
    return nc_path.parent / f"{nc_path.stem}{suffix}_labels.tsv"


# ---------------------------------------------------------------------------
# Load / save labels
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"onset_s", "offset_s", "labels", "individual", "trial"}


def validate_labels_tsv(df: pd.DataFrame, path: str | Path = "") -> None:
    """Validate that a labels DataFrame has all required columns.

    Raises
    ------
    ValueError
        If any of ``onset_s``, ``offset_s``, ``labels``, ``individual``, ``trial``
        are missing from the DataFrame columns.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Labels file {path} is missing required columns: {sorted(missing)}. "
            f"Required: {sorted(REQUIRED_COLUMNS)}"
        )


def load_labels_tsv(path: str | Path) -> pd.DataFrame:
    """Load labels from a TSV file.

    Parameters
    ----------
    path : str or Path
        Path to a ``_labels.tsv`` file.

    Returns
    -------
    pd.DataFrame
        Columns: ``trial``, ``onset_s``, ``offset_s``, ``labels`` (int),
        ``individual``, ``human_verified``, ``changepoint_corrected``,
        ``prediction_source``.

    Examples
    --------
    >>> df = load_labels_tsv("experiment/data_labels.tsv")
    >>> df[["trial", "onset_s", "offset_s", "labels", "individual"]].head()
       trial  onset_s  offset_s  labels individual
    0      1     0.41     0.505       1      Poppy
    1      1     0.51     0.620       2      Poppy
    """
    path = Path(path)
    if not path.exists():
        return _empty_all_labels()

    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    validate_labels_tsv(df, path)

    for col in INTERVAL_COLUMNS:
        if col in df.columns and col in INTERVAL_DTYPES:
            df[col] = df[col].astype(INTERVAL_DTYPES[col])

    for col, default in TRIAL_META_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    df["prediction_source"] = df["prediction_source"].fillna("").astype(str)
    df["n_samples"] = df["n_samples"].fillna(0).astype(int)

    return df


def save_labels_tsv(path: str | Path, df: pd.DataFrame) -> None:
    """Save labels DataFrame to TSV. Uses atomic write (tmp + rename).

    Parameters
    ----------
    path : str or Path
        Destination path.
    df : pd.DataFrame
        Labels DataFrame with required columns (see :data:`REQUIRED_COLUMNS`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    preferred = [
        "onset_s", "offset_s", "labels", "individual", "trial",
        "human_verified", "changepoint_corrected", "prediction_source", "n_samples",
    ]
    cols = [c for c in preferred if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    out = out[cols]

    tmp = path.with_suffix(".tsv.tmp")
    out.to_csv(tmp, sep="\t", index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _empty_all_labels() -> pd.DataFrame:
    df = empty_intervals()
    df.insert(0, "trial", pd.Series(dtype=object))
    for col, default in TRIAL_META_DEFAULTS.items():
        df[col] = pd.Series(dtype=type(default))
    return df




# ---------------------------------------------------------------------------
# Per-trial access
# ---------------------------------------------------------------------------

def get_trial_from_tsv(all_df: pd.DataFrame, trial) -> pd.DataFrame:
    """Extract intervals for a single trial from the all-labels DataFrame.

    Returns a DataFrame with standard interval columns (``onset_s``, ``offset_s``,
    ``labels``, ``individual``), without the ``trial`` column.
    """
    if all_df is None or all_df.empty:
        return empty_intervals()
    mask = all_df["trial"] == trial
    trial_df = all_df.loc[mask, INTERVAL_COLUMNS].reset_index(drop=True)
    if trial_df.empty:
        return empty_intervals()
    return trial_df


def set_trial_in_tsv(
    all_df: pd.DataFrame, trial, trial_df: pd.DataFrame,
) -> pd.DataFrame:
    """Replace all rows for a trial in the all-labels DataFrame.

    Preserves per-trial metadata columns from the existing rows.
    """
    if all_df is None:
        all_df = _empty_all_labels()

    # Preserve existing meta values for this trial
    old_meta = get_trial_meta(all_df, trial)
    other = all_df[all_df["trial"] != trial]

    new_rows = trial_df[INTERVAL_COLUMNS].copy()
    new_rows.insert(0, "trial", trial)
    for col, default in TRIAL_META_DEFAULTS.items():
        new_rows[col] = old_meta.get(col, default)

    result = pd.concat([other, new_rows], ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Per-trial metadata (stored as columns, same value per trial)
# ---------------------------------------------------------------------------

def get_trial_meta(all_df: pd.DataFrame, trial) -> dict:
    """Read per-trial metadata from columns. Returns dict with defaults for missing trials."""
    if all_df is None or all_df.empty:
        return dict(TRIAL_META_DEFAULTS)
    mask = all_df["trial"] == trial
    rows = all_df.loc[mask]
    if rows.empty:
        return dict(TRIAL_META_DEFAULTS)
    first = rows.iloc[0]
    result = {}
    for col, default in TRIAL_META_DEFAULTS.items():
        val = first.get(col, default)
        if col == "prediction_source":
            result[col] = str(val) if pd.notna(val) else ""
        else:
            result[col] = int(val) if pd.notna(val) else default
    return result


def set_trial_meta_attr(all_df: pd.DataFrame, trial, key: str, value) -> pd.DataFrame:
    """Set a per-trial metadata column value for all rows of a trial."""
    if all_df is None:
        return all_df
    mask = all_df["trial"] == trial
    if mask.any():
        all_df.loc[mask, key] = value
    return all_df


def init_empty_labels(trials: list) -> pd.DataFrame:
    """Create empty labels DataFrame."""
    return _empty_all_labels()
