"""Trial metadata table: TSV-based per-trial condition metadata.

The metadata table is a plain TSV file at ``{nc_stem}_metadata.tsv``,
one row per trial.  Required column: ``trial`` (join key).  All other
columns are user-defined conditions (genotype, treatment, etc.).

Three source scenarios (checked in priority order):
1. TSV file exists → editable, user-owned
2. NWB trials table has metadata columns → read-only
3. Legacy ds.attrs have conditions → one-time migration to TSV
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STREAM_COL_RE = re.compile(r"^(video|pose|audio|ephys)_(.+)$")

_NWB_STRUCTURAL_COLUMNS = frozenset({
    "trial",
    "start_time",
    "stop_time",
})

_LEGACY_SKIP_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".wav", ".flac", ".mp3", ".ogg",
    ".csv", ".h5", ".hdf5", ".npy", ".slp", ".nwb",
    ".dat", ".bin", ".raw", ".mda",
})


def metadata_tsv_path(nc_path: str | Path) -> Path:
    """Derive the metadata TSV path from a dataset file path."""
    p = Path(nc_path).resolve()
    return p.parent / f"{p.stem}_metadata.tsv"


def load_metadata_tsv(path: str | Path) -> pd.DataFrame:
    """Load a metadata TSV file."""
    df = pd.read_csv(path, sep="\t")
    if "trial" not in df.columns:
        raise ValueError(f"Metadata table {path} missing required 'trial' column")
    return df


def save_metadata_tsv(path: str | Path, df: pd.DataFrame) -> None:
    """Save a metadata TSV file (atomic write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tsv.tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)
    logger.info("Saved metadata table to %s", path.name)


def condition_columns(df: pd.DataFrame) -> list[str]:
    """Return user-defined condition column names (everything except ``trial``)."""
    return [c for c in df.columns if c != "trial"]


def _is_nwb_infrastructure_col(col: str) -> bool:
    """True if column is structural (timing, media, offsets) rather than metadata."""
    if col in _NWB_STRUCTURAL_COLUMNS:
        return True
    if _STREAM_COL_RE.match(col):
        return True
    if col.endswith("_start"):
        return True
    return False


def metadata_from_attrs(dt: "TrialTree") -> pd.DataFrame:
    """Extract condition metadata from legacy ds.attrs across all trials.

    Scans each trial's ``ds.attrs`` for keys that look like conditions
    (not ``trial``, not common across all trials, not file paths).
    """
    common_attrs = dt.get_common_attrs().keys()

    rows = []
    for trial_id, ds in dt.trial_items():
        row: dict = {"trial": trial_id}
        for key, value in ds.attrs.items():
            if key == "trial" or key in common_attrs:
                continue
            if isinstance(value, str) and Path(value).suffix.lower() in _LEGACY_SKIP_EXTENSIONS:
                continue
            row[key] = value
        rows.append(row)

    if not rows:
        return pd.DataFrame({"trial": dt.trials})

    return pd.DataFrame(rows)
