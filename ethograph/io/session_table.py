"""Session table: TSV-based single source of truth for per-trial metadata.

The session table is a plain TSV file at ``.ethograph/session_table.tsv``.
One row per trial.  Standard columns (``trial``, ``start_time``, ``stop_time``)
plus media columns (``video_cam-1``, ``audio_mic-1``, …) and arbitrary
user-defined condition columns (``genotype``, ``treatment``, …).

The alignment NWB file is a *derived artifact* rebuilt from this table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SETTINGS_DIR = ".ethograph"
_SESSION_TABLE_FILENAME = "session_table.tsv"

# Columns that are structural (not user conditions)
_STRUCTURAL_COLUMNS = frozenset({
    "trial",
    "start_time",
    "stop_time",
})

_MEDIA_PREFIXES = ("video_", "pose_", "audio_", "ephys_")


def session_table_path(nc_path: str | Path) -> Path:
    """Return the session table TSV path for a dataset."""
    return Path(nc_path).resolve().parent / _SETTINGS_DIR / _SESSION_TABLE_FILENAME


def is_media_column(col: str) -> bool:
    """True if column follows the ``{stream}_{device}`` naming convention."""
    return any(col.startswith(p) for p in _MEDIA_PREFIXES)


def condition_columns(df: pd.DataFrame) -> list[str]:
    """Return user-defined condition column names (not structural, not media)."""
    return [
        c for c in df.columns
        if c not in _STRUCTURAL_COLUMNS and not is_media_column(c)
    ]


def load_session_table(path: str | Path) -> pd.DataFrame:
    """Load a session table TSV file."""
    df = pd.read_csv(path, sep="\t")
    if "trial" not in df.columns:
        raise ValueError(f"Session table {path} missing required 'trial' column")
    return df


def save_session_table(path: str | Path, df: pd.DataFrame) -> None:
    """Save a session table TSV file (atomic write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tsv.tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)
    logger.info("Saved session table to %s", path.name)


def inject_metadata(dt: "TrialTree", df: pd.DataFrame) -> None:
    """Inject session table condition columns into per-trial ds.attrs.

    Only injects user-defined condition columns (not structural or media).
    This makes them available for ``filter_by_attr()`` and
    ``_possible_trial_conditions()`` without any changes to those functions.
    """
    cond_cols = condition_columns(df)
    if not cond_cols:
        return

    for trial_id, ds in dt.trial_items():
        row = df[df["trial"] == trial_id]
        if row.empty:
            row = df[df["trial"] == str(trial_id)]
        if row.empty and isinstance(trial_id, (int, float)):
            row = df[df["trial"].astype(str) == str(int(trial_id))]
        if row.empty:
            continue
        row = row.iloc[0]
        for col in cond_cols:
            val = row[col]
            if pd.notna(val):
                ds.attrs[col] = val


def extract_from_nwb(nwb_path: str | Path) -> pd.DataFrame:
    """Extract a session table DataFrame from an existing alignment.nwb.

    Reads the NWB trials table and converts to the session table format.
    """
    from ethograph.io.session_io import NWBSessionIO

    sio = NWBSessionIO(nwb_path)
    df = sio.trials_df.copy()

    # NWB trials table uses a numeric index; reset to flat columns
    if df.index.name == "id":
        df = df.reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Drop NWB internal columns that aren't useful in the TSV
    for col in ["stop_time_is_placeholder"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    sio.close()
    return df


def rebuild_alignment_nwb(
    session_df: pd.DataFrame,
    nwb_path: str | Path,
    camera_fps: float = 30.0,
) -> None:
    """Rebuild alignment.nwb from a session table DataFrame.

    Delegates to :func:`build_nwb_from_trial_table`.
    """
    from ethograph.utils.nwb import build_nwb_from_trial_table

    build_nwb_from_trial_table(
        session_df,
        camera_fps=camera_fps,
        output_path=Path(nwb_path),
    )
    logger.info("Rebuilt alignment NWB from session table")


def ensure_session_table(nc_path: str | Path) -> Path | None:
    """Ensure a session table exists, migrating from NWB if needed.

    Returns the session table path if one exists or was created, else None.
    """
    from ethograph.io.trialtree import _discover_nwb

    tsv_path = session_table_path(nc_path)
    if tsv_path.exists():
        return tsv_path

    # Migrate from existing NWB
    nwb_path = _discover_nwb(nc_path)
    if nwb_path is not None and nwb_path.exists():
        df = extract_from_nwb(nwb_path)
        save_session_table(tsv_path, df)
        logger.info("Migrated session table from %s", nwb_path.name)
        return tsv_path

    return None


def load_and_sync(
    nc_path: str | Path,
    dt: "TrialTree",
    camera_fps: float = 30.0,
) -> pd.DataFrame | None:
    """Load session table, inject metadata into dt, rebuild NWB if stale.

    This is the main entry point called from ``data_loader.load_dataset()``.

    Returns the session table DataFrame, or None if no session table exists.
    """
    tsv_path = ensure_session_table(nc_path)
    if tsv_path is None:
        return None

    df = load_session_table(tsv_path)
    inject_metadata(dt, df)

    # Rebuild NWB if session table is newer
    nwb_dir = tsv_path.parent
    nwb_path = nwb_dir / "alignment.nwb"
    if not nwb_path.exists() or tsv_path.stat().st_mtime > nwb_path.stat().st_mtime:
        rebuild_alignment_nwb(df, nwb_path, camera_fps=camera_fps)
        # Re-point dt at the (re)built NWB
        dt.nwb_path = str(nwb_path)

    return df
