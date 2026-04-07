"""Backend builder: assembles a TrialTree from WizardState (no Qt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from movement.io import load_dataset
from natsort import natsorted

from ethograph.gui.wizard_media_files import extract_file_row
from ethograph.gui.wizard_overview import ModalityConfig, WizardState
from ethograph.labels.intervals import INTERVAL_COLUMNS
from ethograph.io.trialtree import TrialTree

INTERVAL_COLUMNS = {"trial", "onset_s", "offset_s", "labels", "individual"}


def build_multi_trial_dt(state: WizardState) -> TrialTree:
    trial_table = state.trial_table
    if trial_table is None or trial_table.empty:
        raise ValueError("No trial table available. Go back and configure trials.")

    trial_ids = trial_table["trial"].tolist()
    datasets: list[xr.Dataset] = []

    individuals = state.individuals or ["individual_1"]

    if state.video.enabled:
        fps = state.video.fps
    elif state.pose.enabled:
        fps = state.pose.fps
    else:
        raise ValueError("FPS could not be detected from user/video.")

    for i, trial_id in enumerate(trial_ids):
        ds = _build_single_trial_ds(state, trial_table, i, trial_id, fps, individuals)
        datasets.append(ds)

    dt = TrialTree.from_datasets(datasets, validate=True)

    # Build NWB file with trials table + acquisition items
    nwb_path = _build_nwb_file(dt, state, trial_table, trial_ids, fps)
    if nwb_path:
        from ethograph.io.nwb_alignment import make_nwb_alignment
        state.nwb_alignment = make_nwb_alignment(nwb_path)

    return dt


def _build_single_trial_ds(
    state: WizardState,
    trial_table: pd.DataFrame,
    trial_idx: int,
    trial_id,
    fps: int,
    individuals: list[str],
) -> xr.Dataset:
    row = trial_table.iloc[trial_idx]

    ds = xr.Dataset(coords={"individuals": individuals})
    ds.attrs["trial"] = trial_id
    ds.attrs["fps"] = fps

    if state.pose.enabled:
        pose_path = _get_file_for_trial(row, "pose")
        if pose_path:
            ds = _load_pose_into_ds(ds, pose_path, state.pose)

    for var in list(ds.data_vars):
        if var not in INTERVAL_COLUMNS and var != "confidence":
            ds[var].attrs["type"] = "features"

    return ds


def _get_file_for_trial(row: pd.Series, modality: str) -> str | None:
    for col in row.index:
        if col.startswith(modality) and col != "trial":
            val = row[col]
            if pd.notna(val) and str(val):
                return str(val)
    return None


def _load_pose_into_ds(
    ds: xr.Dataset, pose_path: str, cfg: ModalityConfig,
) -> xr.Dataset:
    pose_ds = load_dataset(
        pose_path, source_software=cfg.source_software,
    )

    ds.attrs["source_software"] = cfg.source_software

    if "position" in pose_ds:
        time_coord = pose_ds["position"].coords[
            next(c for c in pose_ds["position"].coords if "time" in str(c))
        ]

        ds.coords["time"] = time_coord
        for var_name in pose_ds.data_vars:
            ds[var_name] = pose_ds[var_name]
        for coord_name in pose_ds.coords:
            if coord_name not in ds.coords:
                ds.coords[coord_name] = pose_ds.coords[coord_name]
    return ds


def _build_nwb_file(
    dt: TrialTree,
    state: WizardState,
    trial_table: pd.DataFrame,
    trial_ids: list,
    fps: float,
) -> Path | None:
    """Create an alignment NWB file from wizard state.

    Writes to ``.ethograph/alignment.nwb`` relative to the output path,
    or falls back to a temp location.
    """
    from ethograph.utils.nwb import build_nwb_from_trial_table

    # Build the NWB trial table with media columns
    nwb_df = trial_table.copy()

    # Ensure start_time and stop_time columns exist
    if "start_time" not in nwb_df.columns:
        nwb_df["start_time"] = 0.0
    if "stop_time" not in nwb_df.columns:
        nwb_df["stop_time"] = 1.0

    # Determine output path
    if state.output_path:
        output_dir = Path(state.output_path).parent
    else:
        output_dir = Path.cwd()

    ethograph_dir = output_dir / ".ethograph"
    nwb_path = ethograph_dir / "alignment.nwb"

    stream_rates: dict[str, float] = {}
    if fps:
        stream_rates["video"] = float(fps)
        stream_rates["pose"] = float(fps)

    build_nwb_from_trial_table(
        trial_table=nwb_df,
        stream_rates=stream_rates,
        output_path=nwb_path,
    )

    return nwb_path
