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
from ethograph.utils.io import dataset_to_basic_trialtree
from ethograph.utils.label_intervals import empty_intervals, intervals_to_xr
from ethograph.utils.trialtree import TrialTree

INTERVAL_COLUMNS = {"onset_s", "offset_s", "labels", "individual"}


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

    # Set media files
    _set_media(dt, state, trial_table, trial_ids)

    # Set stream offsets
    _set_stream_offsets(dt, state)

    # Set session table with start/stop times if provided
    if not state.files_aligned_to_trials:
        _set_session_timing(dt, state, trial_table, trial_ids)

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

    # Load pose data if enabled
    if state.pose.enabled:
        pose_path = _get_file_for_trial(row, "pose")
        if pose_path:
            ds = _load_pose_into_ds(ds, pose_path, state.pose)

    # Ensure labels exist
    if "onset_s" not in ds.data_vars:
        interval_ds = intervals_to_xr(empty_intervals())
        for var_name in interval_ds.data_vars:
            ds[var_name] = interval_ds[var_name]

    # Tag all data variables (except labels and confidence) as features
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
    
    # Merge pose variables into trial ds
    if "position" in pose_ds:
        time_coord = pose_ds["position"].coords[
            next(c for c in pose_ds["position"].coords if "time" in str(c))
        ]
        
        ds.coords["time"] = time_coord
        for var_name in pose_ds.data_vars:
            ds[var_name] = pose_ds[var_name]
        # Copy coords
        for coord_name in pose_ds.coords:
            if coord_name not in ds.coords:
                ds.coords[coord_name] = pose_ds.coords[coord_name]
    return ds



def _set_media(
    dt: TrialTree,
    state: WizardState,
    trial_table: pd.DataFrame,
    trial_ids: list,
):
    if "trial" not in trial_table.columns:
        raise ValueError("trial_table must contain a 'trial' column.")

    trial_order = natsorted([str(t) for t in dt.trials])
    trial_indexed = trial_table.copy()
    trial_indexed["trial"] = trial_indexed["trial"].map(lambda x: str(x).strip())
    trial_indexed = trial_indexed.set_index("trial", drop=False)

    if trial_indexed.index.duplicated().any():
        duplicate_trials = trial_indexed.index[trial_indexed.index.duplicated()].tolist()
        raise ValueError(f"Duplicate trial ids found in trial_table: {duplicate_trials}")

    missing_trials = [t for t in trial_order if t not in trial_indexed.index]
    if missing_trials:
        raise ValueError(
            "trial_table is missing trials present in dt: "
            f"{missing_trials}."
        )

    trial_rows = trial_indexed.loc[trial_order]

    session = xr.Dataset()
    
    # Determine per_trial mode: True if any enabled modality is aligned (aligned_to_trial)



    # Collect video files
    video_files = None
    cameras = state.camera_names or None
    if state.video.enabled:
        video_cols = [c for c in trial_table.columns if c.startswith("video_")]
        if video_cols:
            video_files = []
            for _, row in trial_rows.iterrows():
                trial_videos = [str(row[c]) if pd.notna(row[c]) else "" for c in video_cols]
                video_files.append(trial_videos)
            if cameras is None:
                cameras = [c.replace("video_", "") for c in video_cols]
            
            # keep only rows where at least one camera file exists
            valid_rows = [row for row in video_files if any(v != "" for v in row)]
            
            if len(valid_rows) > 1:
                session["video"] = xr.DataArray(
                    video_files,
                    dims=["trial", "cameras"],
                    coords={"trial": trial_order, "cameras": cameras}
                )
            else:
                session["video"] = xr.DataArray(
                    video_files[0], # Just 1 valid row of trials
                    dims=["cameras"],
                    coords={"cameras": cameras}
                )
                
                
        elif state.video.single_file_path:            
            session["video"] = xr.DataArray(
                [Path(state.video.single_file_path).name],  
                dims=["cameras"],
                coords={"cameras": cameras}  
            )



    # Collect pose files
    pose_files = None
    if state.pose.enabled:
        pose_cols = [c for c in trial_table.columns if c.startswith("pose_")]
        if pose_cols:
            pose_files = []
            for _, row in trial_rows.iterrows():
                trial_poses = [str(row[c]) if pd.notna(row[c]) else "" for c in pose_cols]
                pose_files.append(trial_poses)

            valid_rows = [row for row in pose_files if any(v != "" for v in row)]
            
            if len(valid_rows) > 1:
                session["pose"] = xr.DataArray(
                    pose_files,
                    dims=["trial", "cameras"],
                    coords={"trial": trial_order, "cameras": cameras}
                )
            else:
                session["pose"] = xr.DataArray(
                    pose_files[0],
                    dims=["cameras"],
                    coords={"cameras": cameras}
                )
                    
        elif state.pose.single_file_path:            
            session["pose"] = xr.DataArray(
                [Path(state.pose.single_file_path).name],  
                dims=["cameras"],
                coords={"cameras": cameras}  
            )
            
            

    # Collect audio files
    audio_files = None
    mics = state.mic_names or None
    if state.audio.enabled:
        audio_cols = [c for c in trial_table.columns if c.startswith("audio_")]
        if audio_cols:
            audio_files = []
            for _, row in trial_rows.iterrows():
                trial_audios = [str(row[c]) if pd.notna(row[c]) else "" for c in audio_cols]
                audio_files.append(trial_audios)
            if mics is None:
                mics = [c.replace("audio_", "") for c in audio_cols]
                
            valid_rows = [row for row in audio_files if any(v != "" for v in row)]

            if len(valid_rows) > 1:
                session["audio"] = xr.DataArray(
                    audio_files,
                    dims=["trial", "mics"],
                    coords={"trial": trial_order, "mics": mics}
                )
            else:
                session["audio"] = xr.DataArray(
                    audio_files[0],
                    dims=["mics"],
                    coords={"mics": mics}
                )
                    
        elif state.audio.single_file_path:            
            session["audio"] = xr.DataArray(
                [Path(state.audio.single_file_path).name],  
                dims=["mics"],
                coords={"mics": mics}  
            )
    

    dt["session"] = xr.DataTree(session)




def _set_stream_offsets(
    dt: TrialTree,
    state: WizardState,
):
    for name, stream in [
        ("video", "video"),
        ("pose", "pose"),
        ("audio", "audio"),
        ("ephys", "ephys"),
    ]:
        cfg: ModalityConfig = getattr(state, name)
        if not cfg.enabled:
            continue
        
        if cfg.file_mode == "aligned_to_session":
            # Continuous mode: use configured constant or per-device offsets.
            if cfg.offset_constant_across_devices and cfg.constant_offset != 0.0:
                # Constant offset across all devices
                dt.set_stream_offset(stream, cfg.constant_offset)
            elif cfg.device_offsets:
                # Per-device offsets stored as session variables
                if dt.session is not None:
                    session_ds = dt["session"].to_dataset()
                else:
                    session_ds = xr.Dataset()
                
                for device, offset in cfg.device_offsets.items():
                    if offset != 0.0:
                        session_ds[f"offset_{stream}_{device}"] = xr.DataArray(
                            offset, attrs={"units": "seconds"}
                        )
                
                if session_ds.data_vars:
                    dt["session"] = xr.DataTree(session_ds)
        else:
            # Aligned mode
            if cfg.constant_offset != 0.0:
                dt.set_stream_offset(stream, cfg.constant_offset)


def _set_session_timing(
    dt: TrialTree,
    state: WizardState,
    trial_table: pd.DataFrame,
    trial_ids: list,
):
    def _coerce_numeric_like(value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return value
            try:
                number = float(stripped)
            except ValueError:
                return value
            return int(number) if number.is_integer() else number
        return value

    session_df = trial_table.copy()

    if "trial" not in session_df.columns:
        session_df["trial"] = trial_ids

    for col in session_df.columns:
        if col in {"start_time", "stop_time"}:
            session_df[col] = pd.to_numeric(session_df[col], errors="coerce").astype(float)
        else:
            session_df[col] = session_df[col].map(_coerce_numeric_like)

    session_df = session_df.set_index("trial")
    table_ds = xr.Dataset.from_dataframe(session_df)

    if dt.session is not None:
        session_ds = dt["session"].to_dataset()
    else:
        session_ds = xr.Dataset()

    for var_name in table_ds.data_vars:
        session_ds[var_name] = table_ds[var_name]

    if "trial" in table_ds.coords:
        session_ds = session_ds.assign_coords(trial=table_ds.coords["trial"])

    dt["session"] = xr.DataTree(session_ds)
