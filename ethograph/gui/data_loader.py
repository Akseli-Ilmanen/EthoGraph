"""Data loading utilities for the ethograph GUI."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import xarray as xr
from napari import current_viewer
from qtpy.QtWidgets import QMessageBox

from ethograph import TrialTree, set_media_attrs, minimal_basics
from ethograph.utils.validation import extract_type_vars, validate_datatree
from movement.io import load_poses, save_poses
from movement.kinematics import compute_acceleration, compute_pairwise_distances, compute_speed, compute_velocity



def _show_popup(message: str, title: str = "Load Error") -> None:
    parent = current_viewer().window._qt_window
    QMessageBox.warning(parent, title, message)

def load_dataset(file_path: str) -> Tuple[Optional[xr.Dataset], Optional[dict]]:
    """Load dataset from file path and cache metadata on the instance.

    Returns:
        Tuple of (dt, label_dt, type_vars_dict) on success.

    Raises:
        ValueError: On validation or format errors (popup shown before raising).
    """
    if Path(file_path).suffix != '.nc':
        msg = (
            f"Unsupported file type: {Path(file_path).suffix}. Expected .nc file.\n"
            "See documentation:\n"
            "https://movement.neuroinformatics.dev/user_guide/input_output.html#native-saving-and-loading-with-netcdf"
        )
        _show_popup(msg, title=".nc File Error")
        raise ValueError(msg)

    dt = TrialTree.open(file_path)

    label_dt = dt.get_label_dt()
    ds = dt.itrial(0)

    errors = validate_datatree(dt)
    if errors:
        error_msg = "\n".join(f"• {e}" for e in errors)
        suffix = "\n\nSee documentation: XXX"
        msg = "Validation failed:\n" + error_msg + suffix
        _show_popup(msg, title="Validation Error")
        raise ValueError(msg)

    type_vars_dict = extract_type_vars(ds, dt)

    return dt, label_dt, type_vars_dict



   
def minimal_dt_from_pose(video_path, fps, pose_path, source_software):
    """Create a minimal TrialTree from pose data.

    Args:
        video_path: Path to video file
        fps: Frames per second of the video
        pose_path: Path to pose file (e.g. poses.csv/poses.h5)
        source_software: Software used for pose estimation (e.g., 'DeepLabCut')

    Returns:
        TrialTree with minimal structure
    """
    try:
        ds = load_poses.from_file(
            file_path=pose_path,
            fps=fps,
            source_software=source_software,
        )
    except Exception:
        # Fallback
        df = pd.read_hdf(pose_path)
        pose_path = Path(pose_path).with_suffix(".csv")
        ds = load_poses.from_dlc_style_df(df, fps=fps)
        save_poses.to_dlc_file(ds, str(pose_path))
        ds.attrs["source_software"] = source_software


    ds["velocity"] = compute_velocity(ds.position)
    ds["speed"] = compute_speed(ds.position)
    ds["acceleration"] = compute_acceleration(ds.position)
    
    if len(ds.keypoints) > 1:
        compute_pairwise_distances(ds.position, dim='keypoints', pairs='all')
    
    if len(ds.individuals) > 1:
        # Not sure how this looks like with individuals > 2
        compute_pairwise_distances(ds.position, dim='individuals', pairs='all')
    

    ds = set_media_attrs(
        ds,
        cameras=[Path(video_path).name],
        pose=[Path(pose_path).name],
    )
    dt = minimal_basics(ds, video_motion=False) # Kinematics -> no video motion needed


    return dt


def minimal_dt_from_ds(video_path, ds: xr.Dataset):
    
    ds = set_media_attrs(
        ds,
        cameras=[Path(video_path).name],
    )  
    dt = minimal_basics(ds)
    
    return dt


def minimal_dt_from_npy_file(video_path, fps, npy_path, data_sr, individuals=None, video_motion: bool = False):

    if individuals is None:
        individuals = ["individual 1", "individual 2", "individual 3", "individual 4"]

    data = np.load(npy_path)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_variables = data.shape

    # Assume longer dimension is time
    if n_samples < n_variables:
        data = data.T
        n_samples, n_variables = data.shape

    time_coords = np.arange(n_samples) / data_sr

    ds = xr.Dataset(
        data_vars={
            "data": (["time", "variable"], data)
        },
        coords={
            "time": time_coords,
            "individuals": individuals  
        }
    )    
    
    ds.attrs["fps"] = fps

    
    ds = set_media_attrs(
        ds,
        cameras=[Path(video_path).name],
    )  
    
    
    dt = minimal_basics(ds, video_path=video_path, video_motion=video_motion)
    
    return dt




def minimal_dt_from_ephys(
    video_path, fps, ephys_path, individuals=None,
    video_motion: bool = False, trial_onset: float | None = None,
):
    if individuals is None:
        individuals = ["individual 1", "individual 2", "individual 3", "individual 4"]

    from ethograph.gui.dialog_create_nc import get_video_n_frames

    n_frames = get_video_n_frames(video_path)
    if n_frames is None:
        raise ValueError(f"Could not determine frame count from video: {video_path}")

    ds.attrs["fps"] = fps
    
    # Ephys & video alignment
    if trial_onset is not None:
        ds.attrs["trial_onset"] = float(trial_onset)

    ds = set_media_attrs(
        ds,
        cameras=[Path(video_path).name],
    )
    

    dt = minimal_basics(ds, video_path=video_path, video_motion=video_motion)

    return dt


def minimal_dt_from_audio(video_path, fps, audio_path, individuals=None, video_motion: bool = False):

    if individuals is None:
        individuals = ["individual 1", "individual 2", "individual 3", "individual 4"]


    ds = xr.Dataset(
        coords={
            "individuals": individuals  
        }
    )    
    ds.attrs["fps"] = fps

    
    ds = set_media_attrs(
        ds,
        cameras=[Path(video_path).name],
        mics=[Path(audio_path).name],
    )  
    
    dt = minimal_basics(ds, video_path=video_path, video_motion=video_motion)
    
    return dt

