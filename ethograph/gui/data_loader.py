"""Data loading utilities for the ethograph GUI."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from napari import current_viewer
from qtpy.QtWidgets import QMessageBox

from ethograph import TrialTree, set_media_attrs, minimal_basics
from ethograph.features.audio_features import get_envelope
from ethograph.utils.validation import extract_type_vars, validate_datatree
from ethograph.utils.audio import get_audio_sr
from movement.io import load_poses
from movement.kinematics import compute_acceleration, compute_pairwise_distances, compute_speed, compute_velocity



def show_error_dialog(message: str, title: str = ".nc File Error") -> None:
    QMessageBox.critical(current_viewer().window._qt_window, title, message)

def load_dataset(file_path: str) -> Tuple[Optional[xr.Dataset], Optional[dict]]:
    """Load dataset from file path and cache metadata on the instance.

    Returns:
        Tuple of (dt, label_dt, type_vars_dict) or (None, None, None) on error
    """
    if Path(file_path).suffix != '.nc':
        error_msg = (
            f"Unsupported file type: {Path(file_path).suffix}. Expected .nc file.\n"
            "See documentation:\n"
            "https://movement.neuroinformatics.dev/user_guide/input_output.html#native-saving-and-loading-with-netcdf"
        )
        raise ValueError(error_msg)

    dt = TrialTree.open(file_path)
    
    
    label_dt = dt.get_label_dt()
    ds = dt.itrial(0)


    errors = validate_datatree(dt)
    if errors:
        error_msg = "\n".join(f"• {e}" for e in errors)
        
        suffix = "\n\n See documentation: XXX"
        
        # Display twice to ensure visibility in napari
        show_error_dialog("Validation failed: \n" + error_msg + suffix)
        raise ValueError("Validation failed: \n" + error_msg + suffix)


    # Build type_vars_dict from first trial
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
    ds = load_poses.from_file(
        file_path=pose_path,
        fps=fps,
        source_software=source_software
    )


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




def minimal_dt_from_audio(video_path, fps, audio_path, audio_sr, individuals=None, video_motion: bool = False):

    if individuals is None:
        individuals = ["individual 1", "individual 2", "individual 3", "individual 4"]

    
    envelope, gen_wav_path = get_envelope(audio_path, audio_sr, fps)

    if gen_wav_path:
        audio_path = gen_wav_path
    
    n_frames = len(envelope)
    time_coords = np.arange(n_frames) / fps
    

    ds = xr.Dataset(
        data_vars={
            "labels": (["time", "individuals"], np.zeros((n_frames, len(individuals))))
        },
        coords={
            "time": time_coords,
            "individuals": individuals  
        }
    )    
    
    if envelope.ndim == 1:
        ds["audio_envelope"] = (["time"], envelope)
    elif envelope.ndim == 2:
        ds["audio_envelope"] = (["time", "channels"], envelope)
    else:
        raise ValueError("Envelope must be 1D or 2D array")
                


    ds.attrs["fps"] = fps

    
    ds = set_media_attrs(
        ds,
        cameras=[Path(video_path).name],
        mics=[Path(audio_path).name],
    )  
    
    dt = minimal_basics(ds, video_path=video_path, video_motion=video_motion)
    
    return dt

