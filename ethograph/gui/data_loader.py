"""Data loading utilities for the ethograph GUI."""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import ethograph as eto
from ethograph.utils.validation import extract_type_vars, validate_datatree
from movement.io import load_poses, save_poses
from movement.io import load
from movement.kinematics import compute_acceleration, compute_pairwise_distances, compute_speed, compute_velocity



def _show_popup(message: str, title: str = "Load Error") -> None:
    print(f"[{title}] {message}", flush=True)

def load_dataset(
    file_path: str,
    require_fps: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    max_trials: int | None = None,
    dandiset_id: str | None = None,
) -> Tuple[Optional[xr.Dataset], Optional[dict]]:
    """Load dataset from file path and cache metadata on the instance.

    Args:
        file_path: Path to the .nc file.
        require_fps: When False, missing fps is not an error (audio-only mode).
        progress_callback: Called with status strings during slow loading steps.
        max_trials: If set, limit NWB loading to the first N trials.

    Returns:
        Tuple of (dt, label_dt, type_vars_dict) on success.

    Raises:
        ValueError: On validation or format errors (popup shown before raising).
    """

    dt = eto.open(file_path)
    label_dt = dt.get_label_dt()
    type_vars_dict = extract_type_vars(dt.itrial(0), dt)


    errors = validate_datatree(
        dt, require_fps=require_fps,
    )
    if errors:
        error_msg = "\n".join(f"• {e}" for e in errors)
        suffix_msg = "\n\nSee documentation: XXX"
        msg = "Validation failed:\n" + error_msg + suffix_msg
        _show_popup(msg, title="Validation Error")
        raise ValueError(msg)

    return dt, label_dt, type_vars_dict


def _wizard_single_media_helper(
    dt,
    video_path=None,
    pose_path=None,
    audio_path=None,
):
    data_vars = {}
    coords = {"trial": [1]}

    if video_path is not None:
        data_vars["video"] = (["cameras"], [Path(video_path).name])
        coords["cameras"] = ["cam-1"]

    if pose_path is not None:
        data_vars["pose"] = (["cameras"], [Path(pose_path).name])
        coords["cameras"] = ["cam-1"]

    if audio_path is not None:
        data_vars["audio"] = (["mics"], [Path(audio_path).name])
        coords["mics"] = ["mic-1"]


    session = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
    )

    dt["session"] = xr.DataTree(session)
    return dt

   
def wizard_single_from_pose(
    video_path,
    fps,
    pose_path,
    source_software,
    video_offset: float | None = None,
):
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
        ds = load.load_dataset(
            pose_path,
            fps=fps,
            source_software=source_software,
        )
    except (OSError, ValueError, KeyError):
        _show_popup(f"Failed to load pose data from {pose_path}. Please check the file and try again.", title="Pose Load Error")
        raise


    ds["velocity"] = compute_velocity(ds.position)
    ds["speed"] = compute_speed(ds.position)
    ds["acceleration"] = compute_acceleration(ds.position)
    
    if len(ds.keypoints) > 1:
        compute_pairwise_distances(ds.position, dim='keypoints', pairs='all')
    
    if len(ds.individuals) > 1:
        # Not sure how this looks like with individuals > 2
        compute_pairwise_distances(ds.position, dim='individuals', pairs='all')
    

    dt = eto.dataset_to_basic_trialtree(ds, video_motion=False)
    _wizard_single_media_helper(dt, video_path=video_path, pose_path=pose_path)
    if video_offset is not None:
        dt.set_stream_offset("video", float(video_offset))
    
    return dt


def wizard_single_from_ds(video_path, ds: xr.Dataset, video_offset: float | None = None):
    dt = eto.dataset_to_basic_trialtree(ds)
    _wizard_single_media_helper(dt, video_path=video_path)
    if video_offset is not None:
        dt.set_stream_offset("video", float(video_offset))
    return dt


def wizard_single_from_npy_file(
    video_path,
    fps,
    npy_path,
    data_sr,
    individuals=None,
    video_motion: bool = False,
    video_offset: float | None = None,
):

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

    
    dt = eto.dataset_to_basic_trialtree(ds, video_path=video_path, video_motion=video_motion)
    _wizard_single_media_helper(dt, video_path=video_path)
    if video_offset is not None:
        dt.set_stream_offset("video", float(video_offset))
    return dt




def wizard_single_from_ephys(
    video_path: str | None = None,
    fps: int = 30,
    audio_path: str | None = None,
    individuals: list[str] | None = None,
    video_motion: bool = False,
    video_offset: float | None = None,
    audio_offset: float | None = None,
):
    if individuals is None:
        individuals = ["individual 1", "individual 2", "individual 3", "individual 4"]

    ds = xr.Dataset(coords={"individuals": individuals})
    ds.attrs["fps"] = fps



    dt = eto.dataset_to_basic_trialtree(ds, video_path=video_path, video_motion=video_motion)
    _wizard_single_media_helper(dt, video_path=video_path, audio_path=audio_path)
    
    if video_offset is not None:
        dt.set_stream_offset("video", float(video_offset))
    
    if audio_offset is not None:
        dt.set_stream_offset("audio", float(audio_offset))

    return dt


def wizard_single_from_audio(
    video_path,
    fps,
    audio_path,
    individuals=None,
    video_motion: bool = False,
    audio_sr: int = 44100,
    video_offset: float | None = None,
):

    if individuals is None:
        individuals = ["individual 1", "individual 2", "individual 3", "individual 4"]

    ds = xr.Dataset(
        coords={
            "individuals": individuals
        }
    )
    ds.attrs["fps"] = fps

    dt = eto.dataset_to_basic_trialtree(ds, video_path=video_path, video_motion=video_motion)
    _wizard_single_media_helper(dt, video_path=video_path, audio_path=audio_path)
    if video_offset is not None:
        dt.set_stream_offset("video", float(video_offset))
    return dt

