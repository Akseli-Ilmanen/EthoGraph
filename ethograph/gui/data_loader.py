"""Data loading utilities for the ethograph GUI."""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import ethograph as eto
<<<<<<< HEAD
from ethograph.utils.validation import extract_type_vars, validate_datatree
from movement.io import load_poses, save_poses
from movement.io import load
from movement.kinematics import compute_acceleration, compute_pairwise_distances, compute_speed, compute_velocity



def _show_popup(message: str, title: str = "Load Error") -> None:
    print(f"[{title}] {message}", flush=True)
=======
from ethograph.io.validation import extract_type_vars, validate_datatree
from movement.io import load, load_poses, save_poses
from movement.kinematics import compute_acceleration, compute_pairwise_distances, compute_speed, compute_velocity

from ethograph.gui.notify import notify_dialog
from ethograph.io.trialtree import TrialTree
from ethograph.labels.tsv_store import (
    init_empty_labels,
    labels_tsv_path,
    load_labels_tsv,
    save_labels_tsv,
)

import logging

logger = logging.getLogger(__name__)

>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

def load_dataset(
    file_path: str,
    require_fps: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    max_trials: int | None = None,
    dandiset_id: str | None = None,
<<<<<<< HEAD
) -> Tuple[Optional[xr.Dataset], Optional[dict]]:
    """Load dataset from file path and cache metadata on the instance.

    Args:
        file_path: Path to the .nc file.
        require_fps: When False, missing fps is not an error (audio-only mode).
        progress_callback: Called with status strings during slow loading steps.
        max_trials: If set, limit NWB loading to the first N trials.

    Returns:
        Tuple of (dt, label_dt, type_vars_dict) on success.
=======
    import_labels: bool = True,
) -> tuple:
    """Load dataset from file path.

    Returns:
        Tuple of (dt, all_labels_df, type_vars_dict) on success.
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

    Raises:
        ValueError: On validation or format errors (popup shown before raising).
    """
<<<<<<< HEAD

    dt = eto.open(file_path)
    label_dt = dt.get_label_dt()
    type_vars_dict = extract_type_vars(dt.itrial(0), dt)


    errors = validate_datatree(
        dt, require_fps=require_fps,
    )
=======
    dt = eto.open(file_path)
    type_vars_dict = extract_type_vars(dt.itrial(0), dt)

    errors = validate_datatree(dt, require_fps=require_fps)
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    if errors:
        error_msg = "\n".join(f"• {e}" for e in errors)
        suffix_msg = "\n\nSee documentation: XXX"
        msg = "Validation failed:\n" + error_msg + suffix_msg
<<<<<<< HEAD
        _show_popup(msg, title="Validation Error")
        raise ValueError(msg)

    return dt, label_dt, type_vars_dict
=======
        notify_dialog(msg, "error", "Validation Error")
        raise ValueError(msg)

    nc_path = Path(file_path)
    tsv_path = labels_tsv_path(nc_path)

    if tsv_path.exists():
        all_labels_df = load_labels_tsv(tsv_path)
        logger.info("Loaded labels from %s", tsv_path.name)
    
    else:
        all_labels_df = init_empty_labels(dt.trials)

    return dt, all_labels_df, type_vars_dict
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955


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
<<<<<<< HEAD
=======

    fps = dt.itrial(0).attrs.get("fps")
    if fps is not None and "cameras" in coords:
        dt.set_video_fps(float(fps), device_labels=coords["cameras"])

>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
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
<<<<<<< HEAD
            file_path=pose_path,
=======
            pose_path,
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
            fps=fps,
            source_software=source_software,
        )
    except (OSError, ValueError, KeyError):
<<<<<<< HEAD
        # Fallback: try reading as HDF5 DLC-style DataFrame
        df = pd.read_hdf(pose_path)
        pose_path = Path(pose_path).with_suffix(".csv")
        ds = load_poses.from_dlc_style_df(df, fps=fps)
        save_poses.to_dlc_file(ds, str(pose_path))
        ds.attrs["source_software"] = source_software
=======
        notify_dialog(f"Failed to load pose data from {pose_path}. Please check the file and try again.", "error", "Pose Load Error")
        raise
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955


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

