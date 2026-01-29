"""Video utilities for extracting and exporting motif segments from labeled data."""

from pathlib import Path
from typing import Optional

import numpy as np

from ethograph.utils.io import TrialTree
from ethograph.utils.labels import get_segment_indices


# IMPORTANT: If I want to use this code, need to make some fixes, replaces data_sr with label_sr where appropriate.

# def export_motif_clips(
#     nc_path: str,
#     video_path: str,
#     output_dir: Optional[str] = None,
#     individuals_sel: Optional[str] = None,
#     trial_sel: Optional[int] = None,
#     motif_ids: Optional[list[int]] = None,
#     padding_frames: int = 0,
#     slowdown_factor: float = 1.0,
# ) -> list[Path]:
#     """Export each continuous motif segment from a labeled dataset as a separate .mp4 file.

#     Args:
#         nc_path: Path to the NetCDF file containing labels.
#         video_path: Path to the source video file (.mp4).
#         output_dir: Directory for exported clips. Defaults to same directory as video.
#         individuals_sel: Individual to select from labels (if multiple individuals).
#         trial_sel: Trial number to process. If None, processes first trial.
#         motif_ids: List of motif IDs to export. If None, exports all non-zero motifs.
#         padding_frames: Number of frames to add before/after each clip.
#         slowdown_factor: Factor to slow down playback. 2.0 = half speed, 4.0 = quarter speed.

#     Returns:
#         List of paths to exported video clips.

#     Example:
#         >>> export_motif_clips(
#         ...     nc_path="session.nc",
#         ...     video_path="video.mp4",
#         ...     output_dir="clips/",
#         ...     motif_ids=[2, 3],  # Only export motifs 2 and 3
#         ...     slowdown_factor=2.0,  # Play at half speed
#         ... )
#     """
#     try:
#         import cv2
#     except ImportError as e:
#         raise ImportError("opencv-python is required: pip install opencv-python") from e

#     nc_path = Path(nc_path)
#     video_path = Path(video_path)
#     output_dir = Path(output_dir) if output_dir else video_path.parent / "motif_clips"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     dt = TrialTree.open(str(nc_path))
#     label_dt = dt.get_label_dt()

#     trial = trial_sel if trial_sel is not None else dt.trials[0]
#     ds = dt.trial(trial)
#     label_ds = label_dt.trial(trial)

#     fps = float(ds.attrs.get("fps", ds.fps if hasattr(ds, "fps") else 30.0))
#     data_sr = len(ds.time) / float(ds.time[-1] - ds.time[0]) if len(ds.time) > 1 else fps

#     labels = label_ds.labels.values
#     if labels.ndim > 1 and individuals_sel is not None:
#         individuals = label_ds.individuals.values if "individuals" in label_ds.coords else None
#         if individuals is not None:
#             ind_idx = list(individuals).index(individuals_sel)
#             labels = labels[:, ind_idx]
#         else:
#             labels = labels[:, 0]
#     elif labels.ndim > 1:
#         labels = labels[:, 0]

#     labels = labels.astype(int)

#     segments = get_segment_indices(labels.tolist())

#     if motif_ids is not None:
#         segments = [(val, start, end) for val, start, end in segments if val in motif_ids]

#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise ValueError(f"Cannot open video file: {video_path}")

#     total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     exported_paths = []
#     motif_counters: dict[int, int] = {}

#     for motif_id, start_idx, end_idx in segments:
#         motif_counters[motif_id] = motif_counters.get(motif_id, 0) + 1
#         count = motif_counters[motif_id]

#         start_time = start_idx / data_sr
#         end_time = end_idx / data_sr
#         start_frame = max(0, int(start_time * video_fps) - padding_frames)
#         end_frame = min(total_video_frames - 1, int(end_time * video_fps) + padding_frames)

#         slowdown_suffix = f"_x{slowdown_factor:.1f}" if slowdown_factor != 1.0 else ""
#         output_name = f"motif{motif_id:02d}_{count:03d}_trial{trial}_f{start_frame}-{end_frame}{slowdown_suffix}.mp4"
#         output_path = output_dir / output_name

#         output_fps = video_fps / slowdown_factor
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (frame_width, frame_height))

#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#         for frame_num in range(start_frame, end_frame + 1):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)

#         out.release()
#         exported_paths.append(output_path)
#         print(f"Exported: {output_path.name} ({end_frame - start_frame + 1} frames)")

#     cap.release()

#     print(f"\nExported {len(exported_paths)} clips to {output_dir}")
#     return exported_paths


# def export_motif_clips_all_trials(
#     nc_path: str,
#     video_folder: str,
#     output_dir: Optional[str] = None,
#     individuals_sel: Optional[str] = None,
#     motif_ids: Optional[list[int]] = None,
#     padding_frames: int = 0,
#     slowdown_factor: float = 1.0,
#     video_attr_key: str = "cam1",
# ) -> list[Path]:
#     """Export motif clips from all trials in a dataset.

#     Args:
#         nc_path: Path to the NetCDF file containing labels.
#         video_folder: Folder containing video files referenced in dataset attrs.
#         output_dir: Directory for exported clips.
#         individuals_sel: Individual to select from labels.
#         motif_ids: List of motif IDs to export. If None, exports all.
#         padding_frames: Number of frames to add before/after each clip.
#         slowdown_factor: Factor to slow down playback. 2.0 = half speed.
#         video_attr_key: Attribute key for video filename in dataset (e.g., 'cam1').

#     Returns:
#         List of paths to all exported video clips.
#     """
#     nc_path = Path(nc_path)
#     video_folder = Path(video_folder)
#     output_dir = Path(output_dir) if output_dir else nc_path.parent / "motif_clips"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     dt = TrialTree.open(str(nc_path))
#     all_exported = []

#     for trial in dt.trials:
#         ds = dt.trial(trial)
#         video_filename = ds.attrs.get(video_attr_key)
#         if not video_filename:
#             print(f"Trial {trial}: No video attribute '{video_attr_key}', skipping")
#             continue

#         video_path = video_folder / video_filename
#         if not video_path.exists():
#             print(f"Trial {trial}: Video not found at {video_path}, skipping")
#             continue

#         trial_output_dir = output_dir / f"trial_{trial}"
#         exported = export_motif_clips(
#             nc_path=str(nc_path),
#             video_path=str(video_path),
#             output_dir=str(trial_output_dir),
#             individuals_sel=individuals_sel,
#             trial_sel=trial,
#             motif_ids=motif_ids,
#             padding_frames=padding_frames,
#             slowdown_factor=slowdown_factor,
#         )
#         all_exported.extend(exported)

#     return all_exported


# def get_motif_summary(nc_path: str, trial_sel: Optional[int] = None) -> dict:
#     """Get a summary of all motif segments in a dataset.

#     Args:
#         nc_path: Path to the NetCDF file.
#         trial_sel: Trial number to analyze. If None, analyzes first trial.

#     Returns:
#         Dictionary with motif statistics including segment counts and durations.
#     """
#     dt = TrialTree.open(str(nc_path))
#     label_dt = dt.get_label_dt()

#     trial = trial_sel if trial_sel is not None else dt.trials[0]
#     ds = dt.trial(trial)
#     label_ds = label_dt.trial(trial)

#     fps = float(ds.attrs.get("fps", ds.fps if hasattr(ds, "fps") else 30.0))
#     data_sr = len(ds.time) / float(ds.time[-1] - ds.time[0]) if len(ds.time) > 1 else fps

#     labels = label_ds.labels.values
#     if labels.ndim > 1:
#         labels = labels[:, 0]
#     labels = labels.astype(int)

#     segments = get_segment_indices(labels.tolist())

#     summary: dict = {"trial": trial, "fps": fps, "data_sr": data_sr, "motifs": {}}

#     for motif_id, start_idx, end_idx in segments:
#         duration = (end_idx - start_idx) / data_sr

#         if motif_id not in summary["motifs"]:
#             summary["motifs"][motif_id] = {"count": 0, "total_duration": 0.0, "segments": []}

#         summary["motifs"][motif_id]["count"] += 1
#         summary["motifs"][motif_id]["total_duration"] += duration
#         summary["motifs"][motif_id]["segments"].append(
#             {"start_idx": start_idx, "end_idx": end_idx, "duration": duration}
#         )

#     return summary
