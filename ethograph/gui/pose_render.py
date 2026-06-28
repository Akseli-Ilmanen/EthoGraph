"""Pose rendering pipeline: lazy loading + dynamic filtering.

Pure functions load pose keypoint data and return PoseRenderData.
PoseDisplayManager orchestrates display using keypoint table selections to
drive which keypoints are shown/hidden.

Two loading paths:
- File-based (DLC, SLEAP, etc.): via ``movement.io.load_dataset``
- NWB-based: direct HDF5 reads from ``PoseEstimationSeries`` with lazy slicing
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from movement.io import load_dataset
from movement.napari.convert import ds_to_napari_layers
from movement.napari.layer_styles import (
    BoxesStyle,
    PointsStyle,
    _sample_colormap,
)

from ethograph.gui.notify import notify
from ethograph.io.nwb_import import _get_absolute_timestamps
from ethograph.skeleton import add_skeleton_layer, nwb_skeleton_to_config
from ethograph.skeleton.shapes import SHAPE_TEMPLATES, shape_outline_for_frame
import xarray as xr

@dataclass
class PoseRenderData:
    """Immutable result of the pose loading + filtering pipeline.

    data         : shape (N, 3) — [frame_idx, y, x], as expected by napari Points
    properties   : DataFrame with per-point metadata (keypoint, individual, confidence, ...)
    data_not_nan : bool mask shape (N,) — True for points that should be shown
    file_name    : label used as the napari layer name base
    """

    data: np.ndarray
    properties: pd.DataFrame
    data_not_nan: np.ndarray
    file_name: str
    bbox_data: np.ndarray | None = None
    frame_path: str | None = None
    skeleton_config: dict | None = None

    @property
    def keypoints(self) -> list[str]:
        if "keypoint" not in self.properties.columns:
            return []
        return self.properties["keypoint"].unique().tolist()


def strip_common_prefix(names: list[str]) -> list[str]:
    """Remove the longest common prefix shared by all names."""
    if len(names) <= 1:
        return names
    prefix = os.path.commonprefix(names)
    if not prefix:
        return names
    return [n[len(prefix) :] for n in names]


def _strip_keypoint_prefix(properties: pd.DataFrame) -> pd.DataFrame:
    if "keypoint" not in properties.columns:
        return properties
    names = properties["keypoint"].tolist()
    prefix = os.path.commonprefix(names)
    if not prefix:
        return properties
    props = properties.copy()
    props["keypoint"] = props["keypoint"].str[len(prefix) :]
    return props


def load_pose_from_file(file_path: str, source_software: str, fps: float) -> PoseRenderData:
    """Load a pose file via movement and return a PoseRenderData."""
    ds = load_dataset(file_path, source_software, fps)
    data, bbox_data, properties = ds_to_napari_layers(ds)
    return PoseRenderData(
        data=data,
        properties=_strip_keypoint_prefix(properties),
        data_not_nan=~np.any(np.isnan(data), axis=1),
        file_name=Path(file_path).name,
        bbox_data=bbox_data,
        frame_path=ds.attrs.get("frame_path"),
    )


def slice_pose_to_frames(pr: PoseRenderData, start_frame: int, end_frame: int) -> PoseRenderData:
    """Slice pose data to ``[start_frame, end_frame)`` and reindex to 0.

    Mirrors the ``TrialVideoSlice`` logic so that pose frame indices align
    with the sliced video.  Works for both points and bounding-box data.
    """
    frame_col = 1 if pr.data.shape[1] > 3 else 0
    frames = pr.data[:, frame_col]
    mask = (frames >= start_frame) & (frames < end_frame)

    if not np.any(mask):
        empty = np.empty((0, pr.data.shape[1]))
        return PoseRenderData(
            data=empty,
            properties=pr.properties.iloc[0:0].reset_index(drop=True),
            data_not_nan=np.empty(0, dtype=bool),
            file_name=pr.file_name,
            frame_path=pr.frame_path,
            skeleton_config=pr.skeleton_config,
        )

    data = pr.data[mask].copy()
    data[:, frame_col] -= start_frame

    bbox_data = None
    if pr.bbox_data is not None:
        bbox_data = pr.bbox_data[mask].copy()
        bbox_data[:, :, 1] -= start_frame  # frame column in each corner vertex

    return PoseRenderData(
        data=data,
        properties=pr.properties[mask].reset_index(drop=True),
        data_not_nan=pr.data_not_nan[mask],
        file_name=pr.file_name,
        bbox_data=bbox_data,
        frame_path=pr.frame_path,
        skeleton_config=pr.skeleton_config,
    )


def load_pose_from_nwb_direct(
    nwb_file: Any,
    pose_estimation_key: str,
    t_start: float | None = None,
    t_stop: float | None = None,
) -> PoseRenderData | None:
    """Load pose directly from NWB PoseEstimationSeries via lazy HDF5 slicing.

    Reads ``series.data`` and ``series.confidence`` using searchsorted for
    efficient time-based slicing. No xarray or movement conversion needed.

    Parameters
    ----------
    nwb_file
        Open pynwb.NWBFile object (provides access to processing modules).
    pose_estimation_key
        Container name (e.g., "pose_dlc", "pose_sleap").
    t_start, t_stop
        Optional time window [t_start, t_stop). If provided, slices data
        and makes time trial-relative (0-relative).

    Returns
    -------
    PoseRenderData | None
        Stacked keypoint data with per-point properties, or None if data
        is unavailable or empty after slicing.
    """
    # Find the pose estimation container
    proc_key = "pose_estimation"
    for mod_name, mod in nwb_file.processing.items():
        if pose_estimation_key in mod.data_interfaces:
            proc_key = mod_name
            break

    container = nwb_file.processing[proc_key][pose_estimation_key]
    raw_kp_names = list(container.pose_estimation_series.keys())
    stripped = strip_common_prefix(raw_kp_names)
    name_map = dict(zip(raw_kp_names, stripped))

    all_pts: list[np.ndarray] = []
    all_not_nan: list[np.ndarray] = []
    kp_col: list[str] = []
    ind_col: list[str] = []
    conf_col: list[float] = []

    for kp_name, series in container.pose_estimation_series.items():
        ts = _get_absolute_timestamps(series)
        n_frames = series.data.shape[0]

        # Determine time window index range
        if t_start is not None and t_stop is not None:
            idx = np.where((ts >= t_start) & (ts <= t_stop))[0]
            if len(idx) == 0:
                continue
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
        else:
            i0, i1 = 0, n_frames

        # Lazy slice from HDF5
        data = np.asarray(series.data[i0:i1], dtype=np.float64)
        n = len(data)
        frames = np.arange(n, dtype=np.float64)

        # NWB stores (x, y); napari Points needs (frame, y, x)
        pts = np.column_stack([frames, data[:, 1], data[:, 0]])
        not_nan = ~np.any(np.isnan(data[:, :2]), axis=1)

        # Optionally load confidence
        confidence: np.ndarray | None = None
        if hasattr(series, "confidence") and series.confidence is not None:
            try:
                confidence = np.asarray(series.confidence[i0:i1], dtype=np.float64)
            except Exception:
                confidence = None

        all_pts.append(pts)
        all_not_nan.append(not_nan)
        kp_col.extend([name_map[kp_name]] * n)
        ind_col.extend([pose_estimation_key] * n)
        conf_col.extend(confidence.tolist() if confidence is not None else [1.0] * n)

    if not all_pts:
        return None

    return PoseRenderData(
        data=np.vstack(all_pts),
        properties=pd.DataFrame(
            {
                "keypoint": kp_col,
                "individual": ind_col,
                "confidence": conf_col,
            }
        ),
        data_not_nan=np.concatenate(all_not_nan),
        file_name=f"NWB_pose_{pose_estimation_key}",
        skeleton_config=_read_skeleton_config(container, set(stripped)),
    )


def _read_skeleton_config(container: Any, keypoints: set[str]) -> dict | None:
    """Read an ndx-pose ``Skeleton`` and return a skeleton config dict.

    The skeleton's ``nodes``/``edges`` (node-index pairs) are converted to the
    config format consumed by ``ethograph.skeleton`` via ``nwb_skeleton_to_config``
    — this is the NWB *config layer*; the renderer is reused unchanged. Node
    names are reconciled with the rendered keypoint names (which may have had a
    common prefix stripped). Returns ``None`` if no skeleton is attached.
    """
    skel = getattr(container, "skeleton", None)
    if skel is None:
        return None

    nodes = [
        n.decode() if isinstance(n, bytes) else str(n)
        for n in np.asarray(skel.nodes[:]).ravel()
    ]
    edges = np.asarray(skel.edges[:]).astype(int).reshape(-1, 2)

    name_map = _match_skeleton_nodes(nodes, keypoints)
    mapped = [name_map.get(n) for n in nodes]
    config = nwb_skeleton_to_config(mapped, edges)
    return config if config["connections"] else None


def _match_skeleton_nodes(nodes: list[str], keypoints: set[str]) -> dict[str, str]:
    """Map skeleton node names to rendered keypoint names.

    Tries an exact match first, then a common-prefix-stripped match so that
    skeletons authored against raw series names still line up with the
    prefix-stripped keypoints used by the points layer.
    """
    mapping = {n: n for n in nodes if n in keypoints}
    if len(mapping) == len(nodes):
        return mapping
    stripped = dict(zip(nodes, strip_common_prefix(nodes)))
    for n in nodes:
        if n not in mapping and stripped[n] in keypoints:
            mapping[n] = stripped[n]
    return mapping


def _resolve_skeleton_colors(config: dict | None, base_color: str | None) -> dict | None:
    """Apply the base colour to edges that have no assigned category.

    Edges with a non-empty ``segment`` (assigned a category in the editor) keep
    their colour; unassigned edges — including every edge of an NWB-derived
    skeleton — take ``base_color``. Returns ``config`` unchanged when there is
    nothing to recolour.
    """
    if config is None or not base_color:
        return config
    connections = [
        ({**c, "color": base_color} if not c.get("segment") else c)
        for c in config["connections"]
    ]
    return {**config, "connections": connections}


def pose_render_to_movement_ds(
    pr: PoseRenderData, scale: tuple[float, float] = (1.0, 1.0)
) -> xr.Dataset:
    """Rebuild a movement-format poses ``xr.Dataset`` from a ``PoseRenderData``.

    The skeleton renderer (``PrecomputedRenderer``) consumes a movement poses
    dataset, so this adapter un-flattens the napari points back into a
    ``(time, space, keypoints, individuals)`` ``position`` array. Points masked
    out by ``data_not_nan`` become NaN (and are skipped by the renderer).
    ``scale`` rescales (y, x) to display coordinates so the skeleton lines up
    with downsampled video.
    """
    sy, sx = scale
    coords = pr.data[:, -3:]
    frames = coords[:, 0].astype(int)
    ys = coords[:, 1] * sy
    xs = coords[:, 2] * sx

    kp = pr.properties["keypoint"].to_numpy()
    if "individual" in pr.properties.columns:
        ind = pr.properties["individual"].to_numpy()
    else:
        ind = np.array(["ind_0"] * len(kp))
    conf = (
        pr.properties["confidence"].to_numpy()
        if "confidence" in pr.properties.columns
        else np.ones(len(kp))
    )

    keypoints = list(dict.fromkeys(kp))
    individuals = list(dict.fromkeys(ind))
    kp_idx = {n: i for i, n in enumerate(keypoints)}
    ind_idx = {n: i for i, n in enumerate(individuals)}
    n_t = int(frames.max()) + 1 if len(frames) else 0

    position = np.full((n_t, 2, len(keypoints), len(individuals)), np.nan)
    confidence = np.full((n_t, len(keypoints), len(individuals)), np.nan)
    for row, valid in enumerate(pr.data_not_nan):
        if not valid:
            continue
        t, k, i = frames[row], kp_idx[kp[row]], ind_idx[ind[row]]
        position[t, 0, k, i] = xs[row]
        position[t, 1, k, i] = ys[row]
        confidence[t, k, i] = conf[row]

    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                position, dims=["time", "space", "keypoints", "individuals"]
            ),
            "confidence": xr.DataArray(
                confidence, dims=["time", "keypoints", "individuals"]
            ),
        },
        coords={
            "time": np.arange(n_t),
            "space": ["x", "y"],
            "keypoints": keypoints,
            "individuals": individuals,
        },
        attrs={"ds_type": "poses"},
    )


def apply_confidence_filter(pr: PoseRenderData, threshold: float) -> PoseRenderData:
    """Mask out points below confidence threshold (UI-driven filtering)."""
    if threshold <= 0.0 or "confidence" not in pr.properties.columns:
        return pr
    mask = pr.data_not_nan.copy()
    mask[pr.properties["confidence"].values < threshold] = False
    return PoseRenderData(pr.data, pr.properties, mask, pr.file_name, pr.bbox_data, pr.frame_path, pr.skeleton_config)


def apply_keypoint_filter(pr: PoseRenderData, hidden: set[str]) -> PoseRenderData:
    """Mask out hidden keypoints from the keypoints table selection."""
    if not hidden or "keypoint" not in pr.properties.columns:
        return pr
    mask = pr.data_not_nan.copy()
    mask[pr.properties["keypoint"].isin(hidden).values] = False
    return PoseRenderData(pr.data, pr.properties, mask, pr.file_name, pr.bbox_data, pr.frame_path, pr.skeleton_config)


class PoseDisplayManager:
    """Manages pose loading, filtering, and napari display.

    Keyoint filtering is driven by the keypoints table in the UI:
    - Confidence threshold filters out low-confidence points
    - Hidden keypoints in the table prevent display

    Uses a single rendering path for all cameras (primary and extra) via
    direct ``add_points()`` calls with ``shown`` mask. Each camera's keypoints
    are tracked independently; the UI filter shows the union.
    """

    def __init__(self, viewer, app_state, video_manager, data_widget):
        self.viewer = viewer
        self.app_state = app_state
        self.video_mgr = video_manager
        self._data_widget = data_widget
        self._primary_points_layer = None
        self._primary_shapes_layer = None
        self._primary_skeleton_layer = None
        self._primary_shape_layer = None
        self._primary_frame_layer = None
        self._primary_file_name: str = ""
        self._primary_pr: PoseRenderData | None = None
        self._extra_pr: dict[str, PoseRenderData] = {}
        self._camera_keypoints: dict[str, list[str]] = {}

    @property
    def all_keypoints(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for kps in self._camera_keypoints.values():
            for k in kps:
                if k not in seen:
                    seen.add(k)
                    result.append(k)
        return result

    def _camera_index(self, camera_name: str | None = None) -> int:
        return self.app_state.nwb_alignment.cameras.index(camera_name)

    def _camera_name_for_index(self, camera_idx: int) -> str:
        cameras = self.app_state.nwb_alignment.cameras
        return cameras[camera_idx] if camera_idx < len(cameras) else str(camera_idx)

    def _resolve_camera_fps(self, camera_idx: int) -> float:
        sio = self.app_state.nwb_alignment  # sio = session io (legacy name)
        cameras = sio.cameras
        if camera_idx < len(cameras):
            fps = sio.get_stream_rate("video", cameras[camera_idx])
            if fps is not None and fps > 0:
                return fps
        return self.app_state.video_fps

    def _get_nwb_file(self) -> Any | None:
        sio = self.app_state.nwb_alignment
        return getattr(sio, "nwb", None)

    def _load_pose_for_camera(self, camera_idx: int) -> PoseRenderData | None:
        trial_id = self.app_state.trials_sel
        sio = self.app_state.nwb_alignment
        cameras = sio.cameras

        if camera_idx < len(cameras):
            pose_path = sio.resolve_media_path(
                trial_id,
                "pose",
                device=cameras[camera_idx],
                fallback_folder=self.app_state.pose_folder,
            )
            if not pose_path:
                return None
            try:
                pr = load_pose_from_file(
                    pose_path,
                    getattr(self.app_state.ds, "source_software", None),
                    self._resolve_camera_fps(camera_idx),
                )
            except (OSError, ValueError, KeyError) as e:
                notify(f"Failed to load pose for camera {camera_idx}: {e}", "warning")
                return None
            alignment = getattr(self.app_state, "trial_alignment", None)
            if alignment and alignment.trial_range:
                fps = self._resolve_camera_fps(camera_idx)
                time_offset = sio.stream_offset_for_trial(
                    trial_id,
                    "video",
                    cameras[camera_idx],
                )
                trial_start = -time_offset
                start_frame = max(0, int(trial_start * fps))
                end_frame = int((trial_start + alignment.trial_range.duration) * fps)
                pr = slice_pose_to_frames(pr, start_frame, end_frame)
            return pr

        pose_keys = list(getattr(self.app_state, "nwb_pose_keys", None) or [])
        if not pose_keys and sio:
            pose_keys = sio.pose_keys
        if pose_keys and camera_idx < len(pose_keys):
            nwb_file = self._get_nwb_file()
            if nwb_file is None:
                return None
            try:
                trial_id = self.app_state.trials_sel
                t_start = sio.start_time(trial_id) if trial_id else None
                t_stop = sio.stop_time(trial_id) if trial_id else None
                return load_pose_from_nwb_direct(
                    nwb_file,
                    pose_keys[camera_idx],
                    t_start=t_start,
                    t_stop=t_stop,
                )
            except (OSError, ValueError, KeyError) as e:
                notify(
                    f"Failed to load NWB pose for {pose_keys[camera_idx]}: {e}",
                    "warning",
                )
                return None
        return None

    def _prepare_pose(self, camera_idx: int, hidden_keypoints: set[str]) -> PoseRenderData | None:
        """Load pose and apply UI-driven filtering (confidence + keypoint table selection)."""
        pr = self._load_pose_for_camera(camera_idx)
        if pr is None:
            return None
        # Apply filters driven by UI keypoints table
        pr = apply_confidence_filter(pr, self.app_state.pose_hide_threshold)
        pr = apply_keypoint_filter(pr, hidden_keypoints)
        return pr if np.any(pr.data_not_nan) else None

    # ------------------------------------------------------------------
    # Per-camera keypoint tracking
    # ------------------------------------------------------------------

    def _register_keypoints(self, camera_name: str, keypoints: list[str] | None):
        if keypoints:
            self._camera_keypoints[camera_name] = keypoints
        self._sync_global_keypoints()

    def _sync_global_keypoints(self):
        merged = self.all_keypoints
        if merged != self.app_state.keypoints:
            self.app_state.keypoints = merged

    def on_camera_removed(self, camera_name: str):
        self._camera_keypoints.pop(camera_name, None)
        self._sync_global_keypoints()

    # ------------------------------------------------------------------
    # Downsample-aware coordinate scaling
    # ------------------------------------------------------------------

    def _pose_scale(self, camera_name: str | None = None) -> tuple[float, float]:
        """Return (scale_y, scale_x) for mapping original-res pose coords to display."""
        from .dialog_video_downsample import get_downsample_scale

        video_folder = self.app_state.video_folder
        if not video_folder:
            return 1.0, 1.0
        dt = self.app_state.dt
        trial_id = self.app_state.trials_sel
        if dt is None or trial_id is None:
            return 1.0, 1.0
        sio = self.app_state.nwb_alignment
        device = camera_name or (sio.cameras[0] if sio.cameras else None)
        video_path = sio.resolve_media_path(
            trial_id,
            "video",
            device=device,
            fallback_folder=video_folder,
        )
        if not video_path:
            return 1.0, 1.0
        return get_downsample_scale(video_folder, Path(video_path).name)

    def _apply_pose_scale(self, points_data: np.ndarray, camera_name: str | None = None) -> np.ndarray:
        sy, sx = self._pose_scale(camera_name)
        if sy == 1.0 and sx == 1.0:
            return points_data
        print(f"[ethograph] Video is downsampled — rescaling pose coordinates by {sy:.3f}")
        scaled = points_data.copy()
        scaled[:, -2] *= sy  # y column
        scaled[:, -1] *= sx  # x column
        return scaled

    # ------------------------------------------------------------------
    # Unified display — same path for primary and extra cameras
    # ------------------------------------------------------------------

    def _display_frame_background(self, pr: PoseRenderData) -> None:
        """Show a static frame image as background when no video is loaded.

        Checks (in order): user-provided frame path on the widget, then
        ``frame_path`` from the movement dataset attributes.
        """
        if self.app_state.video_path:
            return
        frame_path = getattr(self._data_widget, "pose_frame_path", None) or pr.frame_path
        if not frame_path:
            return
        frame_file = Path(frame_path)
        if not frame_file.exists():
            return
        import imageio.v3 as iio

        img = iio.imread(frame_file)
        self._primary_frame_layer = self.viewer.add_image(
            img,
            name="frame",
            rgb=img.ndim == 3,
        )
        idx = self.viewer.layers.index(self._primary_frame_layer)
        self.viewer.layers.move(idx, 0)

    def _display_pose_direct(self, viewer_model, pr: PoseRenderData, camera_name: str | None = None) -> Any | None:
        """Add pose points to any napari viewer, preserving the frame dimension.

        ``ds_to_napari_layers`` returns Tracks format (track_id, frame, y, x).
        napari Points needs only the last 3 columns (frame, y, x).
        Uses ``shown`` mask so napari handles per-frame visibility.
        """
        points_data = pr.data[:, 1:] if pr.data.shape[1] > 3 else pr.data
        points_data = self._apply_pose_scale(points_data, camera_name)
        style_kwargs = self._build_pose_style_kwargs(pr.properties)
        return viewer_model.add_points(
            points_data,
            properties=pr.properties,
            shown=pr.data_not_nan,
            **style_kwargs,
        )

    def _display_bbox_direct(self, viewer_model, pr: PoseRenderData, camera_name: str | None = None) -> Any | None:
        """Add bounding boxes to a napari viewer if present in the data.

        Shapes layers don't support a ``shown`` mask, so data is pre-filtered
        by ``data_not_nan``.  Frame indices in the corner vertices still drive
        per-frame visibility in napari.
        """
        if pr.bbox_data is None:
            return None
        mask = pr.data_not_nan
        bbox_filtered = pr.bbox_data[mask, :, 1:]  # strip track_id
        if len(bbox_filtered) == 0:
            return None
        sy, sx = self._pose_scale(camera_name)
        if sy != 1.0 or sx != 1.0:
            bbox_filtered = bbox_filtered.copy()
            bbox_filtered[:, :, 1] *= sy
            bbox_filtered[:, :, 2] *= sx
        props_filtered = pr.properties[mask].copy().reset_index(drop=True)
        style_kwargs, props_factorized = self._build_bbox_style_kwargs(props_filtered)
        return viewer_model.add_shapes(
            bbox_filtered,
            properties=props_factorized,
            **style_kwargs,
        )

    def _skeleton_enabled(self) -> bool:
        checkbox = getattr(self._data_widget, "pose_show_skeleton_checkbox", None)
        if checkbox is not None:
            return checkbox.isChecked()
        return bool(getattr(self.app_state, "pose_show_skeleton", False))

    def _display_skeleton_direct(
        self, viewer_model, pr: PoseRenderData, camera_name: str | None = None
    ) -> Any | None:
        """Add a skeleton Vectors layer via the shared skeleton renderer.

        Rebuilds a movement poses dataset from ``pr`` and delegates to
        ``ethograph.skeleton.add_skeleton_layer`` with the NWB-derived config.
        The Vectors carry their own frame index, so napari's time slider drives
        per-frame visibility just like the points layer.
        """
        override = getattr(self.app_state, "skeleton_config_override", None)
        config = override if override is not None else pr.skeleton_config
        config = _resolve_skeleton_colors(config, getattr(self.app_state, "skeleton_base_color", None))
        if not self._skeleton_enabled() or config is None:
            return None
        ds = pose_render_to_movement_ds(pr, self._pose_scale(camera_name))
        try:
            return add_skeleton_layer(
                viewer_model,
                ds,
                connections=config,
                name=f"skeleton: {pr.file_name}",
                edge_width=self._skeleton_width(),
            )
        except ValueError:
            return None

    def _shapes_config(self, pr: PoseRenderData) -> list[dict]:
        override = getattr(self.app_state, "skeleton_config_override", None)
        config = override if override is not None else pr.skeleton_config
        if not config:
            return []
        return config.get("shapes", []) or []

    def _display_shapes_direct(
        self, viewer_model, pr: PoseRenderData, camera_name: str | None = None
    ) -> Any | None:
        """Add a napari Shapes layer of anchored shapes that follow the pose.

        Each shape's outline is fit per frame from its anchor keypoints (see
        ``ethograph.skeleton.shapes``); the frame index is the first vertex
        coordinate so napari's time slider drives per-frame visibility.
        """
        shapes = self._shapes_config(pr)
        if not self._skeleton_enabled() or not shapes:
            return None
        positions, kp_index = self._positions_and_index(pr, self._pose_scale(camera_name))
        n_frames = positions.shape[0]

        polygons: list[np.ndarray] = []
        edge_colors: list[str] = []
        for shape in shapes:
            template = SHAPE_TEMPLATES.get(shape.get("type", ""))
            anchors = shape.get("anchors", [])
            if template is None or len(anchors) < 2:
                continue
            if any(a["keypoint"] not in kp_index for a in anchors):
                continue
            anchor_names = [a["point"] for a in anchors]
            anchor_kp = [kp_index[a["keypoint"]] for a in anchors]
            scale = tuple(shape.get("scale", (1.0, 1.0)))
            color = shape.get("color", "#FFCC00")
            for frame in range(n_frames):
                outline = shape_outline_for_frame(
                    template, anchor_names, positions[frame, anchor_kp], scale
                )
                if outline is None:
                    continue
                # outline is (K, 2) in (x, y); napari shape vertex is [frame, y, x]
                verts = np.column_stack(
                    [np.full(len(outline), frame), outline[:, 1], outline[:, 0]]
                )
                polygons.append(verts)
                edge_colors.append(color)

        if not polygons:
            return None
        return viewer_model.add_shapes(
            polygons,
            shape_type="polygon",
            edge_color=edge_colors,
            face_color="transparent",
            edge_width=self._skeleton_width(),
            name=f"shapes: {pr.file_name}",
            opacity=0.9,
        )

    def _positions_and_index(
        self, pr: PoseRenderData, scale: tuple[float, float]
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Return ``(positions (n_frames, n_keypoints, 2), name->index)``."""
        ds = pose_render_to_movement_ds(pr, scale)
        positions = (
            ds.position.isel(individuals=0).transpose("time", "keypoints", "space").values
        )
        kp_index = {n: i for i, n in enumerate(ds.coords["keypoints"].values)}
        return positions, kp_index

    def primary_pose_for_editor(self) -> tuple[list[str], np.ndarray] | None:
        """Return ``(keypoints, positions)`` for the primary camera's pose.

        ``positions`` has shape ``(n_frames, n_keypoints, 2)`` in image-space
        ``(x, y)`` — the input format the skeleton editor dialog expects.
        """
        combo = getattr(self._data_widget, "primary_camera_combo", None)
        name = combo.currentText() if combo is not None else None
        if name is None:
            return None
        pr = self._load_pose_for_camera(self._camera_index(name))
        if pr is None:
            return None
        ds = pose_render_to_movement_ds(pr)
        positions = (
            ds.position.isel(individuals=0)
            .transpose("time", "keypoints", "space")
            .values
        )
        return list(ds.coords["keypoints"].values), positions

    def _skeleton_width(self) -> float:
        spin = getattr(self._data_widget, "pose_skeleton_width_spin", None)
        return spin.value() if spin is not None else 2.0

    def update_pose(self, hidden_keypoints: set[str]) -> None:
        """Update pose display for all cameras based on keypoints table selection.

        Parameters
        ----------
        hidden_keypoints
            Set of keypoint names to hide (from the UI keypoints table).
        """
        primary_combo = getattr(self._data_widget, "primary_camera_combo", None)
        primary_name = primary_combo.currentText() if primary_combo else None
        if primary_name is not None:
            self._display_pose_on_primary(self._camera_index(primary_name), hidden_keypoints)

        for camera_name, widget in self.video_mgr.extra_widgets.items():
            self._display_pose_on_extra(camera_name, hidden_keypoints, widget)

    def _display_pose_on_primary(self, camera_idx: int, hidden_keypoints: set[str]) -> None:
        """Render pose on the main viewer (driven by keypoints table selection)."""
        self._remove_pose_layers()
        pr = self._prepare_pose(camera_idx, hidden_keypoints)
        if pr is None:
            return
        camera_name = self._camera_name_for_index(camera_idx)
        self._register_keypoints(camera_name, pr.keypoints)
        self._primary_file_name = pr.file_name
        self._display_frame_background(pr)
        self._primary_pr = pr
        self._primary_points_layer = self._display_pose_direct(self.viewer, pr, camera_name)
        self._primary_shapes_layer = self._display_bbox_direct(self.viewer, pr, camera_name)
        self._primary_skeleton_layer = self._display_skeleton_direct(self.viewer, pr, camera_name)
        self._primary_shape_layer = self._display_shapes_direct(self.viewer, pr, camera_name)
        self.apply_pose_style()

    def _display_pose_on_extra(
        self,
        camera_name: str,
        hidden_keypoints: set[str],
        widget: Any,
    ) -> None:
        """Render pose on an extra camera widget (driven by keypoints table selection)."""
        if not camera_name:
            widget.clear_pose()
            return
        pr = self._prepare_pose(self._camera_index(camera_name), hidden_keypoints)
        if pr is None:
            widget.clear_pose()
            return
        self._register_keypoints(camera_name, pr.keypoints)
        points_data = pr.data[:, 1:] if pr.data.shape[1] > 3 else pr.data
        points_data = self._apply_pose_scale(points_data, camera_name)
        style_kwargs = self._build_pose_style_kwargs(pr.properties)
        self._extra_pr[camera_name] = pr
        widget.set_pose(points_data, pr.properties, pr.data_not_nan, style_kwargs)
        widget._shapes_layer = self._display_bbox_direct(widget._viewer_model, pr, camera_name)
        widget.set_skeleton(self._display_skeleton_direct(widget._viewer_model, pr, camera_name))
        widget.set_glyphs(self._display_shapes_direct(widget._viewer_model, pr, camera_name))
        self.apply_pose_style()

    def update_extra_camera_pose(self, camera_name: str, hidden_keypoints: set[str]) -> None:
        widget = self.video_mgr.extra_widgets.get(camera_name)
        if widget is None:
            return
        self._display_pose_on_extra(camera_name, hidden_keypoints, widget)

    def _remove_pose_layers(self) -> None:
        if self._primary_points_layer is not None:
            try:
                self.viewer.layers.remove(self._primary_points_layer)
            except ValueError:
                pass
            self._primary_points_layer = None
        if self._primary_shapes_layer is not None:
            try:
                self.viewer.layers.remove(self._primary_shapes_layer)
            except ValueError:
                pass
            self._primary_shapes_layer = None
        if self._primary_skeleton_layer is not None:
            try:
                self.viewer.layers.remove(self._primary_skeleton_layer)
            except ValueError:
                pass
            self._primary_skeleton_layer = None
        if self._primary_shape_layer is not None:
            try:
                self.viewer.layers.remove(self._primary_shape_layer)
            except ValueError:
                pass
            self._primary_shape_layer = None
        if self._primary_frame_layer is not None:
            try:
                self.viewer.layers.remove(self._primary_frame_layer)
            except ValueError:
                pass
            self._primary_frame_layer = None
        file_name = self._primary_file_name
        if not file_name:
            return
        for layer in list(self.viewer.layers):
            if layer.name in [
                f"tracks: {file_name}",
                f"points: {file_name}",
                f"boxes: {file_name}",
                f"skeleton: {file_name}",
            ]:
                self.viewer.layers.remove(layer)

    def _build_pose_style_kwargs(self, properties: pd.DataFrame) -> dict[str, Any]:
        color_prop = "individual"
        if len(properties["individual"].unique()) == 1 and "keypoint" in properties.columns:
            color_prop = "keypoint"

        text_prop = "individual"
        if "keypoint" in properties.columns and len(properties["keypoint"].unique()) > 1:
            text_prop = "keypoint"

        style = PointsStyle(name="pose")
        style.set_text_by(property=text_prop)

        if color_prop == "keypoint":
            global_cycle = self._build_global_color_cycle()
            if global_cycle is not None:
                style.face_color = color_prop
                style.face_color_cycle = global_cycle
                if "color" in style.text:
                    style.text["color"].update({"feature": color_prop, "colormap": global_cycle})
                else:
                    style.text["color"] = {
                        "feature": color_prop,
                        "colormap": global_cycle,
                    }
            else:
                style.set_color_by(property=color_prop, properties_df=properties)
        else:
            style.set_color_by(property=color_prop, properties_df=properties)

        return style.as_kwargs()

    def _build_bbox_style_kwargs(self, properties: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        """Build style kwargs and factorized properties for bounding boxes."""
        color_prop = "individual"
        if len(properties["individual"].unique()) == 1 and "keypoint" in properties.columns:
            color_prop = "keypoint"

        text_prop = "individual"
        if "keypoint" in properties.columns and len(properties["keypoint"].unique()) > 1:
            text_prop = "keypoint"

        props = properties.copy()
        codes, _ = pd.factorize(props[color_prop])
        props[color_prop + "_factorized"] = codes

        style = BoxesStyle(name="bbox")
        style.set_text_by(property=text_prop)
        style.set_color_by(property=color_prop, properties_df=props)
        return style.as_kwargs(), props

    def _build_global_color_cycle(self) -> list[tuple] | None:
        all_kps = self.all_keypoints
        if not all_kps:
            return None
        return _sample_colormap(len(all_kps), "turbo")

    def apply_pose_style(self) -> None:
        visible = self._data_widget.pose_show_text_checkbox.isChecked()
        size = self._data_widget.pose_point_size_spin.value()
        text_size = self._data_widget.pose_text_size_spin.value()
        if self._primary_points_layer is not None:
            self._primary_points_layer.text.visible = visible
            self._primary_points_layer.text.size = text_size
            self._primary_points_layer.size = size
        if self._primary_shapes_layer is not None:
            self._primary_shapes_layer.text.visible = visible
            self._primary_shapes_layer.text.size = text_size
        for widget in self.video_mgr.extra_widgets.values():
            if widget._points_layer is not None:
                widget._points_layer.text.visible = visible
                widget._points_layer.text.size = text_size
                widget._points_layer.size = size
            if getattr(widget, "_shapes_layer", None) is not None:
                widget._shapes_layer.text.visible = visible
                widget._shapes_layer.text.size = text_size

    def apply_skeleton_style(self) -> None:
        """Update skeleton edge width on existing layers in place.

        Width is uniform, so it can be set without rebuilding — avoiding a full
        ``update_pose()`` reload (and the layer re-add it causes).
        """
        width = self._skeleton_width()
        layers = [self._primary_skeleton_layer]
        layers += [getattr(w, "_skeleton_layer", None) for w in self.video_mgr.extra_widgets.values()]
        for layer in layers:
            if layer is not None:
                layer.edge_width = width

    def refresh_skeleton(self) -> None:
        """Rebuild only the skeleton layers (per-edge colours may have changed).

        Reuses the last-rendered ``PoseRenderData`` so points/bbox layers are
        left untouched — base-colour changes apply without reloading the pose.
        """
        combo = getattr(self._data_widget, "primary_camera_combo", None)
        primary_name = combo.currentText() if combo is not None else None
        if self._primary_pr is not None and primary_name is not None:
            if self._primary_skeleton_layer is not None:
                try:
                    self.viewer.layers.remove(self._primary_skeleton_layer)
                except ValueError:
                    pass
            self._primary_skeleton_layer = self._display_skeleton_direct(
                self.viewer, self._primary_pr, primary_name
            )
            if self._primary_shape_layer is not None:
                try:
                    self.viewer.layers.remove(self._primary_shape_layer)
                except ValueError:
                    pass
            self._primary_shape_layer = self._display_shapes_direct(
                self.viewer, self._primary_pr, primary_name
            )
        for camera_name, widget in self.video_mgr.extra_widgets.items():
            pr = self._extra_pr.get(camera_name)
            if pr is not None:
                widget.set_skeleton(
                    self._display_skeleton_direct(widget._viewer_model, pr, camera_name)
                )
                widget.set_glyphs(
                    self._display_shapes_direct(widget._viewer_model, pr, camera_name)
                )

    def on_rotate_video_pose(self) -> None:
        self._rotation_count = (getattr(self, "_rotation_count", 0) + 1) % 4
        theta = np.radians(self._rotation_count * 90)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_2d = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        for layer in self.viewer.layers:
            affine = np.eye(layer.ndim + 1)
            affine[-3:-1, -3:-1] = rot_2d
            layer.affine = affine

        for widget in self.video_mgr.extra_widgets.values():
            for layer in widget._viewer_model.layers:
                affine = np.eye(layer.ndim + 1)
                affine[-3:-1, -3:-1] = rot_2d
                layer.affine = affine
