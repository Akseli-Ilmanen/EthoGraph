"""Pose rendering pipeline: pure loading/filtering functions + display manager.

Pure functions are stateless — they load and filter pose data into PoseRenderData.
PoseDisplayManager orchestrates display using a single code path for all cameras
(primary and extra). Each camera's keypoints are tracked independently; the UI
shows the union across all loaded cameras.

Two loading paths:
- File-based (DLC, SLEAP, etc.): via ``movement.io.load_dataset`` + ``ds_to_napari_layers``
- NWB-based: direct lazy HDF5 reads from ``PoseEstimationSeries`` (no xarray/movement)
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
from movement.napari.layer_styles import PointsStyle, _sample_colormap

from ethograph.gui.notify import notify

@dataclass
class PoseRenderData:
    """Immutable result of the pose loading + filtering pipeline.

    data         : shape (N, 3) — [frame_idx, y, x], as expected by napari Points
    properties   : DataFrame with per-point metadata (keypoint, individual, confidence, ...)
    data_not_nan : bool mask shape (N,) — True for points that should be shown
    file_name    : label used as the napari layer name base
    keypoints    : optional list of keypoint names for populating the keypoint filter UI
    """
    data: np.ndarray
    properties: pd.DataFrame
    data_not_nan: np.ndarray
    file_name: str
    keypoints: list[str] | None = None


def strip_common_prefix(names: list[str]) -> list[str]:
    """Remove the longest common prefix shared by all names."""
    if len(names) <= 1:
        return names
    prefix = os.path.commonprefix(names)
    if not prefix:
        return names
    return [n[len(prefix):] for n in names]


def _strip_keypoint_prefix(properties: pd.DataFrame) -> pd.DataFrame:
    if "keypoint" not in properties.columns:
        return properties
    names = properties["keypoint"].tolist()
    prefix = os.path.commonprefix(names)
    if not prefix:
        return properties
    props = properties.copy()
    props["keypoint"] = props["keypoint"].str[len(prefix):]
    return props


def load_pose_from_file(file_path: str, source_software: str, fps: float) -> PoseRenderData:
    """Load a pose file via movement and return a PoseRenderData."""
    ds = load_dataset(file_path, source_software, fps)
    kp_coord = ds.coords.get("keypoints")
    keypoints = kp_coord.values.astype(str).tolist() if kp_coord is not None else None
    data, _, properties = ds_to_napari_layers(ds)
    return PoseRenderData(
        data=data,
        properties=_strip_keypoint_prefix(properties),
        data_not_nan=~np.any(np.isnan(data), axis=1),
        file_name=Path(file_path).name,
        keypoints=keypoints,
    )


def _get_series_timestamps(series: Any) -> np.ndarray:
    """Extract absolute timestamps from a PoseEstimationSeries."""
    if getattr(series, "timestamps", None) is not None:
        return np.asarray(series.timestamps[:], dtype=np.float64)
    n = series.data.shape[0]
    t0 = float(series.starting_time) if getattr(series, "starting_time", None) is not None else 0.0
    return t0 + np.arange(n, dtype=np.float64) / float(series.rate)


def load_pose_from_nwb_direct(
    nwb_file: Any,
    pose_estimation_key: str,
    t_start: float | None = None,
    t_stop: float | None = None,
) -> PoseRenderData | None:
    """Load pose directly from NWB PoseEstimationSeries (no xarray/movement).

    Reads ``series.data`` and ``series.confidence`` via lazy HDF5 slicing.
    Keypoint names come from the series keys.  Optionally slices to
    ``[t_start, t_stop]`` and makes time trial-relative.
    """
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
        ts = _get_series_timestamps(series)

        if t_start is not None and t_stop is not None:
            idx = np.where((ts >= t_start) & (ts <= t_stop))[0]
            if len(idx) == 0:
                continue
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
        else:
            i0, i1 = 0, series.data.shape[0]

        data = np.asarray(series.data[i0:i1], dtype=np.float64)
        n = len(data)
        frames = np.arange(n, dtype=np.float64)

        # NWB stores (x, y); napari Points needs (frame, row/y, col/x)
        pts = np.column_stack([frames, data[:, 1], data[:, 0]])
        not_nan = ~np.any(np.isnan(data[:, :2]), axis=1)

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
        conf_col.extend(
            confidence.tolist() if confidence is not None else [1.0] * n
        )

    if not all_pts:
        return None

    return PoseRenderData(
        data=np.vstack(all_pts),
        properties=pd.DataFrame(
            {"keypoint": kp_col, "individual": ind_col, "confidence": conf_col}
        ),
        data_not_nan=np.concatenate(all_not_nan),
        file_name=f"NWB_pose_{pose_estimation_key}",
        keypoints=stripped,
    )


def apply_confidence_filter(pr: PoseRenderData, threshold: float) -> PoseRenderData:
    """Zero out data_not_nan for points below the confidence threshold."""
    if threshold <= 0.0 or "confidence" not in pr.properties.columns:
        return pr
    mask = pr.data_not_nan.copy()
    mask[pr.properties["confidence"].values < threshold] = False
    return PoseRenderData(pr.data, pr.properties, mask, pr.file_name)


def apply_keypoint_filter(pr: PoseRenderData, hidden: set[str]) -> PoseRenderData:
    """Zero out data_not_nan for keypoints in the ``hidden`` set."""
    if not hidden or "keypoint" not in pr.properties.columns:
        return pr
    mask = pr.data_not_nan.copy()
    mask[pr.properties["keypoint"].isin(hidden).values] = False
    return PoseRenderData(pr.data, pr.properties, mask, pr.file_name)


class PoseDisplayManager:
    """Manages pose loading, filtering, and napari layer display.

    Uses a single rendering path for all cameras (primary and extra) via
    direct ``add_points()`` calls with ``shown`` mask to preserve the frame
    dimension. Tracks per-camera keypoints and exposes their union for the
    UI filter table.
    """

    def __init__(self, viewer, app_state, video_manager, data_widget):
        self.viewer = viewer
        self.app_state = app_state
        self.video_mgr = video_manager
        self._data_widget = data_widget
        self._primary_points_layer = None
        self._primary_file_name: str = ""
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
        return self.app_state.dt.cameras.index(camera_name)

    def _camera_name_for_index(self, camera_idx: int) -> str:
        cameras = self.app_state.dt.cameras
        return cameras[camera_idx] if camera_idx < len(cameras) else str(camera_idx)

    def _resolve_camera_fps(self, camera_idx: int) -> float:
        dt = self.app_state.dt
        cameras = dt.cameras
        if camera_idx < len(cameras):
            fps = dt.get_video_fps(cameras[camera_idx])
            if fps is not None and fps > 0:
                return fps
        return self.app_state.video_fps

    def _get_nwb_file(self) -> Any | None:
        sio = self.app_state.dt.session_io
        return getattr(sio, "nwb", None)

    def _load_pose_for_camera(self, camera_idx: int) -> PoseRenderData | None:
        dt = self.app_state.dt
        trial_id = self.app_state.trials_sel
        cameras = dt.cameras

        if self.app_state.pose_folder and camera_idx < len(cameras):
            pose_file = dt.get_media(trial_id, "pose", device=cameras[camera_idx])
            if not pose_file:
                return None
            pose_path = os.path.join(self.app_state.pose_folder, pose_file)
            if not os.path.isfile(pose_path):
                return None
            try:
                return load_pose_from_file(
                    pose_path,
                    self.app_state.ds.source_software,
                    self._resolve_camera_fps(camera_idx),
                )
            except (OSError, ValueError, KeyError) as e:
                notify(f"Failed to load pose for camera {camera_idx}: {e}", "warning")
                return None

        pose_keys = list(dt.attrs.get("nwb_pose_keys", []))
        if pose_keys and camera_idx < len(pose_keys):
            nwb_file = self._get_nwb_file()
            if nwb_file is None:
                return None
            try:
                trial_id = self.app_state.trials_sel
                t_start = dt.start_time(trial_id) if trial_id else None
                t_stop = dt.stop_time(trial_id) if trial_id else None
                return load_pose_from_nwb_direct(
                    nwb_file,
                    pose_keys[camera_idx],
                    t_start=t_start,
                    t_stop=t_stop,
                )
            except (OSError, ValueError, KeyError) as e:
                notify(f"Failed to load NWB pose for {pose_keys[camera_idx]}: {e}", "warning")
                return None
        return None

    def _prepare_pose(self, camera_idx: int, hidden_keypoints: set[str]) -> PoseRenderData | None:
        pr = self._load_pose_for_camera(camera_idx)
        if pr is None:
            return None
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
    # Unified display — same path for primary and extra cameras
    # ------------------------------------------------------------------

    def _display_pose_direct(self, viewer_model, pr: PoseRenderData) -> Any | None:
        """Add pose points to any napari viewer, preserving the frame dimension.

        ``ds_to_napari_layers`` returns Tracks format (track_id, frame, y, x).
        napari Points needs only the last 3 columns (frame, y, x).
        Uses ``shown`` mask so napari handles per-frame visibility.
        """
        points_data = pr.data[:, 1:] if pr.data.shape[1] > 3 else pr.data
        style_kwargs = self._build_pose_style_kwargs(pr.properties)
        return viewer_model.add_points(
            points_data, properties=pr.properties, shown=pr.data_not_nan, **style_kwargs,
        )

    def update_pose(self, hidden_keypoints: set[str]) -> None:
        primary_combo = getattr(self._data_widget, "primary_camera_combo", None)
        primary_name = primary_combo.currentText() if primary_combo else None
        if primary_name is not None:
            self._display_pose_on_primary(self._camera_index(primary_name), hidden_keypoints)

        for camera_name, widget in self.video_mgr.extra_widgets.items():
            self._display_pose_on_extra(camera_name, hidden_keypoints, widget)

    def _display_pose_on_primary(self, camera_idx: int, hidden_keypoints: set[str]) -> None:
        self._remove_pose_layers()
        pr = self._prepare_pose(camera_idx, hidden_keypoints)
        if pr is None:
            return
        camera_name = self._camera_name_for_index(camera_idx)
        self._register_keypoints(camera_name, pr.keypoints)
        self._primary_file_name = pr.file_name
        self._primary_points_layer = self._display_pose_direct(self.viewer, pr)
        self.apply_pose_style()

    def _display_pose_on_extra(
        self,
        camera_name: str,
        hidden_keypoints: set[str],
        widget: Any,
    ) -> None:
        if not camera_name:
            widget.clear_pose()
            return
        pr = self._prepare_pose(self._camera_index(camera_name), hidden_keypoints)
        if pr is None:
            widget.clear_pose()
            return
        self._register_keypoints(camera_name, pr.keypoints)
        points_data = pr.data[:, 1:] if pr.data.shape[1] > 3 else pr.data
        style_kwargs = self._build_pose_style_kwargs(pr.properties)
        widget.set_pose(points_data, pr.properties, pr.data_not_nan, style_kwargs)
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
                    style.text["color"] = {"feature": color_prop, "colormap": global_cycle}
            else:
                style.set_color_by(property=color_prop, properties_df=properties)
        else:
            style.set_color_by(property=color_prop, properties_df=properties)

        return style.as_kwargs()

    def _build_global_color_cycle(self) -> list[tuple] | None:
        all_kps = self.all_keypoints
        if not all_kps:
            return None
        return _sample_colormap(len(all_kps), "turbo")

    def apply_pose_style(self) -> None:
        visible = self._data_widget.pose_show_text_checkbox.isChecked()
        size = self._data_widget.pose_point_size_spin.value()
        if self._primary_points_layer is not None:
            self._primary_points_layer.text.visible = visible
            self._primary_points_layer.size = size
        for widget in self.video_mgr.extra_widgets.values():
            if widget._points_layer is not None:
                widget._points_layer.text.visible = visible
                widget._points_layer.size = size

    def on_rotate_video_pose(self) -> None:
        self._rotation_count = (getattr(self, '_rotation_count', 0) + 1) % 4
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
