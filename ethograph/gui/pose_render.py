"""Pose rendering pipeline: pure loading/filtering functions + display manager.

Pure functions are stateless — they load and filter pose data into PoseRenderData.
PoseDisplayManager orchestrates display: loading, filtering, napari layer management,
and secondary-video pose sync. It writes into DataLoader attributes only when needed
by movement's layer-creation methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from movement.io import load_dataset
from movement.napari.convert import ds_to_napari_layers
from movement.napari.layer_styles import PointsStyle
from movement.napari.loader_widgets import SUPPORTED_POSES_FILES
from napari.utils.notifications import show_warning

from ethograph.utils.xr_utils import get_time_coord

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
    
    
    data, _,  properties = ds_to_napari_layers(ds)
    return PoseRenderData(
        data=data,
        properties=_strip_keypoint_prefix(properties),
        data_not_nan=~np.any(np.isnan(data), axis=1),
        file_name=Path(file_path).name,
        keypoints=keypoints,
    )


def load_pose_from_nwb(
    ds: xr.Dataset,
    camera_idx: int = 0,
    camera_name: str = "",
) -> PoseRenderData:
    """Extract pose from an NWB-sourced xr.Dataset (multiview ``view`` dim or legacy ``cameras`` dim)."""
    pos = ds["position"]
    if "view" in pos.dims:
        pos = pos.isel(view=camera_idx)
    elif "cameras" in pos.dims:
        pos = pos.isel(cameras=camera_idx)

    if "confidence" in ds.data_vars:
        conf = ds["confidence"]
        if "view" in conf.dims:
            conf = conf.isel(view=camera_idx)
        elif "cameras" in conf.dims:
            conf = conf.isel(cameras=camera_idx)
    else:
        conf = xr.ones_like(pos.isel(space=0).drop_vars("space"))

    return _pose_arrays_to_render_data(pos, conf, camera_name or str(camera_idx))


def load_pose_from_nwb_variable(
    ds: xr.Dataset,
    cam_suffix: str,
    camera_name: str = "",
) -> PoseRenderData:
    """Extract pose from a merged NWB dataset using ``position_{cam_suffix}`` variable."""
    pos = ds[f"position_{cam_suffix}"]
    conf_key = f"confidence_{cam_suffix}"
    conf = ds[conf_key] if conf_key in ds.data_vars else xr.ones_like(pos.isel(space=0).drop_vars("space"))
    return _pose_arrays_to_render_data(pos, conf, camera_name or cam_suffix)


def _pose_arrays_to_render_data(
    pos: xr.DataArray,
    conf: xr.DataArray,
    label_suffix: str,
) -> PoseRenderData:
    time_coord = get_time_coord(pos)
    if time_coord is None:
        raise ValueError("Position data has no time coordinate")
    time_name = time_coord.name or time_coord.dims[0]

    pos_mv = pos.transpose(time_name, "space", "individuals", "keypoints")
    conf_mv = conf.transpose(time_name, "individuals", "keypoints")
    mv_ds = xr.Dataset({"position": pos_mv, "confidence": conf_mv}).rename({time_name: "time"})
    mv_ds.attrs["ds_type"] = "poses"

    data, _,  properties = ds_to_napari_layers(mv_ds)
    return PoseRenderData(
        data=data,
        properties=_strip_keypoint_prefix(properties),
        data_not_nan=~np.any(np.isnan(data), axis=1),
        file_name=f"NWB_pose_{label_suffix}",
        keypoints=None,
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


    Delegates to DataLoader (movement) for the actual layer creation via
    ``_set_common_color_property``, ``_set_text_property``, ``_add_points_layer``,
    and ``_set_initial_state``. Owns the pose lifecycle on behalf of DataWidget.
    """

    def __init__(self, data_loader, app_state, video_manager):
        self._dl = data_loader
        self.app_state = app_state
        self.video_mgr = video_manager

    def _camera_index(self, camera_name: str | None = None) -> int:
        cameras = self.app_state.dt.cameras
        return cameras.index(camera_name)

    def _has_embedded_pose(self) -> bool:
        ds = self.app_state.ds
        if "position" in ds.data_vars:
            return True
        return any(k.startswith("position_") for k in ds.data_vars)

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
                    self.video_mgr.secondary_fps or self.app_state.video_fps,
                )
            except (OSError, ValueError, KeyError) as e:
                show_warning(f"Failed to load pose for camera {camera_idx}: {e}")
                return None
        if self._has_embedded_pose():
            ds = self.app_state.ds
            if "position" in ds.data_vars:
                return load_pose_from_nwb(ds, camera_idx=camera_idx)
            pose_keys = list(dt.attrs.get("nwb_pose_keys", []))
            if camera_idx < len(pose_keys):
                return load_pose_from_nwb_variable(ds, pose_keys[camera_idx])
        return None

    def _prepare_pose(self, camera_idx: int, hidden_keypoints: set[str]) -> PoseRenderData | None:
        pr = self._load_pose_for_camera(camera_idx)
        if pr is None:
            return None
        pr = apply_confidence_filter(pr, self.app_state.pose_hide_threshold)
        pr = apply_keypoint_filter(pr, hidden_keypoints)
        return pr if np.any(pr.data_not_nan) else None

    # ------------------------------------------------------------------
    # Display (primary uses DataLoader pipeline, secondary uses widget)
    # ------------------------------------------------------------------



    def update_pose(self, hidden_keypoints: set[str]) -> None:
        primary_combo = getattr(self._dl, "primary_camera_combo", None)
        primary_name = primary_combo.currentText() if primary_combo else None
        if primary_name is not None:
            self._display_pose_on_primary(self._camera_index(primary_name), hidden_keypoints)

        secondary_combo = getattr(self._dl, "secondary_camera_combo", None)
        if secondary_combo is None:
            return
        self._display_pose_on_secondary(secondary_combo.currentText(), hidden_keypoints)
        
        return 

    def _display_pose_on_primary(self, camera_idx: int, hidden_keypoints: set[str]) -> None:
        self._remove_pose_layers()
        pr = self._prepare_pose(camera_idx, hidden_keypoints)
        if pr is None:
            return
        new_keys = pr.keypoints or []
        existing = self.app_state.keypoints
        merged = existing + [k for k in new_keys if k not in existing]
        if merged != existing:
            self.app_state.keypoints = merged
        self._dl.file_name = pr.file_name
        self._dl.data = pr.data
        self._dl.properties = pr.properties
        self._dl.data_not_nan = pr.data_not_nan
        self._dl._set_common_color_property()
        self._dl._set_text_property()
        self._dl._add_points_layer()
        self.apply_pose_style()
        self._dl._set_initial_state()

    def _display_pose_on_secondary(self, camera_name: str | None, hidden_keypoints: set[str]) -> None:
        sw = self.video_mgr.secondary_widget
        if sw is None:
            return
        if not camera_name or camera_name == "None":
            sw.clear_pose()
            return
        pr = self._prepare_pose(self._camera_index(camera_name), hidden_keypoints)
        if pr is None:
            sw.clear_pose()
            return
        visible_data = pr.data[pr.data_not_nan, 1:]
        visible_props = pr.properties.iloc[pr.data_not_nan, :].reset_index(drop=True)
        style_kwargs = self._build_pose_style_kwargs(pr.properties)
        try:
            sw.set_pose_layer(
                data=visible_data,
                properties=visible_props,
                style_kwargs=style_kwargs,
            )
            self.apply_pose_style()
        except (OSError, ValueError, KeyError) as e:
            show_warning(f"Failed to set secondary pose layer: {e}")
            sw.clear_pose()

    def update_secondary_pose(self, hidden_keypoints: set[str], camera_name: str | None = None) -> None:
        if camera_name is None:
            combo = getattr(self._dl, "secondary_camera_combo", None)
            camera_name = combo.currentText() if combo else None
        self._display_pose_on_secondary(camera_name, hidden_keypoints)

    def _remove_pose_layers(self) -> None:
        file_name = self._dl.file_name
        if not file_name:
            return
        viewer = self._dl.viewer
        for layer in list(viewer.layers):
            if layer.name in [
                f"tracks: {file_name}",
                f"points: {file_name}",
                f"boxes: {file_name}",
                f"skeleton: {file_name}",
            ]:
                viewer.layers.remove(layer)

    def _build_pose_style_kwargs(self, properties: pd.DataFrame) -> dict[str, Any]:
        color_prop = "individual"
        if len(properties["individual"].unique()) == 1 and "keypoint" in properties.columns:
            color_prop = "keypoint"

        text_prop = "individual"
        if "keypoint" in properties.columns and len(properties["keypoint"].unique()) > 1:
            text_prop = "keypoint"

        style = PointsStyle(name="secondary_pose")
        style.set_text_by(property=text_prop)
        style.set_color_by(property=color_prop, properties_df=properties)
        return style.as_kwargs()

    def apply_pose_style(self) -> None:
        visible = self._dl.pose_show_text_checkbox.isChecked()
        size = self._dl.pose_point_size_spin.value()
        points_layer = getattr(self._dl, 'points_layer', None)
        if points_layer is not None:
            points_layer.text.visible = visible
            points_layer.size = size
        sw = self.video_mgr.secondary_widget
        if sw is not None and sw._points_layer is not None:
            sw._points_layer.text.visible = visible
            sw._points_layer.size = size

    def on_rotate_video_pose(self) -> None:
        self._rotation_count = (getattr(self, '_rotation_count', 0) + 1) % 4
        theta = np.radians(self._rotation_count * 90)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_2d = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        for layer in self._dl.viewer.layers:
            affine = np.eye(layer.ndim + 1)
            affine[-3:-1, -3:-1] = rot_2d
            layer.affine = affine

        sw = self.video_mgr.secondary_widget
        if sw is not None:
            for layer in sw._viewer_model.layers:
                affine = np.eye(layer.ndim + 1)
                affine[-3:-1, -3:-1] = rot_2d
                layer.affine = affine

