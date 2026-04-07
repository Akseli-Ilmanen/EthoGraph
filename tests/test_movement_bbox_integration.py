"""Integration test: movement bounding-box sample data through the wizard pipeline.

Fetches the VIA single-crab bounding-box dataset from movement's sample data,
saves it as .nc, reloads via the wizard, and verifies that:
  - The dataset roundtrips correctly (position, shape, confidence, kinematics)
  - Bounding boxes are produced by ds_to_napari_layers
  - The frame_path attribute from fetch_dataset is preserved and readable
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from movement.napari.convert import ds_to_napari_layers
from movement.sample_data import fetch_dataset

import ethograph as eto
from ethograph.io.data_loader import wizard_single_from_pose
from ethograph.gui.pose_render import load_pose_from_file, PoseRenderData
from ethograph.io.trialtree import TrialTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_crab_dataset():
    """Fetch the VIA single-crab bounding-box sample dataset."""
    try:
        return fetch_dataset("VIA_single-crab_MOCA-crab-1_linear-interp.csv")
    except Exception as exc:
        pytest.skip(f"Could not fetch movement sample data: {exc}")


def _safe_to_netcdf(obj, path):
    """Write to netcdf, skipping if netCDF4 has binary compat issues."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            if hasattr(obj, "to_netcdf"):
                obj.to_netcdf(str(path))
    except RuntimeWarning:
        pytest.skip("netCDF4/numpy binary incompatibility in this env")


# ===================================================================
# Dataset fetch + NetCDF roundtrip
# ===================================================================

class TestMovementBBoxFetch:
    """Verify that the movement bbox sample data can be fetched and saved."""

    def test_fetch_returns_bbox_dataset(self):
        ds = _fetch_crab_dataset()
        assert ds.attrs["ds_type"] == "bboxes"
        assert ds.attrs["source_software"] == "VIA-tracks"
        assert "position" in ds.data_vars
        assert "shape" in ds.data_vars

    def test_source_file_attr_from_ds(self):
        ds = _fetch_crab_dataset()
        source_file = ds.attrs.get("source_file")
        assert source_file is not None
        assert Path(source_file).exists()

    def test_frame_path_attr_from_fetch(self):
        ds = _fetch_crab_dataset()
        frame_path = ds.attrs.get("frame_path")
        assert frame_path is not None
        assert Path(frame_path).exists()

    def test_netcdf_roundtrip_preserves_attrs(self, tmp_path):
        ds = _fetch_crab_dataset()
        nc_path = tmp_path / "crab_bbox.nc"
        _safe_to_netcdf(ds, nc_path)
        assert nc_path.exists()

        ds2 = xr.open_dataset(nc_path)
        assert ds2.attrs["ds_type"] == "bboxes"
        assert ds2.attrs["source_file"] == ds.attrs["source_file"]
        assert ds2.attrs["frame_path"] == ds.attrs["frame_path"]
        assert "position" in ds2.data_vars
        assert "shape" in ds2.data_vars
        ds2.close()


# ===================================================================
# Wizard pipeline: fetch → wizard_single_from_pose → TrialTree
# ===================================================================

class TestWizardBBoxPipeline:
    """Load bbox data through wizard_single_from_pose using source_file from attrs."""

    def test_wizard_creates_trialtree(self):
        ds = _fetch_crab_dataset()
        dt = wizard_single_from_pose(
            video_path=None,
            fps=ds.attrs["fps"],
            pose_path=ds.attrs["source_file"],
            source_software="VIA-tracks",
        )
        assert isinstance(dt, TrialTree)
        assert len(dt.trials) == 1

    def test_wizard_preserves_position_and_shape(self):
        ds = _fetch_crab_dataset()
        dt = wizard_single_from_pose(
            video_path=None,
            fps=ds.attrs["fps"],
            pose_path=ds.attrs["source_file"],
            source_software="VIA-tracks",
        )
        trial_ds = dt.itrial(0)
        assert "position" in trial_ds.data_vars
        assert "shape" in trial_ds.data_vars

    def test_wizard_adds_kinematics(self):
        ds = _fetch_crab_dataset()
        dt = wizard_single_from_pose(
            video_path=None,
            fps=ds.attrs["fps"],
            pose_path=ds.attrs["source_file"],
            source_software="VIA-tracks",
        )
        trial_ds = dt.itrial(0)
        assert "velocity" in trial_ds.data_vars
        assert "speed" in trial_ds.data_vars
        assert "acceleration" in trial_ds.data_vars

    def test_wizard_trialtree_netcdf_roundtrip(self, tmp_path):
        ds = _fetch_crab_dataset()
        dt = wizard_single_from_pose(
            video_path=None,
            fps=ds.attrs["fps"],
            pose_path=ds.attrs["source_file"],
            source_software="VIA-tracks",
        )
        nc_out = tmp_path / "crab_trial.nc"
        _safe_to_netcdf(dt, nc_out)
        assert nc_out.exists()

        dt2 = eto.open(str(nc_out))
        assert isinstance(dt2, TrialTree)
        trial_ds = dt2.itrial(0)
        assert "position" in trial_ds.data_vars
        assert "velocity" in trial_ds.data_vars


# ===================================================================
# Bounding-box rendering: ds_to_napari_layers produces bbox_data
# ===================================================================

class TestBBoxRendering:
    """Verify that bounding boxes are produced for napari display."""

    def test_ds_to_napari_layers_returns_bbox_data(self):
        ds = _fetch_crab_dataset()
        data, bbox_data, properties = ds_to_napari_layers(ds)

        assert data is not None
        assert bbox_data is not None
        assert bbox_data.ndim == 3
        # (N_boxes, 4_corners, 4_columns[track_id, frame, y, x])
        assert bbox_data.shape[1] == 4
        assert bbox_data.shape[2] == 4

    def test_load_pose_from_file_returns_bbox(self):
        ds = _fetch_crab_dataset()
        pr = load_pose_from_file(
            ds.attrs["source_file"],
            source_software="VIA-tracks",
            fps=ds.attrs["fps"],
        )
        assert isinstance(pr, PoseRenderData)
        assert pr.bbox_data is not None
        assert pr.bbox_data.shape[1] == 4  # 4 corners per box

    def test_bbox_data_not_nan_mask(self):
        ds = _fetch_crab_dataset()
        pr = load_pose_from_file(
            ds.attrs["source_file"],
            source_software="VIA-tracks",
            fps=ds.attrs["fps"],
        )
        assert pr.data_not_nan.dtype == bool
        assert len(pr.data_not_nan) == len(pr.data)
        assert len(pr.data_not_nan) == len(pr.bbox_data)


# ===================================================================
# Frame path: background image from movement sample data
# ===================================================================

class TestFramePathIntegration:
    """Verify the frame_path from fetch_dataset is usable as a background image."""

    def test_frame_path_is_readable_image(self):
        ds = _fetch_crab_dataset()
        frame_path = ds.attrs.get("frame_path")
        assert frame_path is not None

        import imageio.v3 as iio
        img = iio.imread(frame_path)
        assert img.ndim >= 2  # at least H x W
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_frame_path_preserved_in_netcdf_roundtrip(self, tmp_path):
        """frame_path attr roundtrips through xr.Dataset.to_netcdf."""
        ds = _fetch_crab_dataset()
        nc_path = tmp_path / "with_frame.nc"
        _safe_to_netcdf(ds, nc_path)

        ds2 = xr.open_dataset(nc_path)
        assert ds2.attrs.get("frame_path") == ds.attrs["frame_path"]
        ds2.close()

    def test_pose_render_gets_frame_path_from_fetch_dataset(self):
        """PoseRenderData.frame_path is populated when ds has frame_path attr.

        Note: movement.io.load_dataset does NOT set frame_path (only
        fetch_dataset does), so load_pose_from_file returns frame_path=None
        for regular files. This test verifies the fetch_dataset path works
        directly through ds_to_napari_layers.
        """
        ds = _fetch_crab_dataset()
        frame_path = ds.attrs.get("frame_path")
        assert frame_path is not None

        data, bbox_data, properties = ds_to_napari_layers(ds)
        pr = PoseRenderData(
            data=data,
            properties=properties,
            data_not_nan=~np.any(np.isnan(data), axis=1),
            file_name="test",
            bbox_data=bbox_data,
            frame_path=frame_path,
        )
        assert pr.frame_path is not None
        assert Path(pr.frame_path).exists()
