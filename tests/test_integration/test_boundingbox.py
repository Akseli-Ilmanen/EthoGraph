"""Integration test: movement bounding-box sample data through the wizard pipeline"""

import warnings

import pytest
import xarray as xr
from ethograph.convert import ds_to_napari_layers
from movement.sample_data import fetch_dataset

from ethograph.gui.pose_render import PoseRenderData, load_pose_from_file
from ethograph.io.data_loader import wizard_single_from_pose

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
# Wizard pipeline: fetch → wizard_single_from_pose → TrialTree
# ===================================================================


class TestWizardBBoxPipeline:
    """Load bbox data through wizard_single_from_pose using source_file from attrs."""

    def test_wizard_trialtree_netcdf_roundtrip(self, tmp_path):
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

        nc_out = tmp_path / "crab_trial.nc"
        _safe_to_netcdf(dt, nc_out)
        assert nc_out.exists()

        ds = xr.open_dataset(nc_out)
        assert "position" in ds.data_vars
        assert "shape" in ds.data_vars


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
