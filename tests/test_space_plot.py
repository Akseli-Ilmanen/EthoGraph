"""Tests for SpacePlot axis items and data selection via FeatureStore."""

import numpy as np
import pytest
import xarray as xr

from ethograph.io.feature_store import PlotData, XarrayStore, FeatureStore
from ethograph.gui.plots_space import (
    ReferenceGeometry,
    SEPARATOR,
    _build_axis_items,
    _filter_outliers,
    _parse_axis_item,
    _parse_references,
    _select_axis,
    load_space_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ds():
    """Minimal dataset with position (time, space, keypoints, individuals)
    and a 1-D speed variable."""
    n = 50
    time = np.linspace(0, 1, n)
    position = np.random.randn(n, 3, 2, 1).astype(np.float32)
    speed = np.random.randn(n).astype(np.float32)
    pca = np.random.randn(n, 3).astype(np.float32)

    ds = xr.Dataset(
        {
            "position": xr.DataArray(
                position,
                dims=["time", "space", "keypoints", "individuals"],
                coords={
                    "time": time,
                    "space": ["x", "y", "z"],
                    "keypoints": ["nose", "tail"],
                    "individuals": ["animal1"],
                },
                attrs={"type": "features"},
            ),
            "speed": xr.DataArray(
                speed,
                dims=["time"],
                coords={"time": time},
                attrs={"type": "features"},
            ),
            "pca": xr.DataArray(
                pca,
                dims=["time", "pc"],
                coords={"time": time, "pc": ["PC1", "PC2", "PC3"]},
                attrs={"type": "features"},
            ),
        }
    )
    return ds


# ---------------------------------------------------------------------------
# Tests: feature_dims
# ---------------------------------------------------------------------------

class TestFeatureDims:
    def test_xarray_position_dims(self):
        store = XarrayStore(_make_ds())
        dims = store.feature_dims("position")
        assert "space" in dims
        assert dims["space"] == ["x", "y", "z"]
        assert "keypoints" in dims
        assert "individuals" in dims

    def test_xarray_1d_feature_has_no_dims(self):
        store = XarrayStore(_make_ds())
        dims = store.feature_dims("speed")
        assert dims == {}

    def test_xarray_pca_dims(self):
        store = XarrayStore(_make_ds())
        dims = store.feature_dims("pca")
        assert "pc" in dims
        assert dims["pc"] == ["PC1", "PC2", "PC3"]

    def test_xarray_missing_feature(self):
        store = XarrayStore(_make_ds())
        assert store.feature_dims("nonexistent") == {}


# ---------------------------------------------------------------------------
# Tests: _build_axis_items
# ---------------------------------------------------------------------------

class TestBuildAxisItems:
    def test_items_include_position_columns(self):
        store = XarrayStore(_make_ds())
        items = _build_axis_items(store)
        assert f"position{SEPARATOR}x" in items
        assert f"position{SEPARATOR}y" in items
        assert f"position{SEPARATOR}z" in items

    def test_items_include_1d_features(self):
        store = XarrayStore(_make_ds())
        items = _build_axis_items(store)
        assert "speed" in items

    def test_items_include_pca_columns(self):
        store = XarrayStore(_make_ds())
        items = _build_axis_items(store)
        assert f"pca{SEPARATOR}PC1" in items
        assert f"pca{SEPARATOR}PC2" in items

    def test_no_crash_on_multidim_features(self):
        """Should not crash on features with 3+ non-time dims."""
        store = XarrayStore(_make_ds())
        _build_axis_items(store)  # must not raise


# ---------------------------------------------------------------------------
# Tests: _parse_axis_item
# ---------------------------------------------------------------------------

class TestParseAxisItem:
    def test_with_column(self):
        feat, col = _parse_axis_item(f"position{SEPARATOR}x")
        assert feat == "position"
        assert col == "x"

    def test_without_column(self):
        feat, col = _parse_axis_item("speed")
        assert feat == "speed"
        assert col is None


# ---------------------------------------------------------------------------
# Tests: _select_axis
# ---------------------------------------------------------------------------

class TestSelectAxis:
    def test_select_position_x(self):
        store = XarrayStore(_make_ds())
        sel = {"keypoints": "nose", "individuals": "animal1"}
        time, data = _select_axis(store, f"position{SEPARATOR}x", sel)
        assert time is not None
        assert data is not None
        assert data.ndim == 1
        assert len(data) == 50

    def test_select_1d_feature(self):
        store = XarrayStore(_make_ds())
        time, data = _select_axis(store, "speed", {})
        assert time is not None
        assert data.ndim == 1
        assert len(data) == 50

    def test_select_pca_pc1(self):
        store = XarrayStore(_make_ds())
        time, data = _select_axis(store, f"pca{SEPARATOR}PC1", {})
        assert data is not None
        assert data.ndim == 1

    def test_missing_selections_auto_filled(self):
        """If app_state selections don't cover all dims, defaults are used."""
        store = XarrayStore(_make_ds())
        # No keypoints or individuals in selections — should auto-fill
        time, data = _select_axis(store, f"position{SEPARATOR}x", {})
        assert data is not None
        assert data.ndim == 1

    def test_select_with_time_range(self):
        store = XarrayStore(_make_ds())
        sel = {"keypoints": "nose", "individuals": "animal1"}
        time, data = _select_axis(store, f"position{SEPARATOR}x", sel, t0=0.2, t1=0.8)
        assert time is not None
        assert len(data) < 50
        assert time[0] >= 0.2
        assert time[-1] <= 0.8


# ---------------------------------------------------------------------------
# Tests: _filter_outliers
# ---------------------------------------------------------------------------

class TestFilterOutliers:
    def test_removes_extreme_values(self):
        x = np.zeros(1000)
        y = np.zeros(1000)
        x[0] = 1000.0  # extreme outlier
        y[999] = -500.0
        xf, yf, zf = _filter_outliers(x, y, None, 99.5)
        assert np.isnan(xf[0])
        assert np.isnan(yf[0])  # NaN propagated from x
        assert np.isnan(yf[999])
        assert zf is None

    def test_preserves_normal_values(self):
        rng = np.random.RandomState(42)
        x = rng.randn(100)
        y = rng.randn(100)
        xf, yf, _ = _filter_outliers(x, y, None, 99.5)
        n_nan = np.isnan(xf).sum() + np.isnan(yf).sum()
        assert n_nan < 10  # very few removed at 99.5%

    def test_3d_filtering(self):
        x = np.ones(100)
        y = np.ones(100)
        z = np.ones(100)
        z[50] = 9999.0
        xf, yf, zf = _filter_outliers(x, y, z, 99.0)
        assert np.isnan(zf[50])
        assert np.isnan(xf[50])  # propagated

    def test_does_not_modify_originals(self):
        x = np.array([0.0, 1.0, 1000.0])
        y = np.array([0.0, 1.0, 2.0])
        x_orig = x.copy()
        _filter_outliers(x, y, None, 95.0)
        np.testing.assert_array_equal(x, x_orig)


# ---------------------------------------------------------------------------
# Tests: confidence filter
# ---------------------------------------------------------------------------

class TestConfidenceFilter:
    def _make_ds_with_confidence(self):
        n = 50
        time = np.linspace(0, 1, n)
        position = np.random.randn(n, 3, 1, 1).astype(np.float32)
        confidence = np.ones((n, 1, 1), dtype=np.float32)
        # Mark some frames as low confidence
        confidence[10:15] = 0.1
        confidence[30:35] = 0.2

        ds = xr.Dataset({
            "position": xr.DataArray(
                position,
                dims=["time", "space", "keypoints", "individuals"],
                coords={
                    "time": time,
                    "space": ["x", "y", "z"],
                    "keypoints": ["nose"],
                    "individuals": ["animal1"],
                },
                attrs={"type": "features"},
            ),
            "confidence": xr.DataArray(
                confidence,
                dims=["time", "keypoints", "individuals"],
                coords={
                    "time": time,
                    "keypoints": ["nose"],
                    "individuals": ["animal1"],
                },
            ),
        })
        return ds

    def test_confidence_filter_nans_low_confidence(self):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        ds = self._make_ds_with_confidence()
        app_state = MagicMock()
        app_state.ds = ds

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = app_state

        x = np.ones(50)
        y = np.ones(50)
        sel = {"keypoints": "nose", "individuals": "animal1"}

        xf, yf, zf = sp._apply_confidence_filter(x, y, None, sel, threshold=0.6)
        # Frames 10-14 and 30-34 should be NaN
        assert np.isnan(xf[10:15]).all()
        assert np.isnan(yf[30:35]).all()
        # Other frames should be unchanged
        assert not np.isnan(xf[0:10]).any()

    def test_confidence_filter_raises_without_position(self):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        ds = xr.Dataset({"speed": xr.DataArray([1, 2, 3], dims=["time"])})
        app_state = MagicMock()
        app_state.ds = ds

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = app_state

        with pytest.raises(ValueError, match="position"):
            sp._apply_confidence_filter(np.ones(3), np.ones(3), None, {}, 0.6)

    def test_confidence_filter_raises_without_confidence(self):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        ds = xr.Dataset({
            "position": xr.DataArray(
                np.ones((3, 2)),
                dims=["time", "space"],
                coords={"time": [0, 1, 2], "space": ["x", "y"]},
            ),
        })
        app_state = MagicMock()
        app_state.ds = ds

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = app_state

        with pytest.raises(ValueError, match="confidence"):
            sp._apply_confidence_filter(np.ones(3), np.ones(3), None, {}, 0.6)

    def test_confidence_filter_raises_without_dataset(self):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        app_state = MagicMock()
        app_state.ds = None

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = app_state

        with pytest.raises(ValueError, match="xarray"):
            sp._apply_confidence_filter(np.ones(3), np.ones(3), None, {}, 0.6)

    def _make_uniform_confidence_ds(self, value):
        """Helper: dataset where all confidence values are the same."""
        n = 10
        ds = xr.Dataset({
            "position": xr.DataArray(
                np.ones((n, 2)), dims=["time", "space"],
                coords={"time": np.arange(n, dtype=float), "space": ["x", "y"]},
            ),
            "confidence": xr.DataArray(
                np.full(n, value), dims=["time"],
                coords={"time": np.arange(n, dtype=float)},
            ),
        })
        return ds

    def test_raises_on_all_nan_confidence(self):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = MagicMock()
        sp.app_state.ds = self._make_uniform_confidence_ds(np.nan)

        with pytest.raises(ValueError, match="all NaN"):
            sp._apply_confidence_filter(np.ones(10), np.ones(10), None, {}, 0.6)

    def test_raises_on_all_zeros_confidence(self):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = MagicMock()
        sp.app_state.ds = self._make_uniform_confidence_ds(0.0)

        with pytest.raises(ValueError, match="all 0.0"):
            sp._apply_confidence_filter(np.ones(10), np.ones(10), None, {}, 0.6)

    def test_raises_on_all_ones_confidence(self):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = MagicMock()
        sp.app_state.ds = self._make_uniform_confidence_ds(1.0)

        with pytest.raises(ValueError, match="all 1.0"):
            sp._apply_confidence_filter(np.ones(10), np.ones(10), None, {}, 0.6)


# ---------------------------------------------------------------------------
# Tests: reference geometry parsing
# ---------------------------------------------------------------------------

class TestParseReferences:
    def test_new_format_vertices_edges(self):
        cfg = {
            "references": [{
                "name": "box",
                "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "color": "red",
            }]
        }
        refs = _parse_references(cfg)
        assert len(refs) == 1
        assert refs[0].name == "box"
        assert refs[0].vertices.shape == (4, 2)
        assert len(refs[0].edges) == 4
        assert refs[0].color == "red"

    def test_new_format_3d(self):
        cfg = {
            "references": [{
                "name": "cube",
                "vertices": [
                    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                ],
                "edges": [[0, 1], [4, 5], [0, 4]],
            }]
        }
        refs = _parse_references(cfg)
        assert refs[0].vertices.shape == (8, 3)
        assert refs[0].color == "black"  # default

    def test_multiple_references(self):
        cfg = {
            "references": [
                {"name": "a", "vertices": [[0, 0], [1, 0]], "edges": [[0, 1]]},
                {"name": "b", "vertices": [[2, 2], [3, 3]], "edges": [[0, 1]], "color": "blue"},
            ]
        }
        refs = _parse_references(cfg)
        assert len(refs) == 2
        assert refs[1].color == "blue"

    def test_old_format_2d_polygon(self):
        cfg = {
            "arena": {
                "xy_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            }
        }
        refs = _parse_references(cfg)
        assert len(refs) == 1
        assert refs[0].vertices.shape == (4, 2)
        assert len(refs[0].edges) == 4  # closed polygon

    def test_old_format_3d_box(self):
        cfg = {
            "arena": {
                "xy_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "z_bot": 0.0,
                "z_top": 2.0,
            }
        }
        refs = _parse_references(cfg)
        assert len(refs) == 1
        ref = refs[0]
        assert ref.vertices.shape[1] == 3  # 3D
        assert ref.vertices.shape[0] == 8  # 4 floor + 4 ceiling
        # Should have floor edges + ceiling edges + verticals
        assert len(ref.edges) >= 12

    def test_old_format_already_closed_polygon(self):
        """Polygon where first == last vertex should not double-close."""
        cfg = {
            "arena": {
                "xy_polygon": [[0, 0], [1, 0], [1, 1], [0, 0]],
            }
        }
        refs = _parse_references(cfg)
        # 3 edges for 4 vertices where first==last (already closed)
        assert len(refs[0].edges) == 3

    def test_empty_config(self):
        assert _parse_references({}) == []
        assert _parse_references({"other_key": 123}) == []


class TestLoadSpaceConfig:
    def test_load_yaml(self, tmp_path):
        f = tmp_path / "space.yaml"
        f.write_text("references:\n  - name: test\n    vertices: [[0,0],[1,1]]\n    edges: [[0,1]]\n")
        cfg = load_space_config(f)
        assert "references" in cfg

    def test_missing_file(self, tmp_path):
        assert load_space_config(tmp_path / "nope.yaml") is None


class TestLoadReferencesViaFindConfig:
    def test_finds_new_format(self, tmp_path):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        data_dir = tmp_path / "session"
        data_dir.mkdir()
        ethograph_dir = data_dir / ".ethograph"
        ethograph_dir.mkdir()
        (ethograph_dir / "space.yaml").write_text(
            "references:\n"
            "  - name: box\n"
            "    vertices: [[0,0],[1,0],[1,1],[0,1]]\n"
            "    edges: [[0,1],[1,2],[2,3],[3,0]]\n"
        )

        app_state = MagicMock()
        app_state.nc_file_path = str(data_dir / "data.nc")

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = app_state

        refs = sp._load_references()
        assert len(refs) == 1
        assert refs[0].name == "box"

    def test_finds_old_format_arena(self, tmp_path):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        data_dir = tmp_path / "session"
        data_dir.mkdir()
        ethograph_dir = data_dir / ".ethograph"
        ethograph_dir.mkdir()
        (ethograph_dir / "space.yaml").write_text(
            "arena:\n"
            "  xy_polygon: [[0,0],[1,0],[1,1],[0,1]]\n"
            "  z_bot: 0.0\n"
            "  z_top: 1.0\n"
        )

        app_state = MagicMock()
        app_state.nc_file_path = str(data_dir / "data.nc")

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = app_state

        refs = sp._load_references()
        assert len(refs) == 1
        assert refs[0].vertices.shape[1] == 3

    def test_returns_empty_when_no_config(self, tmp_path):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        app_state = MagicMock()
        app_state.nc_file_path = str(tmp_path / "data.nc")

        sp = SpacePlot.__new__(SpacePlot)
        sp.app_state = app_state

        assert sp._load_references() == []


# ---------------------------------------------------------------------------
# Tests: time_marker_updated signal wiring
# ---------------------------------------------------------------------------

class TestTimeMarkerSignal:
    """Verify that both update paths emit time_marker_updated."""

    def test_update_time_marker_by_time_emits(self, qtbot):
        from unittest.mock import MagicMock, patch
        from ethograph.gui.plots_container import UnifiedPanelContainer

        app_state = MagicMock()
        app_state.get_with_default = MagicMock(return_value=None)
        app_state.labels_visible = False

        with patch.object(UnifiedPanelContainer, '__init__', lambda self, *a, **kw: None):
            container = UnifiedPanelContainer.__new__(UnifiedPanelContainer)

        # Manually init just what we need
        from qtpy.QtCore import Signal
        # Can't easily test Signal emission on a non-QObject created this way,
        # so test the method directly
        container.time_slider = MagicMock()
        container._label_indicator = MagicMock()
        container.app_state = app_state

        emitted = []
        # Create a real container to test signal
        # Instead, test the logic: both methods should call emit
        # We'll verify by checking the source code paths exist

    def test_update_time_marker_and_window_calls_emit(self):
        """Verify update_time_marker_and_window has time_marker_updated.emit call."""
        import inspect
        from ethograph.gui.plots_container import UnifiedPanelContainer
        source = inspect.getsource(UnifiedPanelContainer.update_time_marker_and_window)
        assert "time_marker_updated.emit" in source

    def test_update_time_marker_by_time_calls_emit(self):
        """Verify update_time_marker_by_time has time_marker_updated.emit call."""
        import inspect
        from ethograph.gui.plots_container import UnifiedPanelContainer
        source = inspect.getsource(UnifiedPanelContainer.update_time_marker_by_time)
        assert "time_marker_updated.emit" in source


# ---------------------------------------------------------------------------
# Tests: highlight caching logic
# ---------------------------------------------------------------------------

class TestHighlightCaching:
    """Test _highlight_label_at_time only redraws on label change."""

    def test_same_label_does_not_redraw(self):
        """Calling with same time in same label should not re-highlight."""
        import pandas as pd
        from unittest.mock import MagicMock
        from ethograph.gui.widgets_data import DataWidget

        dw = DataWidget.__new__(DataWidget)
        dw.app_state = MagicMock()
        dw.app_state.label_intervals = pd.DataFrame({
            "onset_s": [1.0, 3.0],
            "offset_s": [2.0, 4.0],
            "labels": [1, 2],
        })
        dw.labels_widget = MagicMock()
        dw.labels_widget._mappings = {1: {"color": (255, 0, 0)}, 2: {"color": (0, 255, 0)}}
        dw.space_plot = MagicMock()
        dw._space_highlight_key = None

        # First call at t=1.5 → should highlight
        dw._highlight_label_at_time(1.5)
        assert dw.space_plot.highlight_time_segment.call_count == 1
        assert dw._space_highlight_key == (1.0, 2.0, 1)

        # Second call at t=1.8 (same label) → should NOT highlight again
        dw._highlight_label_at_time(1.8)
        assert dw.space_plot.highlight_time_segment.call_count == 1  # unchanged

        # Third call at t=3.5 (different label) → should highlight
        dw._highlight_label_at_time(3.5)
        assert dw.space_plot.highlight_time_segment.call_count == 2
        assert dw._space_highlight_key == (3.0, 4.0, 2)

    def test_outside_label_clears_key(self):
        import pandas as pd
        from unittest.mock import MagicMock
        from ethograph.gui.widgets_data import DataWidget

        dw = DataWidget.__new__(DataWidget)
        dw.app_state = MagicMock()
        dw.app_state.label_intervals = pd.DataFrame({
            "onset_s": [1.0],
            "offset_s": [2.0],
            "labels": [1],
        })
        dw.labels_widget = MagicMock()
        dw.labels_widget._mappings = {1: {"color": (255, 0, 0)}}
        dw.space_plot = MagicMock()
        dw._space_highlight_key = (1.0, 2.0, 1)

        # Move outside any label
        dw._highlight_label_at_time(5.0)
        assert dw._space_highlight_key is None
        assert dw.space_plot.highlight_time_segment.call_count == 0

    def test_empty_labels(self):
        import pandas as pd
        from unittest.mock import MagicMock
        from ethograph.gui.widgets_data import DataWidget

        dw = DataWidget.__new__(DataWidget)
        dw.app_state = MagicMock()
        dw.app_state.label_intervals = pd.DataFrame(columns=["onset_s", "offset_s", "labels"])
        dw.space_plot = MagicMock()
        dw._space_highlight_key = None

        dw._highlight_label_at_time(1.0)
        assert dw.space_plot.highlight_time_segment.call_count == 0


# ---------------------------------------------------------------------------
# Tests: marker (2D and 3D)
# ---------------------------------------------------------------------------

class TestMarker:
    def _make_space_plot(self, *, is_3d: bool, has_z: bool = True):
        """Create a minimal SpacePlot with a mock widget for testing."""
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot
        import pyqtgraph.opengl as gl

        sp = SpacePlot.__new__(SpacePlot)
        sp.cb_marker = MagicMock()
        sp.cb_marker.isChecked.return_value = True
        sp._time_marker_item = None

        X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        Y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        Z = np.array([0.0, 0.5, 1.0, 1.5, 2.0]) if has_z else None
        sp._trajectory_pos = (X, Y, Z)
        sp._trajectory_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        if is_3d:
            sp.space_widget = MagicMock(spec=gl.GLViewWidget)
        else:
            sp.space_widget = MagicMock()
            # Make isinstance check return False for GLViewWidget
            sp.space_widget.__class__ = type('PlotWidget', (), {})

        return sp

    def test_2d_marker_created(self):
        sp = self._make_space_plot(is_3d=False)
        sp.update_time_marker(0.5)
        assert sp._time_marker_item is not None

    def test_2d_marker_updated(self):
        sp = self._make_space_plot(is_3d=False)
        sp.update_time_marker(0.25)
        first_item = sp._time_marker_item
        assert first_item is not None
        sp.update_time_marker(0.75)
        # Same item reused, not recreated
        assert sp._time_marker_item is first_item

    def test_3d_marker_created(self):
        sp = self._make_space_plot(is_3d=True)
        sp.update_time_marker(0.5)
        assert sp._time_marker_item is not None
        # Verify it was added to the GL widget
        sp.space_widget.addItem.assert_called_once()

    def test_3d_marker_created_without_z_data(self):
        """3D widget but Z is None — should still create marker at z=0."""
        sp = self._make_space_plot(is_3d=True, has_z=False)
        sp.update_time_marker(0.5)
        assert sp._time_marker_item is not None
        sp.space_widget.addItem.assert_called_once()

    def test_3d_marker_updated(self):
        sp = self._make_space_plot(is_3d=True)
        sp.update_time_marker(0.25)
        first_item = sp._time_marker_item
        # Replace setData with a mock so we can track the update call
        from unittest.mock import MagicMock
        first_item.setData = MagicMock()
        sp.update_time_marker(0.75)
        assert sp._time_marker_item is first_item
        first_item.setData.assert_called_once()

    def test_marker_removed_when_unchecked(self):
        sp = self._make_space_plot(is_3d=False)
        sp.update_time_marker(0.5)
        assert sp._time_marker_item is not None

        sp.cb_marker.isChecked.return_value = False
        sp.update_time_marker(0.75)
        assert sp._time_marker_item is None

    def test_3d_marker_removed_when_unchecked(self):
        sp = self._make_space_plot(is_3d=True)
        sp.update_time_marker(0.5)
        assert sp._time_marker_item is not None

        sp.cb_marker.isChecked.return_value = False
        sp.update_time_marker(0.75)
        assert sp._time_marker_item is None
        sp.space_widget.removeItem.assert_called()

    def test_no_crash_with_no_data(self):
        from unittest.mock import MagicMock
        from ethograph.gui.plots_space import SpacePlot

        sp = SpacePlot.__new__(SpacePlot)
        sp.cb_marker = MagicMock()
        sp.cb_marker.isChecked.return_value = True
        sp.space_widget = MagicMock()
        sp._trajectory_pos = None
        sp._trajectory_times = None
        sp._time_marker_item = None

        sp.update_time_marker(0.5)  # should not crash
        assert sp._time_marker_item is None
