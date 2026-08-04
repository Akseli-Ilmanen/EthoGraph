"""'Load into the GUI' replaces the feature data with the poses dataset.

The keypoints ARE a dataset — a movement poses set whose `keypoint` and
`individual` are ordinary dimensions — so they replace what serves features
rather than being grafted onto it. Merging was the previous approach and it
crashed on any trial carrying an `individual` variable (the labels table has
one), while leaving the grafted dims with no combo, so the features could not
be reduced to something plottable.

Only the data layer moves. Re-running the *file* load instead would rebuild
media, docks and the saved layout over a window that is already up, which is a
native crash — hence `load_keypoint_dataset` takes a Dataset, not a path.

Tests that need a session share one `birdpark_gui` load each and assert several
things per load: the fixture reloads the dataset per test, and enough loads in
one process trips a pre-existing native crash in netCDF.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from qtpy.QtWidgets import QApplication

from ethograph.gui.pose_annotate import KeypointStore, keypoints_dataset_path, store_to_dataset

FPS = 29.88
N_FRAMES = 120


@pytest.fixture
def store() -> KeypointStore:
    """One individual, keypoints named "8" and "9" — a tag-labelled session."""
    store = KeypointStore(keypoint_names=["8", "9"], n_frames=N_FRAMES, individual_names=["individual_0"])
    filled = np.zeros((N_FRAMES, 1, 2, 2))
    filled[:, 0, :, 0] = np.arange(N_FRAMES)[:, None]
    filled[:, 0, :, 1] = 100.0
    store.set_fill(filled, np.ones((N_FRAMES, 1, 2)))
    store.set_point(0, "8", (1.0, 2.0))
    return store


def _load(meta, ds) -> bool:
    loaded = meta.data_widget.load_keypoint_dataset(ds)
    QApplication.processEvents()
    return loaded


def _items(combo) -> list:
    return [combo.itemData(i) for i in range(combo.count())]


# ----------------------------------------------------------------------
# The dataset itself — no session needed
# ----------------------------------------------------------------------


def test_the_written_dataset_is_a_movement_poses_file(store, tmp_path):
    path = tmp_path / "k.nc"
    store_to_dataset(store, FPS, kinematics=["speed"]).to_netcdf(path)

    with xr.open_dataset(path) as ds:
        assert ds.attrs["ds_type"] == "poses"
        assert set(ds.dims) == {"time", "space", "keypoint", "individual"}
        assert list(ds.coords["keypoint"].values) == ["8", "9"]
        assert {"position", "confidence", "speed"} <= set(ds.data_vars)


def test_the_saved_copy_lands_beside_the_video(tmp_path):
    """A real file next to the recording, so it can be reopened or shared —
    but it is a copy, not how the GUI is fed."""
    video = tmp_path / "clip.mp4"
    assert keypoints_dataset_path(video) == tmp_path / "clip.mp4.keypoints.nc"


def test_the_saved_copy_renders_as_a_pose_overlay(store, tmp_path):
    """The other half of "it all works through the .nc": the file the dialog
    writes is a pose file, so the overlay reads it through the ordinary path."""
    from ethograph.gui.pose_render import load_pose_from_file

    path = tmp_path / "k.nc"
    store_to_dataset(store, FPS).to_netcdf(path)

    pr = load_pose_from_file(str(path), source_software=None, fps=FPS)

    assert pr.keypoints == ["8", "9"]
    assert pr.data_not_nan.any()


# ----------------------------------------------------------------------
# In a real session
# ----------------------------------------------------------------------


def test_loading_replaces_the_feature_data(birdpark_gui, store):
    """Replacement, not a merge — and every dimension gets its own combo.

    Merging into a trial that already has an `individual` *variable* raised
    `MergeError`: xarray cannot tell a coordinate from a data variable of the
    same name. And keypoints "8"/"9" have to land in a *keypoint* combo, or
    there is no way to reduce the feature to something a plot can draw.
    """
    _viewer, meta = birdpark_gui
    assert "individual" in meta.app_state.ds.data_vars, "the fixture must have the colliding name"

    assert _load(meta, store_to_dataset(store, FPS, kinematics=["velocity", "speed", "acceleration"]))

    ds = meta.app_state.ds
    assert "individual" in ds.coords
    assert {"position", "confidence", "velocity", "speed", "acceleration"} == set(ds.data_vars)

    combos = meta.data_widget.combos
    assert _items(combos["keypoint"]) == ["8", "9"]
    assert _items(combos["individuals"]) == ["individual_0"]
    assert _items(combos["space"]) == ["x", "y"]

    assert {"position", "velocity", "speed", "acceleration"} <= set(meta.data_widget.catalog.feature_choices())
    assert meta.data_widget._keypoint_names == ["8", "9"]

    # And the end of the chain: pinned dims, so `sel_valid` returns something a
    # plot can draw. Four unpinned axes is what produced an empty panel, and a
    # selection left over from birdpark (`individuals` naming a bird that is not
    # in this data) raised KeyError out of `.sel()`.
    plot = meta.plot_container.line_plots[0]
    plot.set_panel_control("features", "position")
    data = meta.app_state.data_loader.select("position", plot._effective_selections())
    assert data is not None
    assert data.data.ndim <= 2
    assert np.isfinite(data.data).any()


def test_the_recording_and_the_panels_are_left_alone(birdpark_gui, store):
    """Only the data layer moves.

    The media, because re-running the file load lost the alignment whenever the
    dataset sat in another directory — which took every stream rate with it, so
    `video_fps` became None and the first slider drag died on `float * None`.
    The panels, because rebuilding docks over live pygfx canvases is a native
    crash.
    """
    _viewer, meta = birdpark_gui
    fps_before = meta.app_state.video_fps
    alignment_before = meta.app_state.nwb_alignment
    panels_before = list(meta.plot_container.line_plots)
    assert fps_before, "the fixture must have a video rate for this to mean anything"

    _load(meta, store_to_dataset(store, FPS))

    assert meta.app_state.video_fps == fps_before
    assert meta.app_state.nwb_alignment is alignment_before
    assert list(meta.plot_container.line_plots) == panels_before


def test_the_coords_section_is_updated_not_torn_down(birdpark_gui, store):
    """Regression: rebuilding the section deleted widgets it does not own.

    `MetaWidget` inserts "Feature plot type:" at row 0 of this very form and the
    right sidebar borrows the whole group box, but `QFormLayout.removeRow`
    *deletes* what it removes. Clicking a plot afterwards raised
    ``RuntimeError: wrapped C/C++ object of type QComboBox has been deleted``
    from `_on_plot_focus`, and the dimension combos were gone from the sidebar.
    """
    _viewer, meta = birdpark_gui
    foreign = meta.feature_view_combo
    colors = meta.data_widget.combos["colors"]

    _load(meta, store_to_dataset(store, FPS))
    combos_once = sorted(meta.data_widget.combos)
    controls_once = len(meta.data_widget.controls)

    _load(meta, store_to_dataset(store, FPS))

    # Alive, not merely present as a Python wrapper.
    assert foreign is meta.feature_view_combo
    foreign.blockSignals(True)
    foreign.blockSignals(False)
    assert colors is meta.data_widget.combos["colors"]
    colors.count()

    # Rebuilt in place, not appended to.
    assert sorted(meta.data_widget.combos) == combos_once
    assert len(meta.data_widget.controls) == controls_once

    # And the dimension combos are on screen for the user to pick with.
    for key in ("keypoint", "space", "individuals"):
        assert not meta.data_widget._combo_row_fields[key].isHidden()

    # A dimension the next dataset lacks is hidden, not removed, so it can come
    # back. `confidence` alone has no `space`.
    space = meta.data_widget.combos["space"]
    _load(meta, store_to_dataset(store, FPS)[["confidence"]])
    assert meta.data_widget._combo_row_fields["space"].isHidden()
    space.count()  # alive, just hidden
