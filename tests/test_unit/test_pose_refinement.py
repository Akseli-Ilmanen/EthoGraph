"""Pose refinement engine: file round-trips, provenance mapping, fill scopes.

The Qt dialog is a thin shell over these pure pieces; what is tested here is
the part that decides what lands in the ``_refined`` files.
"""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui.dialog_pose_refinement import (
    detections_from_file,
    fill_span,
    full_refined_ds,
    outside_span,
    pose_ds_to_arrays,
    refine_sidecar_path,
    refined_pose_path,
    save_refined_ds,
    shift_anchors,
)
from ethograph.gui.pose_annotate import KeypointStore, KeypointStoreError, store_to_movement_ds

FPS = 25.0


def _file_store(n_frames: int = 6) -> KeypointStore:
    """A store shaped like a loaded DLC file: two keypoints, one individual."""
    return KeypointStore(keypoint_names=["beak", "tail"], n_frames=n_frames, individual_names=["bird"])


def _file_arrays(n_frames: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic file content: beak tracked throughout, tail missing on 2-3."""
    positions = np.full((n_frames, 1, 2, 2), np.nan)
    confidence = np.full((n_frames, 1, 2), np.nan)
    for frame in range(n_frames):
        positions[frame, 0, 0] = (10.0 * frame, 5.0)
        confidence[frame, 0, 0] = 0.9
        if frame not in (2, 3):
            positions[frame, 0, 1] = (10.0 * frame, 50.0)
            confidence[frame, 0, 1] = 0.8
    return positions, confidence


def test_refined_path_keeps_the_source_format(tmp_path):
    assert refined_pose_path(tmp_path / "cam1.csv").name == "cam1_refined.csv"
    assert refined_pose_path(tmp_path / "cam1.h5").name == "cam1_refined.h5"
    assert refined_pose_path(tmp_path / "poses.nc").name == "poses_refined.nc"


def test_slp_refines_to_analysis_h5(tmp_path):
    # movement has no SLEAP project writer, only the analysis .h5.
    assert refined_pose_path(tmp_path / "cam1.slp").name == "cam1_refined.h5"


def test_sidecar_sits_beside_the_pose_file(tmp_path):
    assert refine_sidecar_path(tmp_path / "cam1.csv") == tmp_path / "cam1.csv.refine.json"


def test_pose_ds_round_trips_into_store_arrays():
    store = _file_store()
    store.set_point(1, "beak", (11.0, 12.0))
    ds = store_to_movement_ds(store, FPS)

    positions, confidence, keypoints, individuals = pose_ds_to_arrays(ds)
    assert keypoints == ["beak", "tail"]
    assert individuals == ["bird"]
    assert positions.shape == (6, 1, 2, 2)
    np.testing.assert_allclose(positions[1, 0, 0], [11.0, 12.0])
    assert confidence[1, 0, 0] == 1.0
    assert np.isnan(positions[0, 0, 0]).all()


def test_missing_confidence_means_one_where_present():
    store = _file_store()
    store.set_point(0, "beak", (1.0, 2.0))
    ds = store_to_movement_ds(store, FPS).drop_vars("confidence")

    _, confidence, _, _ = pose_ds_to_arrays(ds)
    assert confidence[0, 0, 0] == 1.0
    assert np.isnan(confidence[0, 0, 1])


def test_detections_skip_empty_frames():
    positions, confidence = _file_arrays()
    positions[4] = np.nan  # frame with nothing in the file
    pos, conf = detections_from_file(positions, confidence)
    assert 4 not in pos
    assert set(pos) == {0, 1, 2, 3, 5}
    np.testing.assert_allclose(pos[0][0, 0], [0.0, 5.0])
    assert conf[0][0, 0] == pytest.approx(0.9)


def test_file_points_confidence_nan_becomes_zero():
    positions, confidence = _file_arrays()
    pos, conf = detections_from_file(positions, confidence)
    # tail is absent on frame 2/3 — its NaN confidence must not poison the store.
    assert conf[2][0, 1] == 0.0


def test_outside_span_keeps_the_edges():
    entries = {0: "a", 2: "b", 4: "c", 6: "d"}
    assert outside_span(entries, (2, 4)) == {0: "a", 6: "d"}
    assert outside_span(entries, None) == entries


def test_user_anchor_beats_the_file_point():
    """The export precedence IS the store's: anchors > file detections > fill."""
    positions, confidence = _file_arrays()
    store = _file_store()
    pos, conf = detections_from_file(positions, confidence)
    store.set_detections(pos, conf)
    store.set_point(1, "beak", (999.0, 999.0))  # the user's correction

    ds = store_to_movement_ds(store, FPS)
    exported = ds["position"].transpose("time", "individual", "keypoint", "space").values
    np.testing.assert_allclose(exported[1, 0, 0], [999.0, 999.0])  # corrected
    np.testing.assert_allclose(exported[0, 0, 0], [0.0, 5.0])  # file kept
    assert np.isnan(exported[2, 0, 1]).all()  # the file's own gap stays a gap


def test_dlc_csv_round_trip(tmp_path):
    """save_refined_ds writes a DLC csv movement can read back verbatim."""
    movement_io = pytest.importorskip("movement.io")

    positions, confidence = _file_arrays()
    store = _file_store()
    pos, conf = detections_from_file(positions, confidence)
    store.set_detections(pos, conf)
    store.set_point(1, "beak", (999.0, 999.0))

    path = tmp_path / "cam1_refined.csv"
    save_refined_ds(store_to_movement_ds(store, FPS), path, "DeepLabCut")
    assert path.exists()

    loaded = movement_io.load_dataset(str(path), "DeepLabCut", FPS)
    reloaded, _, keypoints, individuals = pose_ds_to_arrays(loaded)
    assert keypoints == ["beak", "tail"]
    assert len(individuals) == 1  # split_individuals=False keeps one file
    np.testing.assert_allclose(reloaded[1, 0, 0], [999.0, 999.0])
    np.testing.assert_allclose(reloaded[0, 0, 1], [0.0, 50.0])
    assert np.isnan(reloaded[2, 0, 1]).all()


def test_refined_files_are_overwritten_freely(tmp_path):
    """A refined file is rewritten on every flush — movement's refusal of an
    existing target must not surface, and the previous copy must survive a
    failed write (temp + replace)."""
    movement_io = pytest.importorskip("movement.io")

    store = _file_store()
    store.set_point(0, "beak", (1.0, 2.0))
    path = tmp_path / "cam1_refined.csv"
    save_refined_ds(store_to_movement_ds(store, FPS), path, "DeepLabCut")

    store.set_point(0, "beak", (7.0, 8.0))
    save_refined_ds(store_to_movement_ds(store, FPS), path, "DeepLabCut")

    reloaded, _, _, _ = pose_ds_to_arrays(movement_io.load_dataset(str(path), "DeepLabCut", FPS))
    np.testing.assert_allclose(reloaded[0, 0, 0], [7.0, 8.0])
    assert not list(tmp_path.glob("*.writing*"))  # no temp left behind


def test_full_refined_ds_merges_only_the_window():
    """Editing one trial of a session-wide file leaves the other trials alone."""
    positions, confidence = _file_arrays(n_frames=10)
    # The trial window covers file frames 4..7; the store is trial-local.
    store = KeypointStore(keypoint_names=["beak", "tail"], n_frames=4, individual_names=["bird"])
    store.set_point(1, "beak", (999.0, 999.0))  # file frame 5

    ds = full_refined_ds(positions, confidence, store, start_frame=4, fps=FPS)
    merged = ds["position"].transpose("time", "individual", "keypoint", "space").values
    np.testing.assert_allclose(merged[5, 0, 0], [999.0, 999.0])  # the edit
    np.testing.assert_allclose(merged[0, 0, 0], [0.0, 5.0])  # before the window
    np.testing.assert_allclose(merged[9, 0, 0], [90.0, 5.0])  # after the window
    # The window is authoritative: this store carried no detections (the
    # dialog always loads the file's window points into it), so inside the
    # window an empty store frame means NaN — never a stale file value.
    assert np.isnan(merged[4, 0, 1]).all()
    assert len(merged) == 10


def test_shift_anchors_rekeys_and_drops_negatives():
    anchors = {0: np.zeros((1, 2, 2)), 3: np.ones((1, 2, 2))}
    shifted = shift_anchors(anchors, 4)
    assert set(shifted) == {4, 7}
    back = shift_anchors(shifted, -6)
    assert set(back) == {1}  # frame 4 - 6 < 0 is dropped


def test_netcdf_source_stays_netcdf(tmp_path):
    import xarray as xr

    store = _file_store()
    store.set_point(0, "beak", (1.0, 2.0))
    path = tmp_path / "poses_refined.nc"
    save_refined_ds(store_to_movement_ds(store, FPS), path, "ethograph")
    with xr.open_dataset(path) as ds:
        assert "position" in ds


def test_unwritable_format_raises(tmp_path):
    store = _file_store()
    with pytest.raises(KeypointStoreError):
        save_refined_ds(store_to_movement_ds(store, FPS), tmp_path / "poses_refined.slp", "SLEAP")


def test_fill_scope_my_labels_replaces_the_file_inside_the_span():
    """The 'my labels only' scope: fill between user anchors, file kept outside.

    This mirrors the dialog's _on_fill sequence without Qt: stash detections,
    fill from anchors alone, restore detections outside the filled span.
    """
    from ethograph.gui.pose_fill import SplineBackend

    positions, confidence = _file_arrays()
    store = _file_store()
    file_pos, file_conf = detections_from_file(positions, confidence)
    store.set_detections(file_pos, file_conf)
    # The user marks beak on frames 1 and 4 — disagreeing with the file.
    store.set_point(1, "beak", (100.0, 0.0))
    store.set_point(4, "beak", (400.0, 0.0))

    store.set_detections({}, {})  # scope: my labels only
    filled, conf = SplineBackend().fill(store.flat_observations(), store.n_frames, None, lambda _f: True)
    span = fill_span(filled)
    assert span == (1, 4)
    # Detections first: set_detections discards any fill it finds.
    store.set_detections(outside_span(file_pos, span), outside_span(file_conf, span))
    store.set_fill_from_flat(filled, conf)

    exported = store_to_movement_ds(store, FPS)["position"].transpose("time", "individual", "keypoint", "space").values
    np.testing.assert_allclose(exported[2, 0, 0], [200.0, 0.0])  # fill, not the file
    np.testing.assert_allclose(exported[0, 0, 0], [0.0, 5.0])  # file kept outside
    np.testing.assert_allclose(exported[5, 0, 0], [50.0, 5.0])


def test_fill_scope_with_file_bridges_the_files_own_gap():
    """The 'my labels + file points' scope: file points are observations."""
    from ethograph.gui.pose_fill import SplineBackend

    positions, confidence = _file_arrays()
    store = _file_store()
    file_pos, file_conf = detections_from_file(positions, confidence)
    store.set_detections(file_pos, file_conf)

    filled, conf = SplineBackend().fill(store.flat_observations(), store.n_frames, None, lambda _f: True)
    store.set_fill_from_flat(filled, conf)

    exported = store_to_movement_ds(store, FPS)["position"].transpose("time", "individual", "keypoint", "space").values
    # tail's file gap on frames 2-3 is now bridged...
    np.testing.assert_allclose(exported[2, 0, 1], [20.0, 50.0])
    # ...and the file's own points stand exactly where the file put them.
    np.testing.assert_allclose(exported[1, 0, 1], [10.0, 50.0])
