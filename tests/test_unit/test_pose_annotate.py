"""KeypointStore editing, NaN handling and export round-trips.

The store is hierarchical — individuals × keypoints — so every test here reads
positions as ``(n_individuals, n_keypoints, 2)``. Single-individual labelling is
the ``n_individuals == 1`` case and gets the ``individual=None`` default.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ethograph.gui.pose_annotate import (
    FEATURE_TIME_DIM,
    KINEMATICS,
    KeypointStore,
    KeypointStoreError,
    UnknownIndividualError,
    UnknownKeypointError,
    sidecar_path,
    store_to_kinematics,
    store_to_movement_ds,
)

NAMES = ["beak", "tail", "eye"]
FPS = 25.0


@pytest.fixture
def store() -> KeypointStore:
    return KeypointStore(keypoint_names=list(NAMES), n_frames=10)


@pytest.fixture
def pair() -> KeypointStore:
    """Two individuals sharing one keypoint schema."""
    return KeypointStore(keypoint_names=list(NAMES), n_frames=10, individual_names=["crow_a", "crow_b"])


def test_defaults_to_one_individual(store):
    assert store.individual_names == ["individual_0"]
    assert store.n_points == len(NAMES)


def test_set_and_read_point(store):
    store.set_point(3, "beak", (12.5, 7.0))
    assert store.anchor_frames() == [3]
    np.testing.assert_allclose(store.positions_for(3)[0], [12.5, 7.0])
    assert np.all(np.isnan(store.positions_for(3)[1:]))


def test_unknown_keypoint_raises(store):
    with pytest.raises(UnknownKeypointError):
        store.set_point(0, "wing", (1.0, 2.0))


def test_unknown_individual_raises(store):
    with pytest.raises(UnknownIndividualError):
        store.set_point(0, "beak", (1.0, 2.0), individual="nobody")


def test_partial_anchors_are_per_keypoint(store):
    store.set_point(1, "beak", (1.0, 1.0))
    store.set_point(5, "tail", (2.0, 2.0))
    assert store.anchor_frames() == [1, 5]
    assert store.anchor_frames_for("beak") == [1]
    assert store.anchor_frames_for("tail") == [5]
    assert store.anchor_frames_for("eye") == []


def test_clear_point_drops_empty_anchor(store):
    store.set_point(2, "beak", (1.0, 1.0))
    store.clear_point(2, "beak")
    assert store.anchor_frames() == []


def test_clear_point_keeps_partly_labelled_anchor(store):
    store.set_point(2, "beak", (1.0, 1.0))
    store.set_point(2, "tail", (3.0, 4.0))
    store.clear_point(2, "beak")
    assert store.anchor_frames() == [2]
    assert np.isnan(store.positions_for(2)[0, 0])
    np.testing.assert_allclose(store.positions_for(2)[1], [3.0, 4.0])


def test_undo_restores_previous_value(store):
    store.set_point(4, "beak", (1.0, 1.0))
    store.set_point(4, "beak", (9.0, 9.0))
    store.undo()
    np.testing.assert_allclose(store.positions_for(4)[0], [1.0, 1.0])
    store.undo()
    assert store.anchor_frames() == []


def test_undo_of_clear_restores_point(store):
    store.set_point(4, "tail", (5.0, 6.0))
    store.clear_point(4, "tail")
    store.undo()
    np.testing.assert_allclose(store.positions_for(4)[1], [5.0, 6.0])


def test_undo_on_empty_history_is_a_noop(store):
    store.undo()
    assert store.anchor_frames() == []


def test_nearest_respects_radius(store):
    store.set_point(0, "beak", (10.0, 10.0))
    assert store.nearest(0, (11.0, 10.0), radius=3.0) == ("individual_0", "beak")
    assert store.nearest(0, (30.0, 10.0), radius=3.0) is None
    assert store.nearest(7, (10.0, 10.0), radius=3.0) is None


def test_set_keypoint_names_carries_points_and_clears_fill(store):
    store.set_point(0, "beak", (1.0, 2.0))
    store.set_point(0, "tail", (3.0, 4.0))
    store.set_fill(np.zeros((10, 1, 3, 2)), np.zeros((10, 1, 3)))

    store.set_keypoint_names(["tail", "beak"])

    assert store.keypoint_names == ["tail", "beak"]
    np.testing.assert_allclose(store.positions_for(0)[0], [3.0, 4.0])
    np.testing.assert_allclose(store.positions_for(0)[1], [1.0, 2.0])
    assert store.filled is None


def test_set_keypoint_names_drops_removed_keypoints(store):
    store.set_point(0, "eye", (1.0, 1.0))
    store.set_keypoint_names(["beak", "tail"])
    assert store.anchor_frames() == []


def test_set_fill_reasserts_anchors(store):
    store.set_point(2, "beak", (100.0, 200.0))
    store.set_fill(np.zeros((10, 1, 3, 2)), np.full((10, 1, 3), 0.5))

    np.testing.assert_allclose(store.filled[2, 0, 0], [100.0, 200.0])
    assert store.confidence[2, 0, 0] == 1.0
    # Untouched keypoints keep the backend's values.
    np.testing.assert_allclose(store.filled[2, 0, 1], [0.0, 0.0])
    assert store.confidence[2, 0, 1] == 0.5


def test_fill_range_reports_what_the_fill_covered(store):
    """Backends fill the span between labels, and readers must not run past it."""
    assert store.fill_range is None

    filled = np.full((10, 1, 3, 2), np.nan)
    filled[3:8] = 1.0
    store.set_fill(filled, np.full((10, 1, 3), 0.5))
    assert store.fill_range == (3, 7)

    store.clear_fill()
    assert store.fill_range is None


def test_set_fill_rejects_wrong_shape(store):
    with pytest.raises(KeypointStoreError):
        store.set_fill(np.zeros((5, 1, 3, 2)), np.zeros((5, 1, 3)))


def test_positions_prefers_anchor_over_fill(store):
    store.set_fill(np.ones((10, 1, 3, 2)), np.ones((10, 1, 3)))
    store.set_point(6, "eye", (42.0, 43.0))
    np.testing.assert_allclose(store.positions_for(6)[2], [42.0, 43.0])
    np.testing.assert_allclose(store.positions_for(6)[0], [1.0, 1.0])


def test_sidecar_round_trip(store, tmp_path):
    store.set_point(1, "beak", (1.5, 2.5))
    store.set_point(8, "tail", (3.5, 4.5))
    path = tmp_path / "clip.mp4.keypoints.json"
    store.save(path)

    restored = KeypointStore.load(path)
    assert restored.keypoint_names == store.keypoint_names
    assert restored.individual_names == store.individual_names
    assert restored.n_frames == store.n_frames
    assert restored.anchor_frames() == [1, 8]
    np.testing.assert_allclose(restored.positions_for(1)[0], [1.5, 2.5])


def test_sidecar_path_sits_next_to_the_video(tmp_path):
    assert sidecar_path(tmp_path / "clip.mp4").name == "clip.mp4.keypoints.json"


def test_multi_individual_sidecar_round_trip(pair, tmp_path):
    pair.set_point(2, "beak", (1.0, 2.0), "crow_a")
    pair.set_point(2, "tail", (3.0, 4.0), "crow_b")
    path = tmp_path / "clip.mp4.keypoints.json"
    pair.save(path)

    restored = KeypointStore.load(path)
    assert restored.individual_names == ["crow_a", "crow_b"]
    np.testing.assert_allclose(restored.positions_for(2, "crow_a")[0], [1.0, 2.0])
    np.testing.assert_allclose(restored.positions_for(2, "crow_b")[1], [3.0, 4.0])


# ----------------------------------------------------------------------
# Legacy sidecars written before the individual/keypoint hierarchy
# ----------------------------------------------------------------------


def _legacy_payload() -> dict:
    """A pre-hierarchy sidecar: "names" key, 2-D (n_keypoints, 2) anchors."""
    return {
        "names": ["male", "female"],
        "n_frames": 2861,
        "anchors": {
            "89": [[510.7, 442.8], [464.8, 500.0]],
            "173": [[476.0, 522.4], [441.2, 640.0]],
        },
    }


def test_legacy_flat_sidecar_loads(tmp_path):
    """Regression: a flat sidecar raised KeyError('keypoint') and broke the dialog."""
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    restored = KeypointStore.load(path)

    assert restored.keypoint_names == ["male", "female"]
    assert restored.individual_names == ["individual_0"]
    assert restored.n_frames == 2861
    assert restored.anchor_frames() == [89, 173]


def test_legacy_anchors_gain_an_individual_axis(tmp_path):
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    restored = KeypointStore.load(path)

    assert restored.anchors[89].shape == (1, 2, 2)
    np.testing.assert_allclose(restored.positions_for(89)[0], [510.7, 442.8])
    np.testing.assert_allclose(restored.positions_for(89)[1], [464.8, 500.0])


def test_legacy_names_stay_keypoints(tmp_path):
    """Never silently reinterpret old names as individuals — that would change
    what the user labelled."""
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    restored = KeypointStore.load(path)

    assert restored.n_individuals == 1
    assert restored.n_keypoints == 2


def test_legacy_sidecar_resaves_in_the_new_format(tmp_path):
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    KeypointStore.load(path).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "names" not in payload
    assert payload["keypoint"] == ["male", "female"]
    assert payload["individual"] == ["individual_0"]


def test_sidecar_without_names_raises_clearly(tmp_path):
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps({"n_frames": 10, "anchors": {}}), encoding="utf-8")
    with pytest.raises(KeypointStoreError, match="keypoint names"):
        KeypointStore.load(path)


def test_mismatched_anchor_shape_raises(tmp_path):
    """A truncated/corrupt anchor row must fail loudly, not load half a frame."""
    payload = _legacy_payload()
    payload["anchors"]["89"] = [[1.0, 2.0]]  # one keypoint, schema says two
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KeypointStoreError, match="expected"):
        KeypointStore.load(path)


# ----------------------------------------------------------------------
# Hierarchy: several individuals sharing one keypoint schema
# ----------------------------------------------------------------------


def test_individuals_are_independent(pair):
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(0, "beak", (50.0, 50.0), "crow_b")

    np.testing.assert_allclose(pair.positions_for(0, "crow_a")[0], [1.0, 1.0])
    np.testing.assert_allclose(pair.positions_for(0, "crow_b")[0], [50.0, 50.0])
    assert pair.labelled_count(0) == 2
    assert pair.labelled_count(0, "crow_a") == 1


def test_anchor_frames_for_is_per_individual(pair):
    pair.set_point(1, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(4, "beak", (2.0, 2.0), "crow_b")
    assert pair.anchor_frames() == [1, 4]
    assert pair.anchor_frames_for("beak", "crow_a") == [1]
    assert pair.anchor_frames_for("beak", "crow_b") == [4]


def test_nearest_reports_the_individual(pair):
    pair.set_point(0, "tail", (10.0, 10.0), "crow_b")
    assert pair.nearest(0, (10.5, 10.0), radius=3.0) == ("crow_b", "tail")


def test_clear_individual_leaves_the_others_alone(pair):
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(0, "tail", (2.0, 2.0), "crow_a")
    pair.set_point(0, "beak", (3.0, 3.0), "crow_b")

    pair.clear_individual(0, "crow_a")

    assert pair.labelled_count(0, "crow_a") == 0
    np.testing.assert_allclose(pair.positions_for(0, "crow_b")[0], [3.0, 3.0])


def test_adding_an_individual_keeps_existing_points(store):
    store.set_point(0, "beak", (7.0, 8.0))
    store.set_individual_names(["individual_0", "individual_1"])

    assert store.n_points == 2 * len(NAMES)
    np.testing.assert_allclose(store.positions_for(0, "individual_0")[0], [7.0, 8.0])
    assert np.all(np.isnan(store.positions_for(0, "individual_1")))


def test_removing_an_individual_drops_its_points(pair):
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(0, "beak", (2.0, 2.0), "crow_b")

    pair.set_individual_names(["crow_a"])

    assert pair.n_individuals == 1
    np.testing.assert_allclose(pair.positions_for(0, "crow_a")[0], [1.0, 1.0])


def test_removing_every_individual_is_allowed(pair):
    """The dialog lets the last individual go; the store must not resurrect one."""
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")

    pair.set_individual_names([])

    assert pair.individual_names == []
    assert pair.n_individuals == 0
    assert pair.n_points == 0
    assert pair.anchor_frames() == []
    assert pair.positions(0).shape == (0, len(NAMES), 2)


def test_labelling_without_an_individual_raises(store):
    store.set_individual_names([])
    with pytest.raises(UnknownIndividualError):
        store.set_point(0, "beak", (1.0, 1.0))


def test_empty_individuals_survive_the_sidecar(store, tmp_path):
    store.set_individual_names([])
    path = tmp_path / "clip.mp4.keypoints.json"
    store.save(path)

    assert KeypointStore.load(path).individual_names == []


def test_flat_anchors_round_trip_through_a_fill(pair):
    pair.set_point(0, "beak", (1.0, 2.0), "crow_a")
    pair.set_point(0, "eye", (3.0, 4.0), "crow_b")

    flat = pair.flat_anchors()
    assert flat[0].shape == (pair.n_points, 2)
    np.testing.assert_allclose(flat[0][0], [1.0, 2.0])  # crow_a / beak
    np.testing.assert_allclose(flat[0][5], [3.0, 4.0])  # crow_b / eye

    filled = np.zeros((pair.n_frames, pair.n_points, 2))
    pair.set_fill_from_flat(filled, np.zeros((pair.n_frames, pair.n_points)))

    assert pair.filled.shape == (pair.n_frames, 2, len(NAMES), 2)
    np.testing.assert_allclose(pair.filled[0, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(pair.filled[0, 1, 2], [3.0, 4.0])


# ----------------------------------------------------------------------
# Reading points back: the dialog's table
# ----------------------------------------------------------------------


def test_labelled_points_are_sorted_by_frame(pair):
    pair.set_point(7, "tail", (3.0, 4.0), "crow_b")
    pair.set_point(2, "beak", (1.0, 2.0), "crow_a")

    assert pair.labelled_points() == [
        (2, "crow_a", "beak", 1.0, 2.0),
        (7, "crow_b", "tail", 3.0, 4.0),
    ]


def test_labelled_points_can_be_filtered(pair):
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(0, "tail", (2.0, 2.0), "crow_a")
    pair.set_point(0, "beak", (3.0, 3.0), "crow_b")

    assert [row[1:3] for row in pair.labelled_points(individual="crow_a")] == [
        ("crow_a", "beak"),
        ("crow_a", "tail"),
    ]
    assert [row[1] for row in pair.labelled_points(keypoint="beak")] == ["crow_a", "crow_b"]


def test_undo_reports_the_frame_it_changed(store):
    """An undo can land anywhere, so the caller is told what to redraw."""
    store.set_point(4, "beak", (1.0, 1.0))
    store.set_point(9, "tail", (2.0, 2.0))

    assert store.undo() == 9
    assert store.undo() == 4
    assert store.undo() is None  # nothing left in the history


# ----------------------------------------------------------------------
# Provenance: human labels vs filled predictions
# ----------------------------------------------------------------------


def _fill(store: KeypointStore, value: float = 5.0) -> None:
    shape = (store.n_frames, store.n_individuals, store.n_keypoints)
    store.set_fill(np.full((*shape, 2), value), np.full(shape, 0.5))


def test_human_mask_marks_only_placed_points(store):
    store.set_point(2, "tail", (1.0, 2.0))
    _fill(store)

    assert store.human_mask(2).tolist() == [[False, True, False]]
    assert not store.human_mask(3).any()  # frame 3 is entirely the backend's


def test_is_human_needs_one_point_anywhere_in_the_row(pair):
    _fill(pair)
    pair.set_point(2, "tail", (1.0, 2.0), "crow_b")

    assert pair.is_human(2, "crow_b") is True
    assert pair.is_human(2, "crow_a") is False
    assert pair.is_human(2) is True  # any individual counts for the frame


def test_is_anchor_distinguishes_a_label_from_a_fill(store):
    store.set_point(2, "tail", (1.0, 2.0))
    _fill(store)

    assert store.is_anchor(2, "tail") is True
    assert store.is_anchor(2, "beak") is False  # filled, not labelled
    assert not np.isnan(store.positions_for(2)[0, 0])  # but it does have a position


def test_nearest_ignores_filled_points_unless_asked(store):
    _fill(store, value=10.0)

    assert store.nearest(0, (10.0, 10.0), radius=3.0) is None
    assert store.nearest(0, (10.0, 10.0), radius=3.0, include_fill=True) == ("individual_0", "beak")


def test_promote_fill_turns_predictions_into_labels(store):
    _fill(store, value=7.0)

    assert store.promote_fill(3) == len(NAMES)
    assert store.is_human(3) is True
    np.testing.assert_allclose(store.anchor_positions(3)[0, 1], [7.0, 7.0])


def test_promote_fill_never_overwrites_a_label(store):
    store.set_point(3, "beak", (1.0, 2.0))
    _fill(store, value=7.0)

    store.promote_fill(3)

    np.testing.assert_allclose(store.anchor_positions(3)[0, 0], [1.0, 2.0])


def test_promote_fill_can_take_one_individual(pair):
    _fill(pair)

    pair.promote_fill(3, "crow_b")

    assert pair.is_human(3, "crow_b") is True
    assert pair.is_human(3, "crow_a") is False


def test_clear_fill_for_removes_one_rows_predictions(store):
    """The counterpart of pinning: the animal is not in this frame at all."""
    store.set_point(3, "beak", (1.0, 2.0))
    _fill(store, value=7.0)

    assert store.clear_fill_for(4) == len(NAMES)

    assert store.has_fill(4) is False
    assert np.all(np.isnan(store.positions(4)))
    assert store.has_fill(5) is True  # its neighbours are untouched


def test_clear_fill_for_keeps_the_labels(store):
    store.set_point(3, "beak", (1.0, 2.0))
    _fill(store, value=7.0)

    store.clear_fill_for(3)

    np.testing.assert_allclose(store.positions_for(3)[0], [1.0, 2.0])
    assert store.is_human(3) is True
    assert np.all(np.isnan(store.positions_for(3)[1]))  # the predicted ones are gone


def test_clear_fill_for_can_take_one_individual(pair):
    _fill(pair)

    pair.clear_fill_for(3, "crow_b")

    assert pair.has_fill(3, "crow_b") is False
    assert pair.has_fill(3, "crow_a") is True


def test_clear_fill_for_without_a_fill_does_nothing(store):
    assert store.clear_fill_for(3) == 0
    assert store.has_fill(3) is False


def test_promote_fill_without_a_fill_does_nothing(store):
    assert store.promote_fill(3) == 0
    assert store.anchor_frames() == []


def test_refilling_never_feeds_on_the_previous_fill(store):
    """The invariant behind the Human/Fill split: fills come from labels only."""
    store.set_point(2, "beak", (1.0, 2.0))
    _fill(store, value=7.0)

    anchors = store.flat_anchors()
    assert list(anchors) == [2]  # not one entry per filled frame
    np.testing.assert_allclose(anchors[2][0], [1.0, 2.0])


# ----------------------------------------------------------------------
# Asymmetric schemas: individuals with their own keypoints
# ----------------------------------------------------------------------


def test_keypoints_are_shared_by_default(pair):
    assert pair.shared_keypoints is True
    assert pair.keypoints_for("crow_a") == NAMES
    assert pair.n_schema_points == pair.n_points


def test_unsharing_starts_everyone_on_the_full_schema(pair):
    pair.set_shared_keypoints(False)
    assert pair.keypoints_for("crow_a") == NAMES
    assert pair.keypoints_for("crow_b") == NAMES
    assert pair.n_schema_points == pair.n_points


def test_unsharing_keeps_labelled_points(pair):
    pair.set_point(0, "beak", (1.0, 2.0), "crow_a")
    pair.set_shared_keypoints(False)
    np.testing.assert_allclose(pair.positions_for(0, "crow_a")[0], [1.0, 2.0])


def test_set_keypoints_for_narrows_one_individual(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])

    assert pair.keypoints_for("crow_a") == ["beak"]
    assert pair.keypoints_for("crow_b") == NAMES
    assert pair.keypoint_names == NAMES  # the union is untouched
    assert pair.n_schema_points == 1 + len(NAMES)


def test_set_keypoints_for_adds_new_names_to_the_union_only(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_b", [*NAMES, "wing"])

    assert pair.keypoint_names == [*NAMES, "wing"]
    assert pair.keypoints_for("crow_a") == NAMES
    assert pair.keypoints_for("crow_b") == [*NAMES, "wing"]


def test_narrowing_drops_that_individuals_points(pair):
    pair.set_point(0, "eye", (1.0, 1.0), "crow_a")
    pair.set_point(0, "eye", (2.0, 2.0), "crow_b")
    pair.set_shared_keypoints(False)

    pair.set_keypoints_for("crow_a", ["beak", "tail"])

    assert np.all(np.isnan(pair.positions_for(0, "crow_a")))
    np.testing.assert_allclose(pair.positions_for(0, "crow_b")[2], [2.0, 2.0])


def test_labelling_a_keypoint_the_individual_lacks_raises(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])

    with pytest.raises(UnknownKeypointError):
        pair.set_point(0, "tail", (1.0, 1.0), "crow_a")
    pair.set_point(0, "tail", (1.0, 1.0), "crow_b")  # still fine for the other


def test_set_keypoints_for_refuses_while_shared(pair):
    with pytest.raises(KeypointStoreError):
        pair.set_keypoints_for("crow_a", ["beak"])


def test_fill_blanks_points_outside_an_individuals_schema(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])
    pair.set_fill_from_flat(
        np.ones((pair.n_frames, pair.n_points, 2)),
        np.full((pair.n_frames, pair.n_points), 0.5),
    )

    assert np.all(np.isnan(pair.filled[:, 0, 1:]))  # crow_a: tail, eye
    assert np.all(np.isnan(pair.confidence[:, 0, 1:]))
    np.testing.assert_allclose(pair.filled[:, 0, 0], 1.0)
    np.testing.assert_allclose(pair.filled[:, 1], 1.0)


def test_asymmetric_schema_survives_the_sidecar(pair, tmp_path):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])
    pair.set_point(0, "beak", (1.0, 2.0), "crow_a")
    path = tmp_path / "clip.mp4.keypoints.json"
    pair.save(path)

    restored = KeypointStore.load(path)

    assert restored.shared_keypoints is False
    assert restored.keypoints_for("crow_a") == ["beak"]
    assert restored.keypoints_for("crow_b") == NAMES
    np.testing.assert_allclose(restored.positions_for(0, "crow_a")[0], [1.0, 2.0])


def test_a_new_individual_gets_the_whole_schema(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])
    pair.set_individual_names([*pair.individual_names, "crow_c"])

    assert pair.keypoints_for("crow_c") == NAMES
    assert pair.keypoints_for("crow_a") == ["beak"]


def test_resharing_readmits_every_keypoint(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])
    pair.set_shared_keypoints(True)

    assert pair.keypoints_for("crow_a") == NAMES
    assert pair.keypoint_sets == {}
    pair.set_point(0, "eye", (1.0, 1.0), "crow_a")


def test_legacy_sidecars_are_shared(tmp_path):
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    assert KeypointStore.load(path).shared_keypoints is True


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


def test_store_to_movement_ds_axis_order(store):
    """x goes to space="x" and y to space="y" — the one place the swap lives."""
    store.set_point(0, "beak", (11.0, 22.0))
    ds = store_to_movement_ds(store, fps=FPS)

    assert ds.position.dims == ("time", "space", "keypoint", "individual")
    assert list(ds.coords["space"].values) == ["x", "y"]
    assert ds.position.sel(space="x", keypoint="beak").isel(time=0, individual=0) == 11.0
    assert ds.position.sel(space="y", keypoint="beak").isel(time=0, individual=0) == 22.0


def test_store_to_movement_ds_dims_are_singular(pair):
    ds = store_to_movement_ds(pair, fps=FPS)
    assert set(ds.dims) == {"time", "space", "keypoint", "individual"}
    assert list(ds.coords["individual"].values) == ["crow_a", "crow_b"]


def test_store_to_movement_ds_keeps_individuals_apart(pair):
    pair.set_point(2, "beak", (1.0, 2.0), "crow_a")
    pair.set_point(2, "beak", (30.0, 40.0), "crow_b")
    ds = store_to_movement_ds(pair, fps=FPS)

    assert ds.position.sel(space="x", keypoint="beak", individual="crow_a").isel(time=2) == 1.0
    assert ds.position.sel(space="x", keypoint="beak", individual="crow_b").isel(time=2) == 30.0


def test_store_to_movement_ds_carries_the_fill(pair):
    pair.set_fill_from_flat(
        np.arange(pair.n_frames * pair.n_points * 2, dtype=float).reshape(pair.n_frames, pair.n_points, 2),
        np.full((pair.n_frames, pair.n_points), 0.25),
    )
    ds = store_to_movement_ds(pair, fps=FPS)

    expected = pair.filled[3, 1, 2]
    got = ds.position.isel(time=3, individual=1).sel(keypoint=NAMES[2]).values
    np.testing.assert_allclose(got, expected)
    assert ds.confidence.isel(time=3, individual=1, keypoint=2) == 0.25


def test_store_to_movement_ds_time_uses_fps(store):
    ds = store_to_movement_ds(store, fps=FPS)
    assert ds.sizes["time"] == store.n_frames
    np.testing.assert_allclose(ds.coords["time"].values[1], 1.0 / FPS)


def test_store_to_movement_ds_anchor_confidence_is_one(store):
    store.set_point(3, "tail", (1.0, 2.0))
    ds = store_to_movement_ds(store, fps=FPS)
    assert ds.confidence.sel(keypoint="tail").isel(time=3, individual=0) == 1.0
    assert np.isnan(ds.confidence.sel(keypoint="tail").isel(time=4, individual=0))


def test_store_to_movement_ds_rejects_missing_fps(store):
    with pytest.raises(KeypointStoreError):
        store_to_movement_ds(store, fps=0.0)


def test_store_to_movement_ds_round_trips_through_pose_render(pair):
    """The exported dataset must survive the normal display pipeline."""
    from ethograph.gui.pose_render import movement_ds_to_pose_render

    pair.set_point(0, "beak", (11.0, 22.0), "crow_a")
    pair.set_point(4, "tail", (33.0, 44.0), "crow_b")
    pr = movement_ds_to_pose_render(store_to_movement_ds(pair, fps=FPS), "test")

    # points rows are (track_id, frame, y, x) — the swap back out again.
    rows = pr.data[pr.data_not_nan]
    assert len(rows) == 2
    assert {(row[2], row[3]) for row in rows} == {(22.0, 11.0), (44.0, 33.0)}
    assert set(pr.properties["individual"].unique()) == {"crow_a", "crow_b"}


# ----------------------------------------------------------------------
# Derived kinematics ("Load into the GUI")
# ----------------------------------------------------------------------


def _moving_store(step: float = 10.0, n: int = 11) -> KeypointStore:
    """A store whose fill moves every keypoint *step* px per frame in x."""
    store = KeypointStore(keypoint_names=["beak", "tail"], n_frames=n, individual_names=["bird"])
    filled = np.zeros((n, 1, 2, 2))
    filled[:, 0, :, 0] = (np.arange(n) * step)[:, None]
    store.set_fill(filled, np.ones((n, 1, 2)))
    return store


def test_kinematics_are_named_and_shaped_for_the_gui():
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_moving_store(), FPS), KINEMATICS)

    assert set(arrays) == {f"keypoints_{n}" for n in ("position", *KINEMATICS)}
    assert arrays["keypoints_velocity"].dims == (FEATURE_TIME_DIM, "space", "keypoint", "individual")
    # Speed is a magnitude, so the space axis is gone.
    assert arrays["keypoints_speed"].dims == (FEATURE_TIME_DIM, "keypoint", "individual")


def test_kinematics_rename_time_so_a_merge_cannot_outer_join():
    """The keypoints run at video fps, the trial's `time` almost never does."""
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_moving_store(), FPS), ["speed"])
    for array in arrays.values():
        assert "time" not in array.dims
        assert FEATURE_TIME_DIM in array.dims


def test_speed_is_in_units_per_second_not_per_frame():
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_moving_store(step=10.0), FPS), ["speed"])
    speed = arrays["keypoints_speed"].isel({FEATURE_TIME_DIM: 5, "keypoint": 0, "individual": 0})
    np.testing.assert_allclose(float(speed), 10.0 * FPS)


def test_constant_velocity_has_no_acceleration():
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_moving_store(), FPS), ["acceleration"])
    middle = arrays["keypoints_acceleration"].isel({FEATURE_TIME_DIM: 5})
    np.testing.assert_allclose(middle.values, 0.0, atol=1e-9)


def test_kinematics_use_the_filled_frames():
    """Inspecting a fill is the point — anchors alone would be almost all NaN."""
    pytest.importorskip("movement")
    store = _moving_store()
    assert len(store.anchor_frames()) == 0  # nothing labelled by hand
    speed = store_to_kinematics(store_to_movement_ds(store, FPS), ["speed"])["keypoints_speed"]
    assert not np.any(np.isnan(speed.values[1:-1]))


def test_position_is_always_included():
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_moving_store(), FPS), [])
    assert set(arrays) == {"keypoints_position"}


def test_unknown_kinematic_raises():
    pytest.importorskip("movement")
    with pytest.raises(KeypointStoreError, match="Unknown kinematic"):
        store_to_kinematics(store_to_movement_ds(_moving_store(), FPS), ["jerk"])


def test_time_axis_is_the_videos_frame_grid():
    """The keypoints are a dataset in their own right: one sample per frame,
    spaced by 1/fps, so nothing else has to exist for them to be plottable."""
    store = _moving_store(n=11)
    ds = store_to_movement_ds(store, FPS)
    time = ds.coords["time"].values

    assert len(time) == store.n_frames
    np.testing.assert_allclose(time, np.arange(store.n_frames) / FPS)
    np.testing.assert_allclose(np.diff(time), 1.0 / FPS)


def test_kinematics_keep_the_frame_grid():
    pytest.importorskip("movement")
    store = _moving_store(n=11)
    arrays = store_to_kinematics(store_to_movement_ds(store, FPS), ["speed"])
    time = arrays["keypoints_speed"].coords[FEATURE_TIME_DIM].values
    np.testing.assert_allclose(time, np.arange(store.n_frames) / FPS)


# ----------------------------------------------------------------------
# y-flip: image coordinates (y down) -> plot coordinates (y up)
# ----------------------------------------------------------------------

IMAGE_HEIGHT = 480.0


def test_image_coordinates_are_kept_by_default(store):
    """The overlay and the DLC export both need raw image coordinates."""
    store.set_point(0, "beak", (10.0, 30.0))
    ds = store_to_movement_ds(store, FPS)
    assert float(ds.position.sel(space="y", keypoint="beak").isel(time=0, individual=0)) == 30.0


def test_flipping_puts_the_top_of_the_frame_at_the_top_of_the_plot(store):
    """A keypoint near the top (small y in image space) must end up with a LARGE
    y, since plots count upward."""
    store.set_point(0, "beak", (10.0, 30.0))
    ds = store_to_movement_ds(store, FPS, image_height=IMAGE_HEIGHT)
    assert float(ds.position.sel(space="y", keypoint="beak").isel(time=0, individual=0)) == IMAGE_HEIGHT - 30.0


def test_flipping_leaves_x_alone(store):
    store.set_point(0, "beak", (10.0, 30.0))
    ds = store_to_movement_ds(store, FPS, image_height=IMAGE_HEIGHT)
    assert float(ds.position.sel(space="x", keypoint="beak").isel(time=0, individual=0)) == 10.0


def test_flipping_preserves_vertical_ordering(store):
    """Whatever is higher in the video stays higher in the plot."""
    store.set_point(0, "beak", (0.0, 10.0))  # near the top of the frame
    store.set_point(0, "tail", (0.0, 400.0))  # near the bottom
    ds = store_to_movement_ds(store, FPS, image_height=IMAGE_HEIGHT)
    y = ds.position.sel(space="y").isel(time=0, individual=0)
    assert float(y.sel(keypoint="beak")) > float(y.sel(keypoint="tail"))


def test_flipping_applies_to_filled_frames_too(store):
    filled = np.full((store.n_frames, 1, store.n_keypoints, 2), 100.0)
    store.set_fill(filled, np.ones((store.n_frames, 1, store.n_keypoints)))
    ds = store_to_movement_ds(store, FPS, image_height=IMAGE_HEIGHT)
    y = ds.position.sel(space="y").isel(time=3, individual=0, keypoint=0)
    assert float(y) == IMAGE_HEIGHT - 100.0


def test_flip_rejects_a_missing_image_height(store):
    """A zero height would silently mirror everything about y=0."""
    with pytest.raises(KeypointStoreError, match="image_height must be positive"):
        store_to_movement_ds(store, FPS, image_height=0.0)


def test_flipped_velocity_changes_sign(store):
    """Kinematics follow the flip, so velocity's y sign matches what is drawn."""
    pytest.importorskip("movement")
    filled = np.zeros((11, 1, 2, 2))
    filled[:, 0, :, 1] = (np.arange(11) * 5.0)[:, None]  # moving DOWN the image
    down = KeypointStore(keypoint_names=["beak", "tail"], n_frames=11, individual_names=["bird"])
    down.set_fill(filled, np.ones((11, 1, 2)))

    raw = store_to_kinematics(store_to_movement_ds(down, FPS), ["velocity"])
    flipped = store_to_kinematics(store_to_movement_ds(down, FPS, image_height=IMAGE_HEIGHT), ["velocity"])

    y_raw = raw["keypoints_velocity"].sel(space="y").isel({FEATURE_TIME_DIM: 5, "keypoint": 0, "individual": 0})
    y_flip = flipped["keypoints_velocity"].sel(space="y").isel({FEATURE_TIME_DIM: 5, "keypoint": 0, "individual": 0})
    assert float(y_raw) > 0  # y grows downward in image space
    np.testing.assert_allclose(float(y_flip), -float(y_raw))
