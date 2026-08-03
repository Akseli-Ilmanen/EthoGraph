"""Canvas keypoint editing: click, drag, delete and undo semantics.

Driven through a fake camera view, so no GPU canvas is needed — the pygfx
objects the overlay builds are created headless and never rendered.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygfx")

import pygfx as gfx  # noqa: E402

from ethograph.gui.pose_annotate import KeypointStore  # noqa: E402
from ethograph.gui.pose_edit_mixin import (  # noqa: E402
    HIT_RADIUS_PX,
    LOOP_MODE,
    SEQUENTIAL_MODE,
    KeypointLabelMode,
    glyph_for_individual,
    marker_for_individual,
)

NAMES = ["beak", "tail", "eye"]


class _FakeView:
    """The slice of CameraView that KeypointLabelMode actually touches."""

    def __init__(self):
        self._scene = gfx.Scene()
        self.label_mode = None
        self.draws = 0

    def scene(self):
        return self._scene

    def image_height(self):
        return 480.0

    def image_units_per_pixel(self):
        return 1.0

    def set_label_mode(self, mode):
        self.label_mode = mode

    def request_draw(self):
        self.draws += 1


@pytest.fixture
def mode():
    view = _FakeView()
    store = KeypointStore(keypoint_names=list(NAMES), n_frames=10)
    return KeypointLabelMode(view, store)


@pytest.fixture
def loop_mode():
    """Loop mode: one keypoint, advancing the frame after each click."""
    view = _FakeView()
    store = KeypointStore(keypoint_names=list(NAMES), n_frames=10)
    return KeypointLabelMode(view, store, mode=LOOP_MODE)


@pytest.fixture
def pair_mode():
    """Two individuals sharing the schema, as SLEAP's instances do."""
    view = _FakeView()
    store = KeypointStore(keypoint_names=list(NAMES), n_frames=10, individual_names=["crow_a", "crow_b"])
    return KeypointLabelMode(view, store)


def test_attaching_registers_with_the_view(mode):
    assert mode.view.label_mode is mode
    assert mode.active_keypoint == "beak"


def test_click_places_the_active_keypoint(mode):
    mode.handle_click(100.0, 200.0)
    np.testing.assert_allclose(mode.store.positions_for(0)[0], [100.0, 200.0])


def test_click_advances_to_the_next_unlabelled_keypoint(mode):
    # Well separated, so the second click places rather than hitting the first.
    mode.handle_click(10.0, 10.0)
    assert mode.active_keypoint == "tail"
    mode.handle_release(10.0, 10.0)
    mode.handle_click(200.0, 200.0)
    assert mode.active_keypoint == "eye"


def test_advance_skips_keypoints_already_placed_on_this_frame(mode):
    mode.store.set_point(0, "tail", (300.0, 300.0))
    mode.handle_click(10.0, 10.0)  # places beak
    assert mode.active_keypoint == "eye"


def test_advance_always_returns_to_the_leftmost_gap(mode):
    """Sequential fills the points table left to right, in schema order."""
    # Label out of order: 'eye' (index 2) placed first, by hand.
    mode.set_active("eye")
    mode.handle_click(10.0, 10.0)
    mode.handle_release(10.0, 10.0)

    # The next click must go to 'beak' (index 0), not carry on past 'eye'.
    assert mode.active_keypoint == "beak"


def test_advance_skips_over_a_filled_gap(mode):
    mode.store.set_point(0, "beak", (5.0, 5.0))
    mode.set_active("tail")
    mode.handle_click(200.0, 200.0)  # places tail
    assert mode.active_keypoint == "eye"


def test_active_keypoint_stays_when_the_frame_is_complete(mode):
    for name in NAMES:
        mode.store.set_point(0, name, (10.0 * NAMES.index(name) + 100, 50.0))
    mode.set_active("tail")
    mode._advance_to_unlabelled()
    assert mode.active_keypoint == "tail"


def test_cycle_wraps_around(mode):
    mode.cycle(1)
    assert mode.active_keypoint == "tail"
    mode.cycle(-1)
    assert mode.active_keypoint == "beak"
    mode.cycle(-1)
    assert mode.active_keypoint == "eye"


def test_clicking_an_existing_point_grabs_it_instead_of_placing(mode):
    """Editing needs no mode: a click on a point always selects and drags it."""
    mode.store.set_point(0, "eye", (50.0, 50.0))
    mode.refresh()
    mode.handle_click(51.0, 50.0)  # active keypoint is still 'beak'

    assert mode.active_keypoint == "eye"
    assert mode.store.labelled_count(0) == 1
    np.testing.assert_allclose(mode.store.positions_for(0)[2], [50.0, 50.0])


def _fill(store, value: float = 5.0) -> None:
    """A backend result covering every frame, so predictions exist to grab."""
    shape = (store.n_frames, store.n_individuals, store.n_keypoints)
    store.set_fill(np.full((*shape, 2), value), np.full(shape, 0.5))


def test_clicking_a_filled_point_pins_it_as_a_label(mode):
    """Accepting a prediction: the next fill must treat it as ground truth."""
    _fill(mode.store, value=50.0)
    mode.refresh()

    mode.handle_click(51.0, 50.0)

    assert mode.store.is_anchor(0, "beak") is True
    np.testing.assert_allclose(mode.store.anchor_positions(0)[0, 0], [50.0, 50.0])


def test_dragging_a_filled_point_corrects_it_in_one_undo_step(mode):
    _fill(mode.store, value=50.0)
    mode.refresh()

    mode.handle_click(51.0, 50.0)
    mode.handle_move(80.0, 90.0)
    mode.handle_release(80.0, 90.0)

    np.testing.assert_allclose(mode.store.anchor_positions(0)[0, 0], [80.0, 90.0])
    mode.store.undo()
    assert mode.store.is_anchor(0, "beak") is False  # back to being the fill's


def test_a_filled_point_cannot_be_deleted(mode):
    """There is nothing to remove from a prediction — only a label can go."""
    _fill(mode.store, value=50.0)
    mode.refresh()
    mode._cursor = (50.0, 50.0)

    assert mode.delete_selected() is False
    assert mode.store.anchor_frames() == []


def test_clicking_empty_space_still_places(mode):
    mode.store.set_point(0, "eye", (50.0, 50.0))
    mode.refresh()
    mode.handle_click(300.0, 300.0)

    assert mode.store.labelled_count(0) == 2
    np.testing.assert_allclose(mode.store.positions_for(0)[0], [300.0, 300.0])


def test_switching_mode_stops_a_drag(mode):
    mode.handle_click(10.0, 10.0)
    mode.set_mode(LOOP_MODE)
    mode.handle_move(99.0, 99.0)

    np.testing.assert_allclose(mode.store.positions_for(0)[0], [10.0, 10.0])


def test_set_mode_rejects_nonsense(mode):
    with pytest.raises(ValueError):
        mode.set_mode("scribble")
    assert mode.mode == SEQUENTIAL_MODE


def test_sequential_mode_never_navigates(mode):
    """The playhead only moves when the user moves it."""
    calls = []
    mode.on_advance_frame = lambda: calls.append(1)
    mode.handle_click(10.0, 10.0)
    assert calls == []
    assert mode.active_keypoint == "tail"


def test_loop_mode_advances_the_frame_and_keeps_the_keypoint(loop_mode):
    calls = []
    loop_mode.on_advance_frame = lambda: calls.append(1)
    loop_mode.handle_click(10.0, 10.0)

    assert calls == [1]
    assert loop_mode.active_keypoint == "beak"


def test_loop_mode_still_grabs_an_existing_point_without_advancing(loop_mode):
    loop_mode.store.set_point(0, "beak", (50.0, 50.0))
    loop_mode.refresh()
    calls = []
    loop_mode.on_advance_frame = lambda: calls.append(1)
    loop_mode.handle_click(51.0, 50.0)

    assert calls == []
    assert loop_mode.store.labelled_count(0) == 1


def test_selecting_without_dragging_leaves_no_undo_step(mode):
    """Regression: the click that selects must not make undo pop an older edit."""
    mode.store.set_point(0, "beak", (10.0, 10.0))
    mode.store.set_point(0, "tail", (200.0, 200.0))

    mode.handle_click(200.0, 200.0)  # select 'tail'
    mode.handle_release(200.0, 200.0)
    mode.store.undo()  # undoes placing 'tail', not something older

    assert np.isnan(mode.store.positions_for(0)[1, 0])
    np.testing.assert_allclose(mode.store.positions_for(0)[0], [10.0, 10.0])


def test_dragging_an_existing_anchor_collapses_into_one_undo(mode):
    mode.store.set_point(0, "beak", (10.0, 10.0))

    mode.handle_click(10.0, 10.0)
    mode.handle_move(30.0, 30.0)
    mode.handle_move(60.0, 60.0)
    mode.handle_release(60.0, 60.0)
    np.testing.assert_allclose(mode.store.positions_for(0)[0], [60.0, 60.0])

    mode.store.undo()
    np.testing.assert_allclose(mode.store.positions_for(0)[0], [10.0, 10.0])


def test_dragging_a_freshly_placed_point_collapses_into_one_undo(mode):
    mode.handle_click(10.0, 10.0)
    mode.handle_move(40.0, 40.0)
    mode.handle_move(70.0, 70.0)
    mode.handle_release(70.0, 70.0)
    np.testing.assert_allclose(mode.store.positions_for(0)[0], [70.0, 70.0])

    mode.store.undo()
    assert mode.store.anchor_frames() == []


def test_move_without_a_drag_only_tracks_the_cursor(mode):
    mode.handle_move(5.0, 5.0)
    assert mode.store.anchor_frames() == []


def test_delete_under_cursor(mode):
    mode.handle_click(80.0, 80.0)
    mode.handle_release(80.0, 80.0)
    assert mode.delete_under_cursor() is True
    assert mode.store.anchor_frames() == []


def test_delete_under_cursor_misses_when_far_away(mode):
    mode.handle_click(80.0, 80.0)
    mode.handle_release(80.0, 80.0)
    mode.handle_move(80.0 + 10 * HIT_RADIUS_PX, 80.0)
    assert mode.delete_under_cursor() is False
    assert mode.store.anchor_frames() == [0]


def test_backspace_deletes_the_selected_point(mode):
    """The outline says which point is active; Backspace removes exactly that."""
    mode.store.set_point(0, "eye", (300.0, 300.0))
    mode.set_active("eye")
    mode.handle_move(10.0, 10.0)  # cursor nowhere near it

    assert mode.delete_selected() is True
    assert mode.store.anchor_frames() == []


def test_backspace_prefers_the_selection_over_the_hovered_point(mode):
    mode.store.set_point(0, "beak", (10.0, 10.0))
    mode.store.set_point(0, "eye", (300.0, 300.0))
    mode.set_active("eye")
    mode.handle_move(10.0, 10.0)  # hovering 'beak'

    mode.delete_selected()

    assert np.isnan(mode.store.positions_for(0)[2, 0])  # eye gone
    np.testing.assert_allclose(mode.store.positions_for(0)[0], [10.0, 10.0])  # beak kept


def test_backspace_falls_back_to_the_hovered_point(mode):
    """With the active pair unlabelled here, hover-and-delete still works."""
    mode.store.set_point(0, "eye", (80.0, 80.0))
    mode.set_active("beak")
    mode.handle_move(80.0, 80.0)

    assert mode.delete_selected() is True
    assert mode.store.anchor_frames() == []


def test_backspace_on_nothing_is_a_noop(mode):
    mode.handle_move(500.0, 500.0)
    assert mode.delete_selected() is False


def test_the_active_marker_is_an_outline_not_a_disc(mode):
    material = mode._overlay._active.material
    assert material.marker == "circle"
    assert material.color.a == 0.0  # no fill, so the keypoint stays visible
    assert material.edge_width == 1.0


def test_points_land_on_the_frame_that_is_showing(mode):
    mode.set_frame(4)
    mode.handle_click(9.0, 9.0)
    assert mode.store.anchor_frames() == [4]


def test_changing_frame_ends_any_drag(mode):
    mode.handle_click(10.0, 10.0)
    mode.set_frame(3)
    mode.handle_move(99.0, 99.0)
    np.testing.assert_allclose(mode.store.positions_for(0)[0], [10.0, 10.0])


def test_on_changed_fires_for_edits_only(mode):
    calls = []
    mode.on_changed = lambda: calls.append(1)

    mode.handle_move(1.0, 1.0)
    assert calls == []
    mode.handle_click(1.0, 1.0)
    assert len(calls) == 1


def test_detach_releases_the_view(mode):
    mode.detach()
    assert mode.view.label_mode is None


def test_mode_without_keypoints_places_nothing():
    view = _FakeView()
    mode = KeypointLabelMode(view, KeypointStore(keypoint_names=[], n_frames=5))
    assert mode.active_keypoint is None
    mode.handle_click(1.0, 1.0)
    assert mode.store.anchor_frames() == []


# ----------------------------------------------------------------------
# Several individuals
# ----------------------------------------------------------------------


def test_clicks_land_on_the_active_individual(pair_mode):
    pair_mode.handle_click(10.0, 10.0)
    pair_mode.handle_release(10.0, 10.0)
    assert pair_mode.active_individual == "crow_a"
    assert pair_mode.store.labelled_count(0, "crow_a") == 1
    assert pair_mode.store.labelled_count(0, "crow_b") == 0


def test_number_key_selects_the_individual(pair_mode):
    assert pair_mode.select_individual_by_number(2) is True
    assert pair_mode.active_individual == "crow_b"
    assert pair_mode.select_individual_by_number(3) is False

    pair_mode.handle_click(10.0, 10.0)
    assert pair_mode.store.labelled_count(0, "crow_b") == 1


def test_switching_individual_restarts_the_keypoint_run(pair_mode):
    """Each individual is labelled through the same schema, independently."""
    pair_mode.handle_click(10.0, 10.0)  # crow_a / beak
    pair_mode.handle_release(10.0, 10.0)
    assert pair_mode.active_keypoint == "tail"

    pair_mode.select_individual_by_number(2)
    pair_mode.handle_click(200.0, 200.0)  # crow_b / tail
    pair_mode.handle_release(200.0, 200.0)

    np.testing.assert_allclose(pair_mode.store.positions_for(0, "crow_a")[0], [10.0, 10.0])
    np.testing.assert_allclose(pair_mode.store.positions_for(0, "crow_b")[1], [200.0, 200.0])


def test_advance_only_skips_this_individuals_points(pair_mode):
    pair_mode.store.set_point(0, "tail", (300.0, 300.0), "crow_b")
    pair_mode.handle_click(10.0, 10.0)  # crow_a / beak
    # 'tail' is taken on crow_b, not on crow_a, so crow_a still advances to it.
    assert pair_mode.active_keypoint == "tail"


def test_mode_without_individuals_places_nothing():
    view = _FakeView()
    store = KeypointStore(keypoint_names=list(NAMES), n_frames=10, individual_names=[])
    mode = KeypointLabelMode(view, store)

    assert mode.active_individual is None
    assert mode.active_keypoint is None
    mode.handle_click(1.0, 1.0)
    mode.cycle(1)
    assert mode.store.anchor_frames() == []
    assert mode.select_individual_by_number(1) is False


def test_tab_cycles_only_this_individuals_keypoints(pair_mode):
    pair_mode.store.set_shared_keypoints(False)
    pair_mode.store.set_keypoints_for("crow_a", ["beak", "eye"])
    pair_mode.set_active_individual("crow_a")

    assert pair_mode.active_keypoint == "beak"
    pair_mode.cycle(1)
    assert pair_mode.active_keypoint == "eye"
    pair_mode.cycle(1)
    assert pair_mode.active_keypoint == "beak"


def test_switching_individual_moves_off_a_keypoint_it_lacks(pair_mode):
    pair_mode.store.set_shared_keypoints(False)
    pair_mode.store.set_keypoints_for("crow_b", ["eye"])
    pair_mode.set_active("tail", "crow_a")

    pair_mode.select_individual_by_number(2)

    assert pair_mode.active_individual == "crow_b"
    assert pair_mode.active_keypoint == "eye"


def test_clicking_another_individuals_point_selects_that_individual(pair_mode):
    pair_mode.store.set_point(0, "eye", (50.0, 50.0), "crow_b")
    pair_mode.refresh()

    pair_mode.handle_click(51.0, 50.0)

    assert pair_mode.active_individual == "crow_b"
    assert pair_mode.active_keypoint == "eye"
    assert pair_mode.store.labelled_count(0) == 1


# ----------------------------------------------------------------------
# Shape per individual, colour per keypoint
# ----------------------------------------------------------------------


def test_each_individual_gets_its_own_marker_shape():
    shapes = {marker_for_individual(i) for i in range(4)}
    assert len(shapes) == 4
    assert marker_for_individual(0) == "circle"


def test_marker_shapes_wrap_around():
    from ethograph.gui.pose_edit_mixin import MARKER_SHAPES

    assert marker_for_individual(len(MARKER_SHAPES)) == marker_for_individual(0)
    assert glyph_for_individual(0) == "●"


def test_overlay_draws_one_layer_per_individual(pair_mode):
    shapes = [layer.material.marker for layer in pair_mode._overlay._layers]
    assert shapes == [marker_for_individual(0), marker_for_individual(1)]


def test_overlay_colours_vertices_by_keypoint(pair_mode):
    colors = pair_mode._overlay._layers[0].geometry.colors.data
    assert len(colors) == len(NAMES)
    assert not np.allclose(colors[0], colors[1])


# ----------------------------------------------------------------------
# Marker size
# ----------------------------------------------------------------------


def test_markers_are_sized_in_screen_space(mode):
    """Constant on screen: zooming the canvas must not resize the markers."""
    overlay = mode._overlay
    assert overlay._layers[0].material.size_space == "screen"
    assert overlay._active.material.size_space == "screen"


def test_set_point_size_resizes_every_layer(mode):
    mode.set_point_size(30)
    assert mode.point_size == 30
    for layer in mode._overlay._layers:
        assert layer.material.size == 30


def test_the_active_ring_stays_larger_than_the_markers(mode):
    mode.set_point_size(30)
    assert mode._overlay._active.material.size > 30


def test_hit_radius_grows_with_the_marker(mode):
    """A click visibly inside a big marker must grab it, not place a new point."""
    small = mode._hit_radius()
    mode.set_point_size(50)
    assert mode._hit_radius() > small


def test_hit_radius_never_drops_below_the_floor(mode):
    mode.set_point_size(4)
    # image_units_per_pixel() is 1.0 on the fake view, so this is in px.
    assert mode._hit_radius() == HIT_RADIUS_PX


def test_a_big_marker_grabs_a_click_further_away(mode):
    mode.store.set_point(0, "eye", (100.0, 100.0))
    mode.set_point_size(60)
    mode.refresh()
    mode.handle_click(125.0, 100.0)  # 25 px away: outside the default radius
    assert mode.active_keypoint == "eye"
