"""The labelling dialog's individual/keypoint tree and its key handling.

Driven through a stub data widget and a fake camera view, so no dataset, video
or GPU canvas is needed — only a QApplication and a headless pygfx scene.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygfx")

import pygfx as gfx  # noqa: E402
from qtpy.QtCore import QEvent, Qt  # noqa: E402
from qtpy.QtGui import QKeyEvent, QPixmap  # noqa: E402
from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

from ethograph.gui import dialog_pose_labelling as dialog_module  # noqa: E402
from ethograph.gui.app_state import ObservableAppState  # noqa: E402
from ethograph.gui.dialog_pose_labelling import (  # noqa: E402
    _FIXED_COLUMNS,
    CONFIDENCE_COLUMN,
    FRAME_COLUMN,
    INDIVIDUAL_COLUMN,
    SOURCE_COLUMN,
    PoseLabellingDialog,
)
from ethograph.gui.pose_edit_mixin import LOOP_MODE, SEQUENTIAL_MODE  # noqa: E402

NAMES = ["beak", "tail", "eye"]


class _FakeView:
    """The slice of CameraView the dialog and the label mode touch."""

    def __init__(self):
        self._scene = gfx.Scene()
        self._canvas = QWidget()
        self.fps = 25.0
        self.n_frames = 10
        self.start_frame = 0
        self.label_mode = None

    def scene(self):
        return self._scene

    def canvas_widget(self):
        return self._canvas

    def image_height(self):
        return 480.0

    def image_units_per_pixel(self):
        return 1.0

    def set_label_mode(self, mode):
        self.label_mode = mode

    def request_draw(self):
        pass


class _FakeShell(QWidget):
    """Stands in for the main window: it owns the video area, and the dialog
    installs a key filter on it (Shift+arrows are pressed while looking at the
    video, so they land here rather than on the dialog)."""

    def __init__(self):
        super().__init__()
        self.video_area = type("_Area", (), {"primary": _FakeView()})()


class _FakeDataWidget:
    """Everything ``PoseLabellingDialog`` reads off the data widget."""

    def __init__(self, app_state):
        self.app_state = app_state
        self.pose_mgr = None
        self.shell = _FakeShell()

    def update_pose(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")  # never touch the real settings
    state.keypoints = list(NAMES)
    state.labelling_keypoints = list(NAMES)
    dlg = PoseLabellingDialog(_FakeDataWidget(state))
    yield dlg
    dlg.close()


def _tree_names(tree) -> list[tuple[str, list[str]]]:
    """Branch/leaf names from their UserRole — the branch *text* carries a glyph."""
    return [
        (
            tree.topLevelItem(i).data(0, Qt.UserRole)[0],
            [tree.topLevelItem(i).child(k).text(0) for k in range(tree.topLevelItem(i).childCount())],
        )
        for i in range(tree.topLevelItemCount())
    ]


def _key(dlg, key: int, modifiers=Qt.NoModifier, target=None) -> None:
    """Send a key press, by default to the dialog itself.

    *target* stands in for the widget that really has focus — the tree, the
    table or the video canvas — since those are where the keys are pressed in
    practice and only an event filter can catch them there.
    """
    QApplication.sendEvent(target or dlg, QKeyEvent(QEvent.KeyPress, key, modifiers))


def _select_leaf(dialog, individual: str, keypoint: str) -> None:
    """Select a keypoint in the Keypoints tree, as clicking it would."""
    for i in range(dialog.tree.topLevelItemCount()):
        branch = dialog.tree.topLevelItem(i)
        for k in range(branch.childCount()):
            leaf = branch.child(k)
            if leaf.data(0, Qt.UserRole) == (individual, keypoint):
                dialog.tree.setCurrentItem(leaf)
                return
    raise AssertionError(f"no leaf for {(individual, keypoint)}")


def test_tree_shows_one_branch_per_individual(dialog):
    assert _tree_names(dialog.tree) == [("individual_0", NAMES)]


def test_adding_an_individual_adds_a_branch(dialog):
    dialog._apply_schema(individuals=["individual_0", "individual_1"])

    names = _tree_names(dialog.tree)
    assert [branch for branch, _ in names] == ["individual_0", "individual_1"]
    assert all(keypoints == NAMES for _, keypoints in names)
    assert dialog.app_state.labelling_individuals == ["individual_0", "individual_1"]


def test_keypoints_are_shared_by_every_individual(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog._apply_schema(keypoints=[*NAMES, "wing"])

    assert all(keypoints == [*NAMES, "wing"] for _, keypoints in _tree_names(dialog.tree))
    assert dialog.store.n_points == 2 * 4


def test_selecting_a_leaf_sets_the_active_pair(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    branch = dialog.tree.topLevelItem(1)
    dialog.tree.setCurrentItem(branch.child(2))

    assert dialog._mode.active_individual == "b"
    assert dialog._mode.active_keypoint == "eye"
    # The line is rich text (glyph + names in the marker colour), so match parts.
    assert "<b>b</b>" in dialog.active_label.text()
    assert "eye" in dialog.active_label.text()


def test_number_key_switches_individual_while_labelling(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    _key(dialog, Qt.Key_2)

    assert dialog._mode.active_individual == "b"
    assert dialog.tree.currentItem().data(0, Qt.UserRole)[0] == "b"


def test_backspace_deletes_the_selected_keypoint_without_a_mode(dialog):
    """Regression: the key filter only existed while a mode was armed, so
    selecting a keypoint and pressing Backspace did nothing at all."""
    dialog.store.set_point(0, "tail", (5.0, 6.0))
    _select_leaf(dialog, "individual_0", "tail")

    _key(dialog, Qt.Key_Backspace)

    assert dialog.store.is_anchor(0, "tail") is False


def test_backspace_deletes_the_selected_keypoint_from_the_tree(dialog):
    """The tree has focus while you are picking keypoints in it."""
    dialog.store.set_point(0, "eye", (5.0, 6.0))
    _select_leaf(dialog, "individual_0", "eye")

    _key(dialog, Qt.Key_Delete, target=dialog.tree)

    assert dialog.store.is_anchor(0, "eye") is False


def test_backspace_leaves_the_other_keypoints_alone(dialog):
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    dialog.store.set_point(0, "tail", (3.0, 4.0))
    _select_leaf(dialog, "individual_0", "beak")

    _key(dialog, Qt.Key_Backspace)

    assert dialog.store.is_anchor(0, "beak") is False
    assert dialog.store.is_anchor(0, "tail") is True


def test_backspace_deletes_the_active_point_while_labelling(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.store.set_point(0, "tail", (5.0, 6.0))
    dialog._mode.set_active("tail")

    _key(dialog, Qt.Key_Backspace)

    assert dialog.store.is_anchor(0, "tail") is False


def test_backspace_in_a_spin_box_edits_the_number(dialog):
    """Eating the spin box's Backspace would delete a keypoint per keystroke."""
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    _select_leaf(dialog, "individual_0", "beak")
    dialog.suggest_percent_spin.setFocus()

    _key(dialog, Qt.Key_Backspace, target=dialog.suggest_percent_spin)

    assert dialog.store.is_anchor(0, "beak") is True


def test_deleting_a_label_leaves_the_prediction_in_place(dialog):
    """Only the label goes; the fill keeps predicting that point."""
    _fill(dialog.store)
    dialog.store.promote_fill(0)
    _select_leaf(dialog, "individual_0", "beak")

    _key(dialog, Qt.Key_Backspace)

    assert dialog.store.is_anchor(0, "beak") is False
    assert not np.isnan(dialog.store.positions_for(0)[0, 0])  # still filled


def test_the_suggestion_combo_opens_on_a_method_that_can_run(dialog):
    """ "uncertain" needs a fill to rank by; with none, it could only warn."""
    assert dialog.suggest_method_combo.currentData() == "uniform"


def test_number_key_is_ignored_when_not_labelling(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    _key(dialog, Qt.Key_2)
    assert dialog._mode is None


def test_tab_cycles_keypoints_of_the_active_individual(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    _key(dialog, Qt.Key_Tab)
    assert dialog._mode.active_keypoint == "tail"


def test_tree_marks_the_points_labelled_on_this_frame(dialog):
    dialog.store.set_point(0, "tail", (5.0, 6.0))
    dialog._refresh_tree_marks()

    branch = dialog.tree.topLevelItem(0)
    assert branch.text(1) == f"1/{len(NAMES)}"
    assert branch.child(0).text(1) != branch.child(1).text(1)


def test_removing_the_last_individual_empties_the_tree(dialog):
    """Deleting every individual is allowed — the tree simply goes empty."""
    dialog._on_remove_individual()

    assert dialog.store.n_individuals == 0
    assert dialog.tree.topLevelItemCount() == 0
    assert dialog.app_state.labelling_individuals == []


def test_labelling_needs_an_individual(dialog):
    dialog._on_remove_individual()
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    assert dialog._mode is None
    assert dialog.sequential_btn.isChecked() is False


def test_removing_the_last_individual_stops_labelling(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog._on_remove_individual()

    assert dialog._mode is None
    assert dialog.sequential_btn.isChecked() is False


def test_adding_an_individual_back_restores_labelling(dialog):
    dialog._on_remove_individual()
    dialog._apply_schema(individuals=["crow"])
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    assert dialog._mode is not None
    assert dialog._mode.active_individual == "crow"


# ----------------------------------------------------------------------
# Asymmetric schemas: individuals with their own keypoints
# ----------------------------------------------------------------------


def test_unsharing_keeps_every_individual_on_the_full_schema(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)

    assert dialog.store.shared_keypoints is False
    assert _tree_names(dialog.tree) == [("a", NAMES), ("b", NAMES)]


def test_adding_a_keypoint_affects_only_the_selected_individual(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)
    dialog._apply_schema(individual_keypoints=("b", [*NAMES, "wing"]))

    assert _tree_names(dialog.tree) == [("a", NAMES), ("b", [*NAMES, "wing"])]
    assert dialog.store.keypoint_names == [*NAMES, "wing"]
    assert dialog.store.n_schema_points == len(NAMES) + len(NAMES) + 1


def test_removing_a_keypoint_from_one_individual_keeps_it_on_the_other(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)
    dialog._apply_schema(individual_keypoints=("a", ["beak"]))

    assert _tree_names(dialog.tree) == [("a", ["beak"]), ("b", NAMES)]


def test_tree_marks_count_the_individuals_own_keypoints(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)
    dialog._apply_schema(individual_keypoints=("a", ["beak", "eye"]))
    dialog.store.set_point(0, "eye", (5.0, 6.0), "a")
    dialog._refresh_tree_marks()

    branch = dialog.tree.topLevelItem(0)
    assert branch.text(1) == "1/2"
    assert branch.child(1).text(1) != branch.child(0).text(1)


def test_resharing_gives_everyone_the_union(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)
    dialog._apply_schema(individual_keypoints=("b", [*NAMES, "wing"]))
    dialog.shared_toggle.setChecked(True)

    assert _tree_names(dialog.tree) == [("a", [*NAMES, "wing"]), ("b", [*NAMES, "wing"])]


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------


def test_the_dialog_is_split_into_stages(dialog):
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "Define keypoints",
        "Label && Edit",  # && escapes the mnemonic
        "Fill and export",
    ]


def test_arming_a_mode_shows_the_label_tab(dialog):
    dialog.tabs.setCurrentIndex(0)
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    assert dialog.tabs.currentWidget() is dialog._label_page


def test_disarming_leaves_the_current_tab_alone(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.tabs.setCurrentIndex(2)
    dialog.set_interaction_mode(None)

    assert dialog.tabs.currentIndex() == 2


# ----------------------------------------------------------------------
# Sequential vs loop mode
# ----------------------------------------------------------------------


def test_the_buttons_arm_and_disarm_their_mode(dialog):
    dialog.sequential_btn.click()
    assert dialog.interaction_mode == SEQUENTIAL_MODE
    assert dialog.sequential_btn.isChecked() is True

    dialog.sequential_btn.click()
    assert dialog.interaction_mode is None
    assert dialog.sequential_btn.isChecked() is False


def test_switching_mode_keeps_the_same_canvas_mode_object(dialog):
    dialog.sequential_btn.click()
    mode = dialog._mode
    dialog.loop_btn.click()

    assert dialog._mode is mode  # the canvas overlay is not rebuilt
    assert dialog.interaction_mode == LOOP_MODE
    assert dialog.loop_btn.isChecked() is True
    assert dialog.sequential_btn.isChecked() is False


def test_loop_mode_can_be_armed_from_cold(dialog):
    dialog.loop_btn.click()
    assert dialog.interaction_mode == LOOP_MODE


def test_the_active_line_names_what_the_next_click_places(dialog):
    dialog.set_interaction_mode(LOOP_MODE)
    text = dialog.active_label.text()

    assert "individual_0" in text
    assert "beak" in text
    assert "Loop" in text


def test_mode_survives_a_schema_change(dialog):
    dialog.set_interaction_mode(LOOP_MODE)
    dialog._apply_schema(keypoints=[*NAMES, "wing"])

    assert dialog.interaction_mode == LOOP_MODE


# ----------------------------------------------------------------------
# Labelled-frames table
# ----------------------------------------------------------------------


def _table_headers(dialog) -> list[str]:
    """Fixed labels plus the keypoint groups the header paints over each pair."""
    model = dialog.point_model
    fixed = [model.headerData(col, Qt.Horizontal) for col in range(len(_FIXED_COLUMNS))]
    return fixed + dialog.point_table.horizontalHeader().groups()


def _header_tooltips(dialog) -> list[str]:
    model = dialog.point_model
    return [
        model.headerData(col, Qt.Horizontal, Qt.ToolTipRole) for col in range(len(_FIXED_COLUMNS), model.columnCount())
    ]


def _table_rows(dialog) -> list[tuple[str, ...]]:
    """Every visible row, as text — through the proxy, so filters apply."""
    proxy = dialog.point_proxy
    return [
        tuple(proxy.index(row, col).data() or "" for col in range(proxy.columnCount()))
        for row in range(proxy.rowCount())
    ]


def _fill(store, value: float = 5.0) -> None:
    """A backend result covering every frame, so the store has predictions."""
    shape = (store.n_frames, store.n_individuals, store.n_keypoints)
    store.set_fill(np.full((*shape, 2), value), np.full(shape, 0.5))


def test_a_row_holds_every_keypoint_labelled_on_that_frame(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog.store.set_point(3, "tail", (30.0, 40.0))
    dialog.store.set_point(7, "beak", (50.0, 60.0))
    dialog._refresh_point_table()

    assert _table_headers(dialog) == ["Frame", "Individual", "Source", "Confidence", "beak", "tail"]
    assert _table_rows(dialog) == [
        ("3", "individual_0", "Human", "", "10.0", "20.0", "30.0", "40.0"),
        ("7", "individual_0", "Human", "", "50.0", "60.0", "", ""),
    ]


def test_each_keypoint_spans_an_x_and_a_y_column(dialog):
    """One header name over two columns — repeating it in both was the waste."""
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog.store.set_point(3, "tail", (30.0, 40.0))
    dialog._refresh_point_table()

    assert dialog.point_model.columnCount() == len(_FIXED_COLUMNS) + 2 * 2
    assert _header_tooltips(dialog) == ["beak x", "beak y", "tail x", "tail y"]


def test_the_two_row_header_renders(dialog):
    """Custom paintSection code only breaks when something actually paints."""
    dialog.store.set_point(3, "beak", (1234.5, 20.0))
    dialog.store.set_point(3, "tail", (30.0, 40.0))
    dialog._refresh_point_table()

    table = dialog.point_table
    table.resize(600, 200)
    pixmap = QPixmap(table.size())
    table.render(pixmap)

    assert not pixmap.isNull()
    assert table.horizontalHeader().height() > table.rowHeight(0)  # two rows of labels


def test_unlabelled_keypoints_get_no_columns(dialog):
    """A 20-keypoint schema must not bury the two columns being worked on."""
    dialog.store.set_point(3, "eye", (1.0, 2.0))
    dialog._refresh_point_table()

    assert _table_headers(dialog) == ["Frame", "Individual", "Source", "Confidence", "eye"]


def test_each_individual_gets_its_own_row(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.store.set_point(4, "beak", (1.0, 2.0), "a")
    dialog.store.set_point(4, "beak", (3.0, 4.0), "b")
    dialog._refresh_point_table()

    assert _table_rows(dialog) == [
        ("4", "a", "Human", "", "1.0", "2.0"),
        ("4", "b", "Human", "", "3.0", "4.0"),
    ]


def test_table_drops_deleted_points(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog._refresh_point_table()
    dialog.store.clear_point(3, "beak")
    dialog._refresh_point_table()

    assert _table_rows(dialog) == []


def test_deleting_one_keypoint_blanks_only_its_cells(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog.store.set_point(3, "tail", (30.0, 40.0))
    dialog.store.set_point(7, "tail", (50.0, 60.0))
    dialog._refresh_point_table()
    dialog.store.clear_point(3, "tail")
    dialog._refresh_point_table()

    assert _table_rows(dialog)[0] == ("3", "individual_0", "Human", "", "10.0", "20.0", "", "")


def test_moving_a_point_updates_its_cells_in_place(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog._refresh_point_table()
    dialog.store.set_point(3, "beak", (99.0, 98.0))
    dialog._refresh_point_table()

    assert _table_rows(dialog) == [("3", "individual_0", "Human", "", "99.0", "98.0")]


def _click_cell(dialog, row: int, column: int) -> None:
    dialog._on_table_clicked(dialog.point_proxy.index(row, column))


def test_clicking_a_keypoint_cell_makes_that_point_active(dialog):
    dialog.store.set_point(3, "beak", (1.0, 2.0))
    dialog.store.set_point(3, "eye", (10.0, 20.0))
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog._refresh_point_table()

    _click_cell(dialog, 0, len(_FIXED_COLUMNS) + 2)  # the 'eye x' column

    assert dialog._mode.active_keypoint == "eye"


def test_clicking_the_frame_cell_keeps_the_active_keypoint(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.store.set_point(3, "tail", (1.0, 2.0), "b")
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog._refresh_point_table()

    _click_cell(dialog, 0, 0)

    assert dialog._mode.active_individual == "b"


def test_an_undo_repaints_the_frame_it_landed_on(dialog):
    """Undo can revert a frame you are not looking at — that row must repaint."""
    _fill(dialog.store)  # dense rows, so the layout cannot change and mask this
    dialog.store.set_point(2, "beak", (10.0, 20.0))
    dialog.app_state.current_frame = 8
    dialog._refresh_point_table(full=True)

    repainted = []
    dialog.point_model.dataChanged.connect(lambda first, last, *_: repainted.append((first.row(), last.row())))
    dialog._on_undo()

    assert (2, 2) in repainted  # the undone frame, not the one on screen
    assert _table_rows(dialog)[2][SOURCE_COLUMN] == "Fill"


# ----------------------------------------------------------------------
# Which frames to label, and navigating them
# ----------------------------------------------------------------------


def _seeks(dialog, monkeypatch) -> list[int]:
    """Record where the dialog seeks — the fake view has no video to move."""
    landed: list[int] = []
    monkeypatch.setattr(dialog, "_seek", lambda frame: landed.append(int(frame)))
    return landed


def test_the_methods_are_ordered_by_when_they_apply(dialog):
    """The three that need nothing but the video come first."""
    combo = dialog.suggest_method_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "uniform",
        "motion",
        "diverse",
        "uncertain",
    ]


def test_the_share_resolves_to_a_frame_count(dialog):
    dialog.suggest_percent_spin.setValue(20.0)

    assert dialog._suggest_count() == 2  # of the fake view's 10 frames
    assert dialog.suggest_count_label.text() == "2 of 10 frames"


def test_the_share_never_resolves_to_zero_frames(dialog):
    dialog.suggest_percent_spin.setValue(dialog.suggest_percent_spin.minimum())
    assert dialog._suggest_count() == 1


def test_the_default_share_aims_at_the_recommended_count(dialog):
    """20 anchors of a 10-frame clip is the whole clip, so it clamps."""
    assert dialog._default_suggest_percent() == 100.0


def test_shift_arrows_step_the_suggested_frames(dialog, monkeypatch):
    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_Right, Qt.ShiftModifier)
    _key(dialog, Qt.Key_Left, Qt.ShiftModifier)

    assert landed == [5, 1]


def test_shift_arrows_reach_the_dialog_from_the_main_window(dialog, monkeypatch):
    """Regression: pressed while looking at the video, they hit the main
    window's own window-stepping shortcut instead of the suggestions."""
    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_Right, Qt.ShiftModifier, target=dialog._shell)

    assert landed == [5]


def test_the_main_window_keeps_its_other_keys(dialog):
    """Only the arrows reach across; the user is working over there."""
    dialog._suggestions = [1, 5, 9]
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    _select_leaf(dialog, "individual_0", "beak")

    _key(dialog, Qt.Key_Backspace, target=dialog._shell)

    assert dialog.store.is_anchor(0, "beak") is True


def test_plain_arrows_are_left_to_the_main_window(dialog, monkeypatch):
    """Single-frame stepping stays a global shortcut, suggestions are Shift."""
    dialog._suggestions = [1, 5, 9]
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_Right)

    assert landed == []


def test_shift_arrows_are_not_claimed_without_suggestions(dialog, monkeypatch):
    """Nothing to step through, so the main window keeps its window-stepping."""
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_Right, Qt.ShiftModifier)

    assert landed == []


def test_loop_stays_on_the_frame_when_asked(dialog, monkeypatch):
    dialog.set_interaction_mode(LOOP_MODE)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("stay"))
    landed = _seeks(dialog, monkeypatch)

    dialog._advance_frame()

    assert landed == []


def test_loop_steps_one_frame_when_asked(dialog, monkeypatch):
    dialog.set_interaction_mode(LOOP_MODE)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("frame"))
    dialog.app_state.current_frame = 3
    landed = _seeks(dialog, monkeypatch)

    dialog._advance_frame()

    assert landed == [4]


def test_loop_follows_the_suggestions_when_asked(dialog, monkeypatch):
    dialog.set_interaction_mode(LOOP_MODE)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("suggestion"))
    dialog._suggestions = [2, 6]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    dialog._advance_frame()

    assert landed == [6]


def test_the_between_clicks_row_belongs_to_loop_mode(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    assert dialog.after_click_row.isVisibleTo(dialog) is False

    dialog.set_interaction_mode(LOOP_MODE)
    assert dialog.after_click_row.isVisibleTo(dialog) is True


# ----------------------------------------------------------------------
# Human labels vs filled predictions
# ----------------------------------------------------------------------


def test_filling_gives_every_frame_a_row(dialog):
    """A prediction you cannot see is a prediction you cannot correct."""
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog._refresh_point_table()

    assert dialog.point_model.rowCount() == dialog.store.n_frames
    assert _table_headers(dialog) == ["Frame", "Individual", "Source", "Confidence", *NAMES]


def test_filled_rows_are_marked_fill_and_labelled_ones_human(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)  # confidence 0.5 everywhere, 1.0 on the labelled point
    dialog._refresh_point_table()

    rows = _table_rows(dialog)
    assert rows[3] == ("3", "individual_0", "Human", "0.67", "10.0", "20.0", "5.0", "5.0", "5.0", "5.0")
    assert rows[4] == ("4", "individual_0", "Fill", "0.50", "5.0", "5.0", "5.0", "5.0", "5.0", "5.0")


def test_one_corrected_keypoint_makes_the_whole_row_human(dialog):
    """The rule: touch one point of a (frame, individual) and the row is yours."""
    _fill(dialog.store)
    dialog.store.set_point(6, "tail", (1.0, 2.0))
    dialog._refresh_point_table(full=True)

    assert _table_rows(dialog)[6][SOURCE_COLUMN] == "Human"


def test_predicted_coordinates_are_dimmed(dialog):
    """Per-point provenance, so a mixed row still reads correctly."""
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog._refresh_point_table()

    beak_x = len(_FIXED_COLUMNS)
    tail_x = beak_x + 2
    assert dialog.point_proxy.index(3, beak_x).data(Qt.ForegroundRole) is None  # labelled
    assert dialog.point_proxy.index(3, tail_x).data(Qt.ForegroundRole) is not None  # predicted


def test_a_filled_row_can_be_deleted_from_the_table(dialog):
    """Regression: "Delete labels" is a no-op on a row that is all prediction."""
    _fill(dialog.store)
    dialog._refresh_point_table()

    dialog._clear_fill_rows([(4, "individual_0")])

    assert dialog.store.has_fill(4) is False
    row = _table_rows(dialog)[4]
    assert row[SOURCE_COLUMN] == ""  # no source left to name
    assert row[len(_FIXED_COLUMNS) :] == ("",) * (2 * len(NAMES))


def test_deleting_a_filled_row_keeps_the_labels_on_it(dialog):
    _fill(dialog.store)
    dialog.store.set_point(4, "beak", (10.0, 20.0))
    dialog._refresh_point_table(full=True)

    dialog._clear_fill_rows([(4, "individual_0")])

    assert _table_rows(dialog)[4][: len(_FIXED_COLUMNS) + 1] == ("4", "individual_0", "Human", "1.00", "10.0")


def test_the_disagreement_tolerance_belongs_to_the_tracking_backends(dialog):
    """The spline scores by distance from an anchor — the tolerance is meaningless."""
    combo = dialog.backend_combo
    combo.setCurrentIndex(combo.findData("spline"))
    assert dialog.disagreement_row.isHidden() is True

    combo.setCurrentIndex(combo.findData("flow"))
    assert dialog.disagreement_row.isHidden() is False


def test_the_disagreement_tolerance_is_remembered(dialog):
    dialog.disagreement_spin.setValue(25.0)
    assert dialog.app_state.labelling_disagreement_px == 25.0


def test_the_confidence_column_is_empty_before_a_fill(dialog):
    """There is nothing to be confident about until a backend has run."""
    dialog.store.set_point(3, "beak", (1.0, 2.0))
    dialog._refresh_point_table()

    assert _table_rows(dialog)[0][CONFIDENCE_COLUMN] == ""


def test_the_confidence_column_averages_the_rows_keypoints(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)  # 0.5 everywhere, 1.0 on the labelled point
    dialog._refresh_point_table()

    assert _table_rows(dialog)[3][CONFIDENCE_COLUMN] == "0.67"
    assert _table_rows(dialog)[4][CONFIDENCE_COLUMN] == "0.50"


def test_the_confidence_cell_names_the_worst_keypoint(dialog):
    _fill(dialog.store)
    dialog.store.confidence[6, 0, 2] = 0.1  # 'eye'
    dialog._refresh_point_table(full=True)

    tooltip = dialog.point_proxy.index(6, CONFIDENCE_COLUMN).data(Qt.ToolTipRole)
    assert tooltip == "Lowest: eye 0.10"


def test_the_confidence_header_explains_the_number(dialog):
    """The column is meaningless without saying what produced it."""
    tooltip = dialog.point_model.headerData(CONFIDENCE_COLUMN, Qt.Horizontal, Qt.ToolTipRole)

    assert "forwards" in tooltip and "backwards" in tooltip
    assert "Spline" in tooltip


def test_the_confidence_column_filters_numerically(dialog):
    """The point of showing it: find the rows worth correcting."""
    _fill(dialog.store)
    dialog.store.confidence[2] = 0.1
    dialog._refresh_point_table(full=True)

    dialog.point_proxy.set_numeric_filter(CONFIDENCE_COLUMN, "<=", 0.2)

    assert _frames_shown(dialog) == ["2"]


def test_pinning_a_row_turns_its_predictions_into_labels(dialog):
    """ "Accepting" a fill: the next fill must treat these as ground truth."""
    _fill(dialog.store)
    dialog._refresh_point_table()

    dialog._pin_table_rows([(4, "individual_0")])

    assert dialog.store.is_human(4) is True
    assert 4 in dialog.store.flat_anchors()
    assert _table_rows(dialog)[4][SOURCE_COLUMN] == "Human"


# ----------------------------------------------------------------------
# Column filters
# ----------------------------------------------------------------------


def _frames_shown(dialog) -> list[str]:
    return [row[0] for row in _table_rows(dialog)]


def test_source_filter_keeps_only_the_labelled_rows(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog._refresh_point_table()

    dialog.point_proxy.set_cat_filter(SOURCE_COLUMN, {"Human"})

    assert _frames_shown(dialog) == ["3"]


def test_individual_filter_keeps_one_animal(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.store.set_point(4, "beak", (1.0, 2.0), "a")
    dialog.store.set_point(5, "beak", (3.0, 4.0), "b")
    dialog._refresh_point_table()

    dialog.point_proxy.set_cat_filter(INDIVIDUAL_COLUMN, {"b"})

    assert _table_rows(dialog) == [("5", "b", "Human", "", "3.0", "4.0")]


def test_the_frame_column_filters_numerically(dialog):
    for frame in (1, 5, 9):
        dialog.store.set_point(frame, "beak", (1.0, 2.0))
    dialog._refresh_point_table()

    dialog.point_proxy.set_numeric_filter(FRAME_COLUMN, ">=", 5)

    assert _frames_shown(dialog) == ["5", "9"]


def test_clicking_a_funnel_applies_the_chosen_categories(dialog, monkeypatch):
    """The whole header path: funnel → popup → proxy → the funnel marked active."""
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog._refresh_point_table()

    class _PickedHuman:
        def __init__(self, *args):
            pass

        def exec_(self):
            return True

        def get_allowed(self):
            return {"Human"}

    monkeypatch.setattr(dialog_module, "CategoryFilterDialog", _PickedHuman)
    dialog._on_filter_requested(SOURCE_COLUMN)

    assert _frames_shown(dialog) == ["3"]
    assert dialog.point_proxy.active_filters() == {SOURCE_COLUMN}


def test_the_filterable_columns_are_the_fixed_ones(dialog):
    """Filtering an x/y pair is meaningless — those are coordinates."""
    assert dialog.point_table.horizontalHeader().filterable == {
        FRAME_COLUMN,
        INDIVIDUAL_COLUMN,
        SOURCE_COLUMN,
        CONFIDENCE_COLUMN,
    }


def test_a_schema_change_drops_the_filters(dialog):
    """A filter naming an individual that no longer exists empties the table."""
    dialog._apply_schema(individuals=["a", "b"])
    dialog.store.set_point(4, "beak", (1.0, 2.0), "a")
    dialog._refresh_point_table()
    dialog.point_proxy.set_cat_filter(INDIVIDUAL_COLUMN, {"b"})

    dialog._apply_schema(individuals=["a"])

    assert dialog.point_proxy.active_filters() == set()
    assert _frames_shown(dialog) == ["4"]
