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
    INDIVIDUAL_COLUMN,
    SOURCE_COLUMN,
    PoseLabellingDialog,
)
from ethograph.gui.pose_annotate import RECOMMENDED_LABEL_SHARE  # noqa: E402
from ethograph.gui.pose_edit_mixin import LOOP_MODE, SEQUENTIAL_MODE  # noqa: E402
from ethograph.gui.pose_fill import SplineBackend  # noqa: E402

NAMES = ["beak", "tail", "eye"]


class _FakeRenderWidget(QWidget):
    """The inner widget rendercanvas actually gives focus to.

    Its ``keyPressEvent`` neither ignores the event nor calls the base class,
    exactly like ``rendercanvas.qt.QRenderWidget`` — so a key pressed over the
    video stops here and never propagates to the wrapper or the main window.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event):
        event.accept()


class _FakeView:
    """The slice of CameraView the dialog and the label mode touch."""

    def __init__(self):
        self._scene = gfx.Scene()
        # A wrapper holding the focusable render widget, as rendercanvas nests
        # them: filtering the wrapper alone sees no keys at all.
        self._canvas = QWidget()
        self._render_widget = _FakeRenderWidget(self._canvas)
        self.fps = 25.0
        self.n_frames = 10
        self.start_frame = 0
        self.label_mode = None
        self.pan_with_left_drag = True

    def scene(self):
        return self._scene

    def canvas_widget(self):
        return self._canvas

    def key_target(self):
        focusable = [w for w in self._canvas.findChildren(QWidget) if w.focusPolicy() != Qt.NoFocus]
        return focusable[0] if focusable else self._canvas

    def image_height(self):
        return 480.0

    def image_units_per_pixel(self):
        return 1.0

    def set_label_mode(self, mode):
        self.label_mode = mode
        self.pan_with_left_drag = mode is None or mode.locked

    def set_label_locked(self, locked):
        self.pan_with_left_drag = locked

    def request_draw(self):
        pass


class _FakeShell(QWidget):
    """Stands in for the main window: it owns the video area, and the dialog
    installs a key filter on it (Shift+arrows are pressed while looking at the
    video, so they land here rather than on the dialog)."""

    def __init__(self):
        super().__init__()
        self.video_area = type("_Area", (), {"primary": _FakeView()})()


class _FakePoseManager:
    """Records what the dialog hands to the ordinary pose overlay."""

    def __init__(self):
        self.override = None

    def set_pose_override(self, render):
        self.override = render


class _FakeDataWidget:
    """Everything ``PoseLabellingDialog`` reads off the data widget."""

    def __init__(self, app_state):
        self.app_state = app_state
        self.pose_mgr = _FakePoseManager()
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


def test_tab_cycles_when_the_tree_has_focus(dialog):
    """Regression: Tab silently did nothing whenever a child widget had focus.

    Sending the event to the dialog (as the test above does) cannot catch this —
    Qt turns Tab into focus navigation inside the *focused* widget's ``event()``,
    so it never propagates up to the dialog's filter. Only the dialog's own
    QShortcut runs early enough, and the filter has to decline the
    ShortcutOverride for Tab or it suppresses that shortcut itself.
    """
    from qtpy.QtTest import QTest

    dialog.show()
    dialog.activateWindow()
    QApplication.processEvents()
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.tree.setFocus()
    QApplication.processEvents()
    assert isinstance(QApplication.focusWidget(), type(dialog.tree))

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
    QApplication.processEvents()

    assert dialog._mode.active_keypoint == "tail"


def test_backspace_reaches_across_from_the_main_window(dialog):
    """You press it right after clicking the video, and where the click left
    focus is not something the user should have to know."""
    dialog.store.set_point(0, "tail", (5.0, 6.0))
    _select_leaf(dialog, "individual_0", "tail")

    _key(dialog, Qt.Key_Backspace, target=dialog._shell)

    assert dialog.store.is_anchor(0, "tail") is False


def test_backspace_works_when_the_render_widget_has_focus(dialog):
    """Regression: the filter sat on the canvas *wrapper*, which never sees a key.

    Clicking the video to place a point focuses rendercanvas's inner render
    widget, whose ``keyPressEvent`` swallows the event — so Backspace did
    nothing at exactly the moment it is wanted. The filter has to go on
    ``CameraView.key_target()``, the widget the press actually lands on.
    """
    dialog.store.set_point(0, "eye", (5.0, 6.0))
    _select_leaf(dialog, "individual_0", "eye")

    _key(dialog, Qt.Key_Backspace, target=dialog._view.key_target())

    assert dialog._view.key_target() is not dialog._view.canvas_widget()
    assert dialog.store.is_anchor(0, "eye") is False


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


def test_the_lock_suspends_labelling_without_dropping_the_mode(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    mode = dialog._mode
    dialog.lock_check.setChecked(True)

    assert dialog._mode is mode  # still armed, still drawing its anchors
    assert dialog.interaction_mode == SEQUENTIAL_MODE
    assert mode.locked is True
    mode.handle_click(100.0, 100.0)
    assert dialog.store.anchor_frames() == []

    dialog.lock_check.setChecked(False)
    assert mode.locked is False
    mode.handle_click(100.0, 100.0)
    assert dialog.store.anchor_frames() == [0]


def test_the_lock_is_kept_when_the_mode_is_rearmed(dialog):
    """A schema change restarts the canvas mode; the lock must not fall off."""
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.lock_check.setChecked(True)
    dialog._apply_schema(keypoints=[*NAMES, "wing"])

    assert dialog._mode.locked is True


def test_the_lock_says_so_in_place_of_the_target(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.lock_check.setChecked(True)

    assert "Locked" in dialog.active_label.text()
    assert "beak" not in dialog.active_label.text()


def test_the_lock_is_only_offered_while_the_canvas_is_armed(dialog):
    dialog.set_interaction_mode(None)
    assert dialog.lock_check.isEnabled() is False
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    assert dialog.lock_check.isEnabled() is True


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


def test_the_default_share_is_a_spacing_not_a_count(dialog):
    """Roughly every 10th frame, whatever the clip length — the backends bridge
    gaps, and a gap is measured in frames."""
    assert dialog._default_suggest_percent() == RECOMMENDED_LABEL_SHARE
    assert dialog.suggest_percent_spin.value() == RECOMMENDED_LABEL_SHARE
    assert dialog._suggest_count() == 1  # of the fake view's 10 frames


def test_n_steps_to_the_next_suggested_frame(dialog, monkeypatch):
    """One direction only — the suggestions are a queue, and the points table
    is how you go back to any frame at all."""
    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_N)
    _key(dialog, Qt.Key_N)

    assert landed == [5, 9]


def test_n_wraps_at_the_end_of_the_suggestions(dialog, monkeypatch):
    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 2
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_N)

    assert landed == [1]


def test_n_reaches_the_dialog_from_the_main_window(dialog, monkeypatch):
    """It is pressed while looking at the video, not at the dialog."""
    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_N, target=dialog._shell)

    assert landed == [5]


def test_n_steps_once_per_press_with_the_table_focused(dialog, qapp, monkeypatch):
    """``N`` is printable, so ``QAbstractItemView`` eats it as a type-ahead
    search — only the dialog's own QShortcut runs early enough. And exactly one
    of the shortcut and the event filter may act, or a press skips a frame."""
    from qtpy.QtTest import QTest

    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 0
    dialog.show()
    dialog.activateWindow()
    dialog.point_table.setFocus()
    QApplication.processEvents()
    landed = _seeks(dialog, monkeypatch)

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_N)
    QApplication.processEvents()

    assert landed == [5]


def test_the_main_window_keeps_its_number_keys(dialog):
    """`1`-`9` are behaviour labels over there, so they must not reach across —
    unlike Backspace and Ctrl+Z, which nothing in the main window binds."""
    dialog._suggestions = [1, 5, 9]
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog._apply_schema(individuals=["a", "b"])

    _key(dialog, Qt.Key_2, target=dialog._shell)

    assert dialog._mode.active_individual == "a"


def test_the_arrows_are_left_to_the_main_window(dialog, monkeypatch):
    """Both plain and Shift: frame stepping and window stepping stay global
    shortcuts, and the suggestions moved off the arrows entirely."""
    dialog._suggestions = [1, 5, 9]
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_Right)
    _key(dialog, Qt.Key_Right, Qt.ShiftModifier)
    _key(dialog, Qt.Key_Left, Qt.ShiftModifier)

    assert landed == []


def test_n_warns_instead_of_seeking_without_suggestions(dialog, monkeypatch):
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_N)

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


def test_the_then_go_to_row_belongs_to_loop_mode(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    assert dialog.after_click_row.isVisibleTo(dialog) is False

    dialog.set_interaction_mode(LOOP_MODE)
    assert dialog.after_click_row.isVisibleTo(dialog) is True


def test_the_then_go_to_row_also_appears_for_approving(dialog):
    """It says where Shift+H lands too, and approving needs no mode.

    ``isHidden`` rather than ``isVisibleTo``: with no mode armed the dialog is
    still on the Keypoints tab, so the whole Label page is hidden — arming one
    to look would defeat the point of the test.
    """
    dialog.set_interaction_mode(None)
    assert dialog.after_click_row.isHidden() is True

    _fill(dialog.store)
    dialog._refresh_active_label()

    assert dialog.after_click_row.isHidden() is False


# ----------------------------------------------------------------------
# Approving a fill (Shift+H)
# ----------------------------------------------------------------------


def test_shift_h_keeps_the_frames_predictions_as_labels(dialog):
    _fill(dialog.store, value=7.0)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert all(dialog.store.is_anchor(4, name) for name in NAMES)
    assert dialog.store.anchor_positions_for(4).tolist() == [[7.0, 7.0]] * len(NAMES)


def test_shift_h_leaves_the_other_frames_predicted(dialog):
    """Approving is per frame — that is what makes it a review step."""
    _fill(dialog.store)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert dialog.store.is_human(4) is True
    assert dialog.store.is_human(5) is False


def test_shift_h_never_overwrites_a_label_you_placed(dialog):
    dialog.store.set_point(4, "beak", (1.0, 2.0))
    _fill(dialog.store, value=7.0)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert dialog.store.anchor_positions_for(4)[0].tolist() == [1.0, 2.0]
    assert dialog.store.anchor_positions_for(4)[1].tolist() == [7.0, 7.0]


def test_shift_h_then_goes_where_the_dropdown_says(dialog, monkeypatch):
    _fill(dialog.store)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("frame"))
    dialog.app_state.current_frame = 4
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert landed == [5]


def test_shift_h_can_follow_the_suggestions_instead(dialog, monkeypatch):
    _fill(dialog.store)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("suggestion"))
    dialog._suggestions = [2, 6]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert landed == [6]


def test_shift_h_moves_on_from_an_already_approved_frame(dialog, monkeypatch):
    """Agreeing with a frame that is already yours still means "next"."""
    _fill(dialog.store)
    dialog.app_state.current_frame = 4
    dialog.store.promote_fill(4)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("frame"))
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert landed == [5]


def test_shift_h_stays_put_when_the_frame_carries_nothing(dialog, monkeypatch):
    """Silently advancing past empty frames would look like a broken key."""
    dialog.app_state.current_frame = 4
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert landed == []


def test_plain_h_is_left_to_the_main_windows_behaviour_label(dialog):
    _fill(dialog.store)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H)

    assert dialog.store.is_human(4) is False


def test_shift_h_works_when_the_points_table_has_focus(dialog, qapp):
    """Regression: the table ate it as type-ahead search.

    You pick the frame to review by clicking its row, which leaves focus on the
    table — and ``QAbstractItemView`` turns any printable key into a keyboard
    search and accepts it, so the press never propagated to the dialog's filter.
    Only the dialog's own QShortcut runs early enough.
    """
    from qtpy.QtTest import QTest

    _fill(dialog.store)
    dialog.show()
    dialog.activateWindow()
    dialog.app_state.current_frame = 4
    dialog.point_table.setFocus()
    QApplication.processEvents()

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_H, Qt.ShiftModifier)
    QApplication.processEvents()

    assert dialog.store.is_human(4) is True


def test_shift_h_works_when_the_tree_has_focus(dialog, qapp):
    """Same swallowing, from the other item view."""
    from qtpy.QtTest import QTest

    _fill(dialog.store)
    dialog.show()
    dialog.activateWindow()
    dialog.app_state.current_frame = 4
    dialog.tree.setFocus()
    QApplication.processEvents()

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_H, Qt.ShiftModifier)
    QApplication.processEvents()

    assert dialog.store.is_human(4) is True


def test_shift_h_approves_once_per_press(dialog, qapp, monkeypatch):
    """The QShortcut and the event filter both know the key — only one may act.

    Firing twice would silently skip the frame after the one approved.
    """
    from qtpy.QtTest import QTest

    _fill(dialog.store)
    dialog.show()
    dialog.activateWindow()
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("frame"))
    dialog.app_state.current_frame = 4
    dialog.tree.setFocus()
    QApplication.processEvents()
    landed = _seeks(dialog, monkeypatch)

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_H, Qt.ShiftModifier)
    QApplication.processEvents()

    assert landed == [5]


def test_shift_h_reaches_across_from_the_main_window(dialog):
    """You press it while looking at the video, not at the dialog."""
    _fill(dialog.store)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H, Qt.ShiftModifier, target=dialog._shell)

    assert dialog.store.is_human(4) is True


def test_the_approve_button_waits_for_a_fill(dialog):
    """With no predictions on screen there is nothing to approve."""
    assert dialog.approve_btn.isHidden() is True

    _fill(dialog.store)
    dialog._refresh_active_label()

    assert dialog.approve_btn.isHidden() is False


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


def test_the_table_stops_where_the_fill_does(dialog):
    """A fill bridges the labels and no further, and rows follow it.

    Rows past the last label would be permanently empty ones — nothing can ever
    populate them short of labelling further out, which changes the span anyway.
    """
    for frame in (2, 5):
        dialog.store.set_point(frame, "beak", (float(frame), 1.0))
    anchors, n_frames = dialog.store.flat_anchors(), dialog.store.n_frames
    dialog.store.set_fill_from_flat(*SplineBackend().fill(anchors, n_frames, None))
    dialog._refresh_point_table(full=True)

    assert {frame for frame, _individual in dialog.point_model.rows} == {2, 3, 4, 5}


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


def test_the_canvas_shows_predictions_alongside_labels(dialog):
    """Judging a prediction means seeing it — hollow, next to the solid labels."""
    dialog.store.set_point(0, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    overlay = dialog._mode._overlay

    drawn = overlay._layers[0].geometry.positions.data[:, 0] > -1e5
    predicted = overlay._fill_layers[0].geometry.positions.data[:, 0] > -1e5
    assert list(drawn) == [True, False, False]
    assert list(predicted) == [False, True, True]


def test_the_pose_overlay_yields_the_canvas_while_a_mode_is_armed(dialog):
    """Otherwise every point carries two markers, one of them provenance-blind."""
    _fill(dialog.store)
    dialog._push_pose_override()
    assert dialog._data_widget.pose_mgr.override is not None

    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    assert dialog._data_widget.pose_mgr.override is None

    dialog.set_interaction_mode(None)
    assert dialog._data_widget.pose_mgr.override is not None


def test_the_legend_appears_only_once_predictions_are_on_screen(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    assert dialog.legend_label.isVisibleTo(dialog) is False

    _fill(dialog.store)
    dialog._refresh_active_label()
    assert dialog.legend_label.isVisibleTo(dialog) is True


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


def test_custom_weights_belong_to_posepal(dialog):
    """Only PosePAL loads a state dict — the others have nothing to point at."""
    combo = dialog.backend_combo
    combo.setCurrentIndex(combo.findData("flow"))
    assert dialog.checkpoint_row.isHidden() is True

    combo.setCurrentIndex(combo.findData(dialog_module.POSEPAL_BACKEND))
    assert dialog.checkpoint_row.isHidden() is False


def test_custom_weights_are_remembered(dialog, tmp_path):
    """The stock checkpoint is a default: fine-tuned weights must not mean editing pose_fill."""
    weights = tmp_path / "animals.pth"
    dialog.checkpoint_edit.setText(str(weights))
    dialog.checkpoint_edit.editingFinished.emit()
    assert dialog.app_state.labelling_cotracker_checkpoint == str(weights)


def test_no_custom_weights_means_the_stock_checkpoint(dialog):
    """Empty is the default, not a path — build_backend must see None, not ''."""
    assert dialog.app_state.labelling_cotracker_checkpoint == ""
    assert (dialog.app_state.labelling_cotracker_checkpoint or None) is None


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


def test_the_confidence_filter_opens_on_at_most(dialog, monkeypatch):
    """Low confidence is what the column is read for, so "≤" is the default."""
    captured = {}

    class _Cancelled:
        def __init__(self, column, current, parent, default_op=">="):
            captured["default_op"] = default_op

        def exec_(self):
            return False

    monkeypatch.setattr(dialog_module, "NumericFilterDialog", _Cancelled)
    dialog._on_filter_requested(CONFIDENCE_COLUMN)

    assert captured["default_op"] == "<="


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
    """Filtering an x/y pair is meaningless — those are coordinates. Frame
    carries no funnel either: the rows are already ordered by it."""
    assert dialog.point_table.horizontalHeader().filterable == {
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


# ----------------------------------------------------------------------
# Test-time refinement
# ----------------------------------------------------------------------


def _select_backend(dialog, key: str) -> bool:
    index = dialog.backend_combo.findData(key)
    if index < 0:
        return False
    dialog.backend_combo.setCurrentIndex(index)
    return True


def test_the_refinement_row_belongs_to_posepal_alone(dialog):
    """Fit state is the one thing no other backend has — and it shows nowhere else."""
    # isHidden(), not isVisibleTo(): the tab holding the Fill group is itself
    # hidden while another tab is current, which says nothing about this row.
    _select_backend(dialog, "spline")
    dialog._refresh_backend_rows()
    assert dialog.refinement_row.isHidden()

    if not _select_backend(dialog, dialog_module.POSEPAL_BACKEND):
        pytest.skip("PosePAL not offered on this machine")
    dialog._refresh_backend_rows()
    assert not dialog.refinement_row.isHidden()


def test_an_unfitted_refinement_says_the_fill_will_fit(dialog):
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    assert "Not fitted" in dialog._refinement_status_text()


def test_labelling_another_frame_changes_the_refinement_signature(dialog):
    """A new label must mark a fit stale — it was made from fewer frames."""
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    before = dialog._refinement_signature()

    dialog.store.set_point(5, "beak", (3.0, 4.0))
    assert dialog._refinement_signature() != before

    # ...and moving a point counts just as much as adding one.
    moved = dialog._refinement_signature()
    dialog.store.set_point(5, "beak", (9.0, 9.0))
    assert dialog._refinement_signature() != moved


def test_a_schema_change_changes_the_refinement_signature(dialog):
    """The delta is indexed by point row, so a renamed keypoint invalidates it."""
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    before = dialog._refinement_signature()
    dialog._apply_schema(individuals=["a", "b"])
    assert dialog._refinement_signature() != before


def test_filling_is_the_only_fit_button(dialog):
    """Fill fits by itself when the labels have changed, so nothing else may.

    A second button for a step the first one already takes reads as a choice
    about the result, and there has never been one: a refit is a fresh fit.
    """
    assert not hasattr(dialog, "refit_btn")


class _CancellingBusy:
    """A progress dialog the user cancels as soon as work reports progress."""

    def __init__(self, label, parent=None):
        self.label = label

    def setLabelText(self, text) -> None:
        pass

    def pump_events(self) -> None:
        pass

    def wasCanceled(self) -> bool:
        return True

    def execute(self, fn, *args, **kwargs):
        return fn(*args, **kwargs), None


def test_cancelling_a_fill_keeps_the_fill_it_would_have_replaced(dialog, monkeypatch):
    """Cancelling is a way out of the wait, not a way to a worse fill.

    Backends answer a cancel with the spline seed they started from — they have
    arrays to return — so the dialog is what has to refuse it.
    """
    for frame in (0, 5):
        dialog.store.set_point(frame, "beak", (float(frame), 1.0))
    _fill(dialog.store, value=5.0)
    before = dialog.store.filled.copy()

    monkeypatch.setattr(dialog_module, "BusyProgressDialog", _CancellingBusy)
    dialog._on_fill()

    np.testing.assert_array_equal(dialog.store.filled, before)


def test_cancelling_the_first_fill_leaves_no_fill(dialog, monkeypatch):
    for frame in (0, 5):
        dialog.store.set_point(frame, "beak", (float(frame), 1.0))
    monkeypatch.setattr(dialog_module, "BusyProgressDialog", _CancellingBusy)

    dialog._on_fill()

    assert dialog.store.filled is None
