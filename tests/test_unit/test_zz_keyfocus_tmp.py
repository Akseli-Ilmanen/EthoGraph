"""Do Tab / Backspace / Ctrl+Z reach the dialog when a child widget has focus?

The existing tests send the key straight to a widget without giving it focus,
which is not what happens in the app.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pygfx")

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from ethograph.gui.app_state import ObservableAppState  # noqa: E402
from ethograph.gui.dialog_pose_labelling import PoseLabellingDialog  # noqa: E402
from ethograph.gui.pose_edit_mixin import SEQUENTIAL_MODE  # noqa: E402
from tests.test_unit.test_pose_labelling_dialog import NAMES, _FakeDataWidget  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state.keypoints = list(NAMES)
    state.labelling_keypoints = list(NAMES)
    dlg = PoseLabellingDialog(_FakeDataWidget(state))
    dlg.show()
    dlg.activateWindow()
    QApplication.processEvents()
    yield dlg
    dlg.close()


def _press(qapp, key, modifiers=Qt.NoModifier):
    """As close to a real key press as a test gets: to the real focus widget."""
    from qtpy.QtCore import QEvent
    from qtpy.QtGui import QKeyEvent

    target = QApplication.focusWidget()
    assert target is not None, "nothing has focus"
    QApplication.sendEvent(target, QKeyEvent(QEvent.KeyPress, key, modifiers))
    return target


def test_tab_from_the_tree(dialog, qapp):
    from qtpy.QtTest import QTest

    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.tree.setFocus()
    QApplication.processEvents()
    before = dialog._mode.active_keypoint
    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
    QApplication.processEvents()
    print(f"\nTAB   {before!r} -> {dialog._mode.active_keypoint!r}")
    assert dialog._mode.active_keypoint != before


def test_backspace_from_the_tree(dialog, qapp):
    dialog.store.set_point(0, "tail", (5.0, 6.0))
    for i in range(dialog.tree.topLevelItemCount()):
        branch = dialog.tree.topLevelItem(i)
        for k in range(branch.childCount()):
            if branch.child(k).data(0, Qt.UserRole)[1] == "tail":
                dialog.tree.setCurrentItem(branch.child(k))
    dialog.tree.setFocus()
    QApplication.processEvents()
    focused = _press(qapp, Qt.Key_Backspace)
    print(f"\nBKSP  focus={type(focused).__name__} anchor={dialog.store.is_anchor(0, 'tail')}")
    assert dialog.store.is_anchor(0, "tail") is False


def test_undo_from_the_table(dialog, qapp):
    dialog.store.set_point(0, "beak", (5.0, 6.0))
    dialog.point_table.setFocus()
    QApplication.processEvents()
    focused = _press(qapp, Qt.Key_Z, Qt.ControlModifier)
    print(f"\nUNDO  focus={type(focused).__name__} anchor={dialog.store.is_anchor(0, 'beak')}")
    assert dialog.store.is_anchor(0, "beak") is False


def test_backspace_from_the_canvas(dialog, qapp):
    dialog.store.set_point(0, "eye", (5.0, 6.0))
    for i in range(dialog.tree.topLevelItemCount()):
        branch = dialog.tree.topLevelItem(i)
        for k in range(branch.childCount()):
            if branch.child(k).data(0, Qt.UserRole)[1] == "eye":
                dialog.tree.setCurrentItem(branch.child(k))
    canvas = dialog._view.canvas_widget()
    canvas.setFocusPolicy(Qt.StrongFocus)
    canvas.setFocus()
    QApplication.processEvents()
    focused = _press(qapp, Qt.Key_Backspace)
    print(f"\nCANV  focus={type(focused).__name__} anchor={dialog.store.is_anchor(0, 'eye')}")
    assert dialog.store.is_anchor(0, "eye") is False
