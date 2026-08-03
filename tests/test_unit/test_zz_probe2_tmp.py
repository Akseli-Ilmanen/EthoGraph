"""Probe 2: are the dialog's Tab shortcuts registered, and do they fire?"""

from __future__ import annotations

import pytest

pytest.importorskip("pygfx")

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from ethograph.gui.app_state import ObservableAppState  # noqa: E402
from ethograph.gui.dialog_pose_labelling import PoseLabellingDialog  # noqa: E402
from ethograph.gui.pose_edit_mixin import SEQUENTIAL_MODE  # noqa: E402
from tests.test_unit.test_pose_labelling_dialog import NAMES, _FakeDataWidget  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_probe(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state.keypoints = list(NAMES)
    state.labelling_keypoints = list(NAMES)
    dlg = PoseLabellingDialog(_FakeDataWidget(state))
    dlg.show()
    dlg.activateWindow()
    QApplication.processEvents()

    print(f"\nshortcuts={dlg._shortcuts}")
    for sc in dlg._shortcuts:
        print(f"  key={sc.key().toString()!r} enabled={sc.isEnabled()} ctx={sc.context()}")

    dlg.set_interaction_mode(SEQUENTIAL_MODE)
    print(f"mode={dlg._mode} active={dlg._mode.active_keypoint!r}")

    hits = []
    for sc in dlg._shortcuts:
        sc.activated.connect(lambda: hits.append("fired"))
        sc.activatedAmbiguously.connect(lambda: hits.append("ambiguous"))

    dlg.tree.setFocus()
    QApplication.processEvents()
    print(f"active_window={dlg.isActiveWindow()} focus={type(QApplication.focusWidget()).__name__}")
    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
    QApplication.processEvents()
    print(f"hits={hits} active={dlg._mode.active_keypoint!r}")

    # And directly, bypassing the key entirely:
    dlg._cycle_keypoint(1)
    print(f"after direct _cycle_keypoint: {dlg._mode.active_keypoint!r}")
    dlg.close()
