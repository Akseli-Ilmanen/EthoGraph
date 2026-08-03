"""Probe: does a QShortcut fire at all in this harness, and with which context?"""

from __future__ import annotations

import pytest

pytest.importorskip("pygfx")

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtGui import QKeySequence, QShortcut  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication, QDialog, QTreeWidget, QVBoxLayout  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("context", [Qt.WindowShortcut, Qt.ApplicationShortcut])
@pytest.mark.parametrize("key", [Qt.Key_Tab, Qt.Key_F7])
def test_shortcut_fires(qapp, context, key):
    dlg = QDialog()
    layout = QVBoxLayout(dlg)
    tree = QTreeWidget()
    layout.addWidget(tree)
    dlg.show()
    QApplication.processEvents()
    dlg.activateWindow()
    QApplication.processEvents()

    fired = []
    sc = QShortcut(QKeySequence(key), dlg)
    sc.setContext(context)
    sc.activated.connect(lambda: fired.append(1))

    tree.setFocus()
    QApplication.processEvents()
    QTest.keyClick(tree, key)
    QApplication.processEvents()

    print(
        f"\nkey={key} context={context} fired={bool(fired)} "
        f"active={dlg.isActiveWindow()} focus={type(QApplication.focusWidget()).__name__}"
    )
    dlg.close()
    assert fired
