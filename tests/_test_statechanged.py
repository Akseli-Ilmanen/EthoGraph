from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QCheckBox

app = QApplication([])
cb = QCheckBox("x")
got = {}


def on_state(state):
    got["state"] = state
    got["type"] = type(state).__name__
    got["eq_enum"] = state == Qt.Checked
    got["eq_int"] = state == 2
    got["checkstate_eq"] = Qt.CheckState(state) == Qt.Checked


cb.stateChanged.connect(on_state)
cb.setChecked(True)
print(got)
