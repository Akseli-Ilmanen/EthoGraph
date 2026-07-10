"""Popup dialog for showing/hiding individual pose keypoints."""

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class KeypointFilterDialog(QDialog):
    """Table of keypoints with show/hide checkboxes.

    Emits ``hidden_changed`` (a ``set[str]`` of hidden keypoint names) on
    every toggle so the pose display updates live while the dialog is open.
    """

    hidden_changed = Signal(object)  # set[str]

    def __init__(self, keypoint_names: list[str], hidden: set[str], parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Filter keypoints")
        self.resize(280, 420)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.table = QTableWidget(len(keypoint_names), 2)
        self.table.setHorizontalHeaderLabels(["Show", "Keypoint"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        for row, name in enumerate(keypoint_names):
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox_item.setCheckState(Qt.Unchecked if name in hidden else Qt.Checked)
            self.table.setItem(row, 0, checkbox_item)

            name_item = QTableWidgetItem(str(name))
            name_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 1, name_item)
        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        close_btn = QPushButton("Close")
        select_all_btn.clicked.connect(lambda: self._set_all(True))
        deselect_all_btn.clicked.connect(lambda: self._set_all(False))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(deselect_all_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def hidden_keypoints(self) -> set[str]:
        hidden: set[str] = set()
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() != Qt.Checked:
                hidden.add(self.table.item(row, 1).text())
        return hidden

    def _on_cell_changed(self, row: int, column: int):
        if column == 0:
            self.hidden_changed.emit(self.hidden_keypoints())

    def _set_all(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(state)
        self.table.blockSignals(False)
        self.hidden_changed.emit(self.hidden_keypoints())
