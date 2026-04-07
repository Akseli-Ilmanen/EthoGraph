from qtpy.QtCore import Qt, QSortFilterProxyModel
from qtpy.QtWidgets import QComboBox, QCompleter


def make_searchable(combo_box: QComboBox) -> None:
    combo_box.setFocusPolicy(Qt.StrongFocus)
    combo_box.setEditable(True)
    combo_box.setInsertPolicy(QComboBox.NoInsert)

    filter_model = QSortFilterProxyModel(combo_box)
    filter_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
    filter_model.setSourceModel(combo_box.model())

    completer = QCompleter(filter_model, combo_box)
    completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
    combo_box.setCompleter(completer)
    combo_box.lineEdit().textEdited.connect(filter_model.setFilterFixedString)
