from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QSortFilterProxyModel, Qt
from qtpy.QtGui import QColor, QFont, QIcon, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QCompleter,
    QLineEdit,
    QStyledItemDelegate,
    QWidget,
)

# ── QComboBox helpers ────────────────────────────────────────────────────────


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


def get_combo_value(combo: QComboBox) -> str:
    data = combo.currentData(Qt.ItemDataRole.UserRole)
    return data if data is not None else combo.currentText()


def find_combo_index(combo: QComboBox, value: str) -> int:
    idx = combo.findData(value, Qt.ItemDataRole.UserRole)
    if idx < 0:
        idx = combo.findText(str(value))
    return idx


def set_combo_to_value(combo: QComboBox, value: str) -> None:
    idx = find_combo_index(combo, str(value))
    if idx >= 0:
        combo.setCurrentIndex(idx)


def add_combo_separator(combo: QComboBox) -> None:
    from qtpy.QtGui import QStandardItem, QStandardItemModel

    model: QStandardItemModel = combo.model()
    item = QStandardItem("\u2500" * 24)
    item.setFlags(Qt.NoItemFlags)
    model.appendRow(item)


# ── QIcon helpers ────────────────────────────────────────────────────────────


def color_icon(color_01: tuple, size: int = 14) -> QIcon:
    r, g, b = (int(c * 255) for c in color_01[:3])
    pix = QPixmap(size, size)
    pix.fill(QColor(r, g, b))
    return QIcon(pix)


def gray_icon(size: int = 14) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(QColor(160, 160, 160))
    return QIcon(pix)


# ── QFont helpers ────────────────────────────────────────────────────────────


def mono_font(size: int = 13) -> QFont:
    f = QFont("Menlo")
    f.setPointSize(size)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


# ── QLineEdit helpers ────────────────────────────────────────────────────────


def populate_if_exists(line_edit: QLineEdit, path: str | Path | None) -> None:
    """Set a QLineEdit's text only if *path* points to an existing file or folder."""
    if path is None:
        return
    p = Path(path)
    if p.exists():
        line_edit.setText(str(p))


# ── Delegate ─────────────────────────────────────────────────────────────────


class ElidedDelegate(QStyledItemDelegate):
    def __init__(self, elide_mode=Qt.ElideMiddle, parent=None):
        super().__init__(parent)
        self._elide_mode = elide_mode

    def paint(self, painter, option, index):
        text = index.data(Qt.DisplayRole)
        if text:
            metrics = painter.fontMetrics()
            elided = metrics.elidedText(text, self._elide_mode, option.rect.width())
            painter.drawText(option.rect, Qt.AlignVCenter | Qt.AlignLeft, elided)
        else:
            super().paint(painter, option, index)


# ── Widget styling ───────────────────────────────────────────────────────────


def apply_compact_widget_style(widget: QWidget, font_size: int = 8) -> None:
    """Apply compact font and stylesheet to a widget subtree."""
    font = QFont()
    font.setPointSize(font_size)
    widget.setFont(font)

    widget.setStyleSheet(f"""
        * {{
            font-size: {font_size}pt;
            padding: 2px;
            margin: 1px;
        }}
        QLabel {{
            font-size: {font_size}pt;
            padding: 2px;
        }}
        QPushButton {{
            font-size: {font_size}pt;
            padding: 4px 8px;
        }}
        QComboBox {{
            font-size: {font_size}pt;
            padding: 2px 4px;
        }}
        QSpinBox, QDoubleSpinBox {{
            font-size: {font_size}pt;
            padding: 2px;
        }}
        QLineEdit {{
            font-size: {font_size}pt;
            padding: 2px 4px;
        }}
        QGroupBox {{
            margin-top: 4px;
            margin-bottom: 2px;
            padding-top: 12px;
        }}
        QGroupBox::title {{
            padding: 2px 4px;
        }}
        QFrame {{
            margin: 1px;
            padding: 1px;
        }}
        QCollapsible {{
            margin: 1px;
            padding: 1px;
            border: none;
            spacing: 2px;
        }}
        QCollapsible > QToolButton {{
            padding: 2px 6px;
            margin: 1px;
            min-height: 18px;
            max-height: 20px;
            border: none;
            border-bottom: 1px solid palette(mid);
        }}
        QCollapsible > QFrame {{
            margin: 2px;
            padding: 2px;
            border: none;
        }}
    """)


def normalize_child_layouts(root: QWidget, spacing: int, margin: int) -> None:
    """Apply consistent spacing/margins to direct child widget layouts."""
    layout = root.layout()
    if layout is None:
        return
    for i in range(layout.count()):
        item = layout.itemAt(i)
        child = item.widget() if item else None
        child_layout = child.layout() if child is not None else None
        if child_layout is None:
            continue
        child_layout.setSpacing(spacing)
        child_layout.setContentsMargins(margin, margin, margin, margin)
