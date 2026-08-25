"""Reusable column filtering for Qt table views.

Excel-style filtering: a funnel icon sits in a reserved zone at the right of
every filterable header section, clicking it opens a checkbox popup (categorical
columns) or a threshold dialog (numeric columns), and
:class:`MultiColumnFilterProxy` ANDs the criteria of every column together.

Originally written for the Kilosort cluster table in
:mod:`~ethograph.gui.widgets_ephys` and lifted here so the keypoint labelling
dialog can use the same interaction. The proxy reads through ``QModelIndex``
rather than ``QStandardItemModel.item()``, so it works over any model — the
cluster table's item model as well as the labelling dialog's virtual one.

Numeric columns are compared on :data:`SORT_ROLE`, not on the displayed text,
so a formatted cell ("12.3 ms") still filters and sorts by its real value.
"""

from __future__ import annotations

import re

from qtpy.QtCore import QRect, QSortFilterProxyModel, Qt, Signal
from qtpy.QtGui import QColor, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QVBoxLayout,
)

#: Role carrying a cell's sortable/comparable value, independent of its text.
SORT_ROLE = Qt.UserRole + 1

_DIGIT_RUN = re.compile(r"(\d+)")


def _natural_key(text: str) -> tuple:
    """Split *text* into alternating non-digit/digit chunks for natural sort.

    ``re.split`` on a capturing group always alternates non-digit, digit,
    non-digit, ... starting and ending on a (possibly empty) non-digit chunk,
    so same-index chunks compare same-typed across two keys and "20" sorts
    before "120" instead of after it.
    """
    parts = _DIGIT_RUN.split(text)
    return tuple(int(part) if i % 2 else part.lower() for i, part in enumerate(parts))


class MultiColumnFilterProxy(QSortFilterProxyModel):
    """Filters rows by categorical or numeric criteria on several columns at once.

    An empty ``allowed`` set removes a column's filter — "everything is allowed"
    and "no filter" are the same thing, and keeping them distinct only invites
    a state where every row is hidden.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cat_filters: dict[int, set[str]] = {}
        self._num_filters: dict[int, tuple[str, float]] = {}

    def set_cat_filter(self, col: int, allowed: set[str]) -> None:
        if not allowed:
            self._cat_filters.pop(col, None)
        else:
            self._cat_filters[col] = set(allowed)
        self.invalidateFilter()

    def set_numeric_filter(self, col: int, op: str | None, value: float | None) -> None:
        if op is None or value is None:
            self._num_filters.pop(col, None)
        else:
            self._num_filters[col] = (op, float(value))
        self.invalidateFilter()

    def cat_filter(self, col: int) -> set[str]:
        """The values *col* is restricted to; empty means unfiltered."""
        return set(self._cat_filters.get(col, ()))

    def num_filter(self, col: int) -> tuple[str, float] | None:
        """``(op, threshold)`` for *col*, or ``None`` when unfiltered."""
        return self._num_filters.get(col)

    def active_filters(self) -> set[int]:
        """Columns currently carrying a filter — what the header marks."""
        return set(self._cat_filters) | set(self._num_filters)

    def clear_filters(self) -> None:
        self._cat_filters.clear()
        self._num_filters.clear()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent):
        model = self.sourceModel()
        for col, allowed in self._cat_filters.items():
            if model.index(source_row, col, source_parent).data(Qt.DisplayRole) not in allowed:
                return False
        for col, (op, threshold) in self._num_filters.items():
            try:
                value = float(model.index(source_row, col, source_parent).data(SORT_ROLE))
            except (ValueError, TypeError):
                return False
            if op == ">=" and value < threshold:
                return False
            if op == "<=" and value > threshold:
                return False
        return True

    def lessThan(self, left, right):
        left_val, right_val = left.data(SORT_ROLE), right.data(SORT_ROLE)
        if left_val is not None and right_val is not None:
            return float(left_val) < float(right_val)
        left_text = left.data(Qt.DisplayRole)
        right_text = right.data(Qt.DisplayRole)
        if isinstance(left_text, str) and isinstance(right_text, str):
            return _natural_key(left_text) < _natural_key(right_text)
        return super().lessThan(left, right)


class CategoryFilterDialog(QDialog):
    """Checkbox popup for categorical column filtering."""

    def __init__(self, col: int, all_values: list[str], active: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter")
        self._col = col
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(8, 8, 8, 8)
        self._all_cb = QCheckBox("(All)")
        self._all_cb.setChecked(not active)
        layout.addWidget(self._all_cb)
        self._checks: list[tuple[str, QCheckBox]] = []
        for val in sorted(all_values):
            cb = QCheckBox(val)
            cb.setChecked(not active or val in active)
            layout.addWidget(cb)
            self._checks.append((val, cb))
        self._all_cb.toggled.connect(self._on_all)
        for _, cb in self._checks:
            cb.toggled.connect(self._on_item)
        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def _on_all(self, checked: bool):
        for _, cb in self._checks:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

    def _on_item(self, _):
        self._all_cb.blockSignals(True)
        self._all_cb.setChecked(all(cb.isChecked() for _, cb in self._checks))
        self._all_cb.blockSignals(False)

    def get_allowed(self) -> set[str]:
        """The checked values, or an empty set when they all are (= no filter)."""
        checked = {v for v, cb in self._checks if cb.isChecked()}
        return set() if checked == {v for v, _ in self._checks} else checked


class NumericFilterDialog(QDialog):
    """Threshold filter dialog for numeric columns.

    ``default_op`` picks which comparison a fresh filter opens on: which of
    the two is the interesting one depends on the column (a firing rate is
    usually filtered from below, a confidence from above).
    """

    def __init__(
        self,
        col: int,
        current: tuple[str, float] | None,
        parent=None,
        default_op: str = ">=",
    ):
        super().__init__(parent)
        self.setWindowTitle("Filter")
        self._col = col
        self._cleared = False
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        op_row = QHBoxLayout()
        self._op_combo = QComboBox()
        self._op_combo.addItems(["≥", "≤"])
        self._op_combo.setCurrentText("≥" if default_op == ">=" else "≤")
        op_row.addWidget(self._op_combo)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(-1e9, 1e9)
        self._spin.setDecimals(3)
        op_row.addWidget(self._spin)
        layout.addLayout(op_row)
        if current:
            op, val = current
            self._op_combo.setCurrentText("≥" if op == ">=" else "≤")
            self._spin.setValue(val)
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        clear_btn = QPushButton("Remove filter")
        clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

    def _clear(self):
        self._cleared = True
        self.accept()

    def get_filter(self) -> tuple[str, float] | None:
        if self._cleared:
            return None
        return (
            ">=" if self._op_combo.currentText() == "≥" else "<=",
            self._spin.value(),
        )


class FilterHeaderView(QHeaderView):
    """Column header that draws filter icons for filterable columns.

    A zone of width :attr:`FILTER_ZONE_W` is reserved on the right of each
    filterable column. Clicking inside it emits :attr:`filter_requested`;
    clicking elsewhere behaves normally (sorting, resizing). Subclasses that
    override ``paintSection`` must call :meth:`paint_filter_icon` themselves.
    """

    filter_requested = Signal(int)
    FILTER_ZONE_W = 20  # px reserved on the right of filterable columns

    def __init__(self, cat_cols: set[int] | None = None, num_cols: set[int] | None = None, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._cat_cols = set(cat_cols or ())
        self._num_cols = set(num_cols or ())
        self._active: set[int] = set()
        self.setSectionsClickable(True)

    def set_filterable(self, cat_cols: set[int], num_cols: set[int]) -> None:
        self._cat_cols = set(cat_cols)
        self._num_cols = set(num_cols)
        self.viewport().update()

    def set_active_filters(self, active: set[int]) -> None:
        self._active = set(active)
        self.viewport().update()

    @property
    def filterable(self) -> set[int]:
        return self._cat_cols | self._num_cols

    def is_categorical(self, logical: int) -> bool:
        return logical in self._cat_cols

    def is_numeric(self, logical: int) -> bool:
        return logical in self._num_cols

    def filter_zone_x(self, logical: int) -> int:
        """Left edge of the filter zone for *logical* column."""
        return self.sectionViewportPosition(logical) + self.sectionSize(logical) - self.FILTER_ZONE_W

    def _icon_rect(self, logical: int) -> QRect:
        s = 11
        x = self.filter_zone_x(logical) + (self.FILTER_ZONE_W - s) // 2
        return QRect(x, (self.height() - s) // 2, s, s)

    def paint_filter_icon(self, painter, rect, logical: int) -> None:
        """Draw the funnel (and its separator) for *logical*, if it filters."""
        if logical not in self.filterable:
            return
        zone_x = self.filter_zone_x(logical)
        painter.save()
        painter.setPen(QPen(QColor(120, 120, 120, 80), 1))
        painter.drawLine(zone_x, rect.top() + 3, zone_x, rect.bottom() - 3)
        painter.restore()

        icon = self._icon_rect(logical)
        x, y, s = icon.x(), icon.y(), icon.width()
        color = QColor(255, 215, 0) if logical in self._active else QColor(180, 180, 180)
        painter.save()
        painter.setPen(QPen(color, 1.5))
        painter.drawLine(x, y, x + s, y)
        painter.drawLine(x + 2, y + 4, x + s - 2, y + 4)
        painter.drawLine(x + 4, y + 8, x + s - 4, y + 8)
        painter.restore()

    def paintSection(self, painter, rect, logical):
        painter.save()
        super().paintSection(painter, rect, logical)
        painter.restore()
        self.paint_filter_icon(painter, rect, logical)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            logical = self.logicalIndexAt(event.pos())
            if logical in self.filterable and event.pos().x() >= self.filter_zone_x(logical):
                self.filter_requested.emit(logical)
                return
        super().mousePressEvent(event)
