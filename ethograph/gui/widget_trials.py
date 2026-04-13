"""Trials metadata widget: tabular display + combinatorial filtering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QRect, Signal
from qtpy.QtGui import QColor, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QDialog,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_MAX_INT_CATEGORICAL_VALUES = 15


class _CatFilterDialog(QDialog):
    """Checkbox popup for categorical column filtering."""

    def __init__(self, all_values: list[str], active: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter")
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(8, 8, 8, 8)

        self._all_cb = QCheckBox("(All)")
        self._all_cb.setChecked(not active)
        layout.addWidget(self._all_cb)

        self._checks: list[tuple[str, QCheckBox]] = []
        for val in sorted(all_values):
            cb = QCheckBox(val)
            cb.setChecked((not active) or (val in active))
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
            cb.setChecked(checked)

    def _on_item(self, _):
        self._all_cb.blockSignals(True)
        self._all_cb.setChecked(all(cb.isChecked() for _, cb in self._checks))
        self._all_cb.blockSignals(False)

    def get_allowed(self) -> set[str]:
        checked = {v for v, cb in self._checks if cb.isChecked()}
        return set() if checked == {v for v, _ in self._checks} else checked


class _NumFilterDialog(QDialog):
    """Threshold filter dialog for numeric columns."""

    def __init__(self, current: tuple[str, float] | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter")
        self._cleared = False

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        op_row = QHBoxLayout()
        self._op_combo = QComboBox()
        self._op_combo.addItems(["≥", "≤"])
        op_row.addWidget(self._op_combo)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(-1e9, 1e9)
        self._spin.setDecimals(6)
        op_row.addWidget(self._spin)
        layout.addLayout(op_row)

        if current:
            op, value = current
            self._op_combo.setCurrentText("≥" if op == ">=" else "≤")
            self._spin.setValue(value)

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
        return (">=" if self._op_combo.currentText() == "≥" else "<=", self._spin.value())


class _FilterHeaderView(QHeaderView):
    """Column header that draws filter icons and opens filter dialogs on click."""

    filter_requested = Signal(int)
    _FILTER_ZONE_W = 20

    def __init__(self, cat_cols: set[int], num_cols: set[int], parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._cat_cols = cat_cols
        self._num_cols = num_cols
        self._active: set[int] = set()
        self.setSectionsClickable(True)

    @property
    def _all_filterable(self) -> set[int]:
        return self._cat_cols | self._num_cols

    def set_filterable(self, cat_cols: set[int], num_cols: set[int]):
        self._cat_cols = cat_cols
        self._num_cols = num_cols
        self.viewport().update()

    def set_active_filters(self, active: set[int]):
        self._active = active
        self.viewport().update()

    def _filter_zone_x(self, logical: int) -> int:
        return self.sectionViewportPosition(logical) + self.sectionSize(logical) - self._FILTER_ZONE_W

    def _icon_rect(self, logical: int) -> QRect:
        size = 11
        zone_x = self._filter_zone_x(logical)
        h = self.height()
        x = zone_x + (self._FILTER_ZONE_W - size) // 2
        return QRect(x, (h - size) // 2, size, size)

    def paintSection(self, painter, rect, logical):
        painter.save()
        super().paintSection(painter, rect, logical)
        painter.restore()

        if logical not in self._all_filterable:
            return

        zone_x = rect.right() - self._FILTER_ZONE_W + 1
        painter.save()
        painter.setPen(QPen(QColor(120, 120, 120, 80), 1))
        painter.drawLine(zone_x, rect.top() + 3, zone_x, rect.bottom() - 3)
        painter.restore()

        ir = QRect(
            zone_x + (self._FILTER_ZONE_W - 11) // 2,
            rect.top() + (rect.height() - 11) // 2,
            11,
            11,
        )
        x, y, s = ir.x(), ir.y(), ir.width()
        color = QColor(255, 215, 0) if logical in self._active else QColor(180, 180, 180)
        painter.save()
        painter.setPen(QPen(color, 1.5))
        painter.drawLine(x, y, x + s, y)
        painter.drawLine(x + 2, y + 4, x + s - 2, y + 4)
        painter.drawLine(x + 4, y + 8, x + s - 4, y + 8)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            logical = self.logicalIndexAt(event.pos())
            if logical in self._all_filterable:
                section_right = self.sectionViewportPosition(logical) + self.sectionSize(logical)
                if event.pos().x() >= section_right - self._FILTER_ZONE_W:
                    self.filter_requested.emit(logical)
                    event.accept()
                    return
        super().mousePressEvent(event)


class TrialsWidget(QWidget):
    """Tabular metadata display with combinatorial filtering.

    Shows one row per trial, all metadata columns, and a dynamic filter
    combo per condition column.  Filters are AND-combined.

    Signals
    -------
    trials_filtered : list
        Emitted when active filters change; carries the filtered trial list.
    """

    trials_filtered = Signal(list)

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._metadata_df: pd.DataFrame = pd.DataFrame()
        self._base_trials: list = []
        self._cat_cols: set[int] = set()
        self._num_cols: set[int] = set()
        self._cat_values: dict[int, list[str]] = {}
        self._cat_active: dict[int, set[str]] = {}
        self._num_active: dict[int, tuple[str, float]] = {}
        self._building = False

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)

        # Table
        self._table = QTableWidget()
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.cellClicked.connect(self._on_row_clicked)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumHeight(280)
        layout.addWidget(self._table)

        # Status
        self._status_label = QLabel()
        layout.addWidget(self._status_label)

        self._note_label = QLabel(
            "Visible rows define which trials are used for trial navigation."
        )
        self._note_label.setWordWrap(True)
        self._note_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self._note_label)

    def setup(self, metadata_df: pd.DataFrame) -> None:
        """Populate the widget from a metadata DataFrame."""
        self._building = True
        if "trial" not in metadata_df.columns:
            raise ValueError("TrialsWidget requires a metadata table with a 'trial' column")
        self._metadata_df = metadata_df.copy()
        self._base_trials = self._metadata_df["trial"].dropna().drop_duplicates().tolist()
        self._cat_active.clear()
        self._num_active.clear()

        # Populate table
        self._refresh_table()
        self._setup_filter_header()

        self._building = False
        self._apply_filters()

    def _refresh_table(self) -> None:
        """Rebuild the table from _metadata_df."""
        df = self._metadata_df
        if df.empty:
            self._table.clear()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return

        cols = list(df.columns)
        self._table.setSortingEnabled(False)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(df))

        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(cols):
                val = row[col]
                item = QTableWidgetItem()
                if pd.notna(val):
                    # Store numeric types natively so Qt sorts numerically
                    if isinstance(val, (int, np.integer)):
                        item.setData(Qt.DisplayRole, int(val))
                    elif isinstance(val, (float, np.floating)):
                        item.setData(Qt.DisplayRole, float(val))
                    else:
                        item.setData(Qt.DisplayRole, str(val))
                else:
                    item.setData(Qt.DisplayRole, "")
                item.setData(Qt.UserRole, row.get("trial"))
                self._table.setItem(r, c, item)

        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _setup_filter_header(self) -> None:
        self._cat_cols.clear()
        self._num_cols.clear()
        self._cat_values.clear()
        col_names = list(self._metadata_df.columns)

        df = self._metadata_df
        for col_idx, col_name in enumerate(df.columns):
            series = df[col_name].dropna()
            if series.empty:
                self._cat_cols.add(col_idx)
                self._cat_values[col_idx] = []
                continue

            if pd.api.types.is_integer_dtype(series) and series.nunique() <= _MAX_INT_CATEGORICAL_VALUES:
                self._cat_cols.add(col_idx)
                int_vals = sorted({int(v) for v in series.tolist()})
                self._cat_values[col_idx] = [str(v) for v in int_vals]
                continue

            if pd.api.types.is_bool_dtype(series):
                self._cat_cols.add(col_idx)
                self._cat_values[col_idx] = sorted({str(v) for v in series.tolist()})
                continue

            if pd.api.types.is_numeric_dtype(series):
                self._num_cols.add(col_idx)
            else:
                self._cat_cols.add(col_idx)
                self._cat_values[col_idx] = sorted({str(v) for v in series.tolist()})

        header = _FilterHeaderView(self._cat_cols, self._num_cols, self._table)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        header.setVisible(True)
        header.filter_requested.connect(self._on_filter_header_clicked)
        self._table.setHorizontalHeader(header)
        # Re-apply header labels after replacing the header view to ensure text is visible.
        self._table.setHorizontalHeaderLabels(col_names)
        self._update_header_active_filters()

    def _on_filter_header_clicked(self, logical_col: int) -> None:
        if logical_col in self._cat_cols:
            active = self._cat_active.get(logical_col, set())
            dialog = _CatFilterDialog(self._cat_values.get(logical_col, []), active, self)
            if dialog.exec_() != QDialog.Accepted:
                return
            allowed = dialog.get_allowed()
            if allowed:
                self._cat_active[logical_col] = allowed
            else:
                self._cat_active.pop(logical_col, None)
            self._update_header_active_filters()
            self._apply_filters()
            return

        if logical_col in self._num_cols:
            current = self._num_active.get(logical_col)
            dialog = _NumFilterDialog(current, self)
            if dialog.exec_() != QDialog.Accepted:
                return
            value = dialog.get_filter()
            if value is None:
                self._num_active.pop(logical_col, None)
            else:
                self._num_active[logical_col] = value
            self._update_header_active_filters()
            self._apply_filters()

    def _update_header_active_filters(self) -> None:
        header = self._table.horizontalHeader()
        if isinstance(header, _FilterHeaderView):
            active = set(self._cat_active.keys()) | set(self._num_active.keys())
            header.set_active_filters(active)

    def _apply_filters(self) -> None:
        """Compute filtered trial list and emit signal."""
        original_trials = set(self._base_trials)
        filtered = set(original_trials)

        mdf = self._metadata_df
        if not mdf.empty:
            for _, row in mdf.iterrows():
                trial = row.get("trial")
                if pd.isna(trial):
                    continue

                keep = True
                for col_idx, allowed in self._cat_active.items():
                    col_name = mdf.columns[col_idx]
                    if str(row.get(col_name, "")) not in allowed:
                        keep = False
                        break

                if keep:
                    for col_idx, (op, threshold) in self._num_active.items():
                        col_name = mdf.columns[col_idx]
                        val = row.get(col_name)
                        if pd.isna(val):
                            keep = False
                            break
                        if op == ">=" and float(val) < threshold:
                            keep = False
                            break
                        if op == "<=" and float(val) > threshold:
                            keep = False
                            break

                if not keep and trial in filtered:
                    filtered.remove(trial)

        try:
            sorted_trials = sorted(filtered, key=int)
        except (ValueError, TypeError):
            sorted_trials = sorted(filtered, key=str)

        # Never produce an empty trial list — fall back to unfiltered.
        if not sorted_trials:
            sorted_trials = list(self._base_trials)

        # Keep table and navigation in sync with filtered subset.
        for row_idx in range(self._table.rowCount()):
            item = self._table.item(row_idx, 0)
            trial_id = item.data(Qt.UserRole) if item is not None else None
            self._table.setRowHidden(row_idx, trial_id not in filtered)

        self.app_state.trials = sorted_trials

        n_total = len(original_trials)
        n_shown = len(sorted_trials)
        self._status_label.setText(f"Showing {n_shown} of {n_total} trials")

        self.trials_filtered.emit(sorted_trials)

    def _on_row_clicked(self, row: int, _col: int) -> None:
        """Navigate to the trial in the clicked row."""
        item = self._table.item(row, 0)
        if item is None:
            return
        trial_id = item.data(Qt.UserRole)
        if trial_id is not None and trial_id in set(self.app_state.trials) and hasattr(self.app_state, "set_key_sel"):
            self.app_state.set_key_sel("trials", trial_id)
            self.app_state.trial_changed.emit()
