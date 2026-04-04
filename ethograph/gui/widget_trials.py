"""Trials metadata widget: tabular display + combinatorial filtering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.io.metadata_table import condition_columns

logger = logging.getLogger(__name__)


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
        self._filter_combos: dict[str, QComboBox] = {}
        self._confidence_combo: QComboBox | None = None
        self._building = False

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)

        # Filter row
        self._filter_group = QGroupBox("Filter trials")
        self._filter_layout = QHBoxLayout()
        self._filter_group.setLayout(self._filter_layout)
        layout.addWidget(self._filter_group)

        # Confidence filter (always present)
        self._confidence_combo = QComboBox()
        self._confidence_combo.addItems(["Show all", "Low confidence only", "High confidence only"])
        self._confidence_combo.currentTextChanged.connect(self._on_filter_changed)
        conf_label = QLabel("Confidence:")
        self._filter_layout.addWidget(conf_label)
        self._filter_layout.addWidget(self._confidence_combo)
        self._filter_layout.addStretch()

        # Table
        self._table = QTableWidget()
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.cellClicked.connect(self._on_row_clicked)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

        # Status
        self._status_label = QLabel()
        layout.addWidget(self._status_label)

    def setup(self, metadata_df: pd.DataFrame) -> None:
        """Populate the widget from a metadata DataFrame."""
        self._building = True
        self._metadata_df = metadata_df

        # Clear old dynamic filter combos
        for combo in self._filter_combos.values():
            combo.currentTextChanged.disconnect(self._on_filter_changed)
            combo.setParent(None)
            combo.deleteLater()
        # Remove old labels (all widgets except confidence label/combo and stretch)
        while self._filter_layout.count() > 3:
            item = self._filter_layout.takeAt(2)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        self._filter_combos.clear()

        # Build filter combos for each condition column
        cond_cols = condition_columns(metadata_df)
        # Re-insert stretch at end
        stretch_item = self._filter_layout.takeAt(self._filter_layout.count() - 1)

        for col in cond_cols:
            label = QLabel(f"{col}:")
            combo = QComboBox()
            unique_vals = sorted(metadata_df[col].dropna().unique(), key=str)
            combo.addItem("All")
            combo.addItems([str(v) for v in unique_vals])
            combo.currentTextChanged.connect(self._on_filter_changed)
            self._filter_layout.addWidget(label)
            self._filter_layout.addWidget(combo)
            self._filter_combos[col] = combo

        self._filter_layout.addStretch()

        self._filter_group.setVisible(bool(cond_cols))

        # Populate table
        self._refresh_table()

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
                item = QTableWidgetItem(str(val) if pd.notna(val) else "")
                item.setData(Qt.UserRole, row.get("trial"))
                self._table.setItem(r, c, item)

        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _on_filter_changed(self, _text=None) -> None:
        if self._building:
            return
        self._apply_filters()

    def _apply_filters(self) -> None:
        """Compute filtered trial list and emit signal."""
        dt = self.app_state.dt
        if dt is None:
            return

        original_trials = set(dt.trials)
        filtered = set(original_trials)

        # Condition filters
        mdf = self._metadata_df
        if not mdf.empty:
            for col, combo in self._filter_combos.items():
                val = combo.currentText()
                if val == "All":
                    continue
                matching = set()
                for _, row in mdf.iterrows():
                    if str(row.get(col, "")) == val:
                        matching.add(row["trial"])
                filtered &= matching

        # Confidence filter
        if self._confidence_combo is not None:
            mode = self._confidence_combo.currentText()
            levels = getattr(self.app_state, "pred_confidence_levels", {})
            if mode != "Show all" and levels:
                target = "low" if mode == "Low confidence only" else "high"
                filtered = {t for t in filtered if levels.get(t) == target}

        try:
            sorted_trials = sorted(filtered, key=int)
        except (ValueError, TypeError):
            sorted_trials = sorted(filtered, key=str)

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
        if trial_id is not None and hasattr(self.app_state, "set_key_sel"):
            self.app_state.set_key_sel("trials", trial_id)
            self.app_state.trial_changed.emit()
