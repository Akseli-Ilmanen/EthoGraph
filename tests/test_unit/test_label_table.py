"""The label table dialog: what it writes back, and what it refuses.

The dialog addresses rows by **position** in ``_all_labels_df``, so the tests
that matter are the ones about the write-back path: a parsed cell reaching the
frame, a value the row contradicts being turned away, and a table replaced
elsewhere being re-read instead of written through at the old positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qtpy")

from qtpy.QtCore import QEvent, Qt  # noqa: E402
from qtpy.QtGui import QKeyEvent  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from ethograph.gui.dialog_label_table import LabelTableDialog, parse_cell  # noqa: E402
from ethograph.labels.tsv_store import get_trial_from_tsv, set_trial_in_tsv  # noqa: E402

MAPPINGS = {
    1: {"name": "peck", "color": np.array([1.0, 0.0, 0.0])},
    2: {"name": "hop", "color": np.array([0.0, 1.0, 0.0])},
}


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": ["1", "1", "2"],
            "individual": ["crow1", "crow1", "crow2"],
            "individual_rec": ["", "", ""],
            "labels": np.array([1, 2, 1], dtype=np.int32),
            "onset_s": [0.5, 1.5, 2.5],
            "offset_s": [1.0, np.nan, 3.0],  # row 1 is a point event
            "event_type": ["state", "point", "state"],
            "confidence": [1.0, 0.42, 1.0],
            "labeling_method": ["manual", "automated", "manual"],
            "changepoint_corrected": [0, 0, 0],
            "prediction_source": ["", "run1", ""],
            "n_samples": [100, 100, 100],
        }
    )


@pytest.fixture
def dialog(app_state, qtbot):
    app_state.trials = ["1", "2"]
    app_state.trials_sel = "1"
    app_state._all_labels_df = _labels()
    app_state.label_intervals = get_trial_from_tsv(app_state._all_labels_df, "1")
    dlg = LabelTableDialog(app_state, mappings=MAPPINGS)
    qtbot.addWidget(dlg)
    return dlg


def _col(dialog, name: str) -> int:
    """Model column of the label-table column *name* (0 is the derived name)."""
    return dialog._columns.index(name) + 1


class TestParseCell:
    def test_numbers_and_choices(self):
        assert parse_cell("onset_s", " 1.25 ") == 1.25
        assert parse_cell("labels", "3") == 3
        assert parse_cell("labeling_method", "curated") == "curated"

    def test_only_offset_may_be_empty(self):
        """An empty end is what a point event has; an empty onset is a slip."""
        assert np.isnan(parse_cell("offset_s", ""))
        with pytest.raises(ValueError):
            parse_cell("onset_s", "")

    def test_refuses_a_word_where_a_number_belongs(self):
        with pytest.raises(ValueError):
            parse_cell("confidence", "high")

    def test_refuses_a_method_outside_the_vocabulary(self):
        with pytest.raises(ValueError):
            parse_cell("labeling_method", "guessed")


class TestEdit:
    def test_cell_edit_reaches_the_table_and_is_undoable(self, dialog):
        state = dialog.app_state
        dialog._model.item(0, _col(dialog, "onset_s")).setText("0.750")
        assert state._all_labels_df.loc[state._all_labels_df["trial"] == "1", "onset_s"].min() == 0.75

        state.undo_label_edit()
        assert sorted(state._all_labels_df["onset_s"]) == [0.5, 1.5, 2.5]

    def test_backwards_interval_is_refused(self, dialog):
        """A negative duration is dropped on save, so it must not be typed in."""
        offset = _col(dialog, "offset_s")
        dialog._model.item(0, offset).setText("0.100")
        assert list(dialog.app_state._all_labels_df["offset_s"])[0] == 1.0
        assert dialog._model.item(0, offset).text() == "1.000"

    def test_unknown_class_is_refused(self, dialog):
        dialog._model.item(0, _col(dialog, "labels")).setText("99")
        assert list(dialog.app_state._all_labels_df["labels"]) == [1, 2, 1]

    def test_set_selected_cells_writes_every_selected_row(self, dialog, monkeypatch):
        monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: ("curated", True)))
        dialog.table.selectColumn(_col(dialog, "labeling_method"))
        dialog.set_selected_cells()
        assert list(dialog.app_state._all_labels_df["labeling_method"]) == ["curated"] * 3

    def test_read_only_column_is_never_written(self, dialog, monkeypatch):
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("9", True)))
        dialog.table.selectColumn(_col(dialog, "n_samples"))
        dialog.set_selected_cells()
        assert list(dialog.app_state._all_labels_df["n_samples"]) == [100, 100, 100]


class TestDelete:
    def test_deletes_every_row_the_selection_touches(self, dialog, monkeypatch):
        monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
        dialog.table.selectRow(0)
        dialog.table.selectionModel().select(
            dialog._proxy.index(1, 3),
            dialog.table.selectionModel().Select,
        )
        dialog.delete_selected_rows()
        assert list(dialog.app_state._all_labels_df["onset_s"]) == [2.5]

    def test_the_cursor_stays_where_the_deletion_happened(self, dialog):
        """Deleting must not send the user back to the top of the table."""
        dialog.table.selectRow(1)
        dialog.delete_selected_rows()
        assert dialog.table.currentIndex().row() == 1  # the row that moved up

    def test_deleting_the_last_row_lands_on_the_new_last(self, dialog):
        dialog.table.selectRow(2)
        dialog.delete_selected_rows()
        assert dialog.table.currentIndex().row() == 1

    def test_declining_the_prompt_keeps_the_rows(self, dialog, monkeypatch):
        monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
        dialog.table.selectAll()
        dialog.delete_selected_rows()
        assert len(dialog.app_state._all_labels_df) == 3


class TestStaleTable:
    def test_a_table_replaced_elsewhere_is_re_read_not_written(self, dialog):
        """Rows are addressed by position, so an outside rewrite must not be
        edited through at the positions this view was built from."""
        state = dialog.app_state
        dialog.table.selectRow(0)
        # Any label edit in the main window rebuilds the frame trial by trial.
        trial_2 = get_trial_from_tsv(state._all_labels_df, "2")
        state._all_labels_df = set_trial_in_tsv(state._all_labels_df, "2", trial_2)
        dialog.delete_selected_rows()
        assert len(state._all_labels_df) == 3
        assert dialog._is_current()  # reloaded onto the frame that is current now

    def test_refresh_if_stale_only_reloads_when_the_frame_moved(self, dialog):
        state = dialog.app_state
        dialog.table.selectRow(0)
        dialog.refresh_if_stale()
        assert dialog.table.selectionModel().hasSelection()  # nothing to do

        trial_2 = get_trial_from_tsv(state._all_labels_df, "2")
        state._all_labels_df = set_trial_in_tsv(state._all_labels_df, "2", trial_2)
        dialog.refresh_if_stale()
        assert dialog._is_current()


class TestKeys:
    """The shell binds Ctrl+A / Ctrl+C as *application* shortcuts, which fire
    before a focused table ever sees the key. The dialog takes them back by
    accepting the ShortcutOverride — without that, Ctrl+A autoscales the plots
    instead of selecting rows."""

    @pytest.mark.parametrize("key", [Qt.Key_A, Qt.Key_C])
    def test_the_table_takes_back_the_ctrl_keys(self, dialog, key):
        event = QKeyEvent(QEvent.ShortcutOverride, key, Qt.ControlModifier)
        QApplication.sendEvent(dialog.table, event)
        assert event.isAccepted()

    def test_a_key_the_table_does_not_own_is_left_alone(self, dialog):
        """Ctrl+S still saves, Ctrl+Z still undoes: only our keys are taken."""
        event = QKeyEvent(QEvent.ShortcutOverride, Qt.Key_S, Qt.ControlModifier)
        QApplication.sendEvent(dialog.table, event)
        assert not event.isAccepted()

    def test_ctrl_a_selects_every_visible_row_and_column(self, dialog):
        QTest.keyClick(dialog.table, Qt.Key_A, Qt.ControlModifier)
        selected = dialog.table.selectionModel().selectedIndexes()
        assert len(selected) == dialog._proxy.rowCount() * dialog._proxy.columnCount()
        assert len(dialog._selected_positions()) == 3

    def test_ctrl_a_leaves_filtered_out_rows_out(self, dialog):
        dialog._proxy.set_cat_filter(_col(dialog, "trial"), {"2"})
        QTest.keyClick(dialog.table, Qt.Key_A, Qt.ControlModifier)
        assert dialog._selected_positions() == [2]  # only the visible row

    def test_delete_key_deletes(self, dialog):
        dialog.table.selectRow(0)
        QTest.keyClick(dialog.table, Qt.Key_Delete)
        assert len(dialog.app_state._all_labels_df) == 2


def test_empty_table_opens(app_state, qtbot):
    app_state._all_labels_df = _labels().iloc[:0]
    dlg = LabelTableDialog(app_state, mappings=MAPPINGS)
    qtbot.addWidget(dlg)
    assert dlg._model.rowCount() == 0
