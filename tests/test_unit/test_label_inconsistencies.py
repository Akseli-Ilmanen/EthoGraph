"""Find label inconsistencies: the dialog and its filter on the trials table."""

from __future__ import annotations

import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication, QWidget

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_label_inconsistencies import LabelInconsistencyDialog
from ethograph.gui.widget_trials import TrialsWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


MAPPINGS = {
    1: {"name": "approach", "event_type": "point"},
    2: {"name": "peck", "event_type": "point"},
}


class _LabelsStub(QWidget):
    def __init__(self):
        super().__init__()
        self._mappings = MAPPINGS


class _Meta:
    def __init__(self, app_state, trials_widget, labels_widget):
        self.app_state = app_state
        self.trials_widget = trials_widget
        self.labels_widget = labels_widget


def _labels_df() -> pd.DataFrame:
    """trial 1: 1 then 2 · trial 2: only 1 · trial 3: 2 then 1 · trial 4: nothing."""
    rows = [
        ("1", 1, 0.0),
        ("1", 2, 1.0),
        ("2", 1, 0.0),
        ("3", 2, 0.0),
        ("3", 1, 1.0),
    ]
    return pd.DataFrame(
        [{"trial": t, "labels": lab, "onset_s": on, "individual": "a"} for t, lab, on in rows],
    )


@pytest.fixture()
def dialog(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state._all_labels_df = _labels_df()
    metadata = pd.DataFrame({"trial": ["1", "2", "3", "4"], "genotype": ["wt", "wt", "ko", "ko"]})
    state.trials_sel = "1"
    state.ready = True
    trials = TrialsWidget(state)
    trials.setup(metadata)
    labels = _LabelsStub()
    dlg = LabelInconsistencyDialog(_Meta(state, trials, labels))
    yield dlg
    dlg.close()
    trials.close()
    labels.close()


def _ask(dialog, pattern, mode, invert=False):
    dialog.pattern_edit.setText(pattern)
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(mode))
    dialog.invert_check.setChecked(invert)
    return dialog.matching_trials()


class TestQuestions:
    def test_uncoupled_finds_the_lonely_label(self, dialog):
        """The headline case: one event without its partner."""
        assert _ask(dialog, "1-2", "partial") == {"2"}

    def test_order_is_not_mere_presence(self, dialog):
        assert _ask(dialog, "1-2", "present") == {"1", "3"}
        assert _ask(dialog, "1-2", "order") == {"1"}

    def test_invert_includes_the_trials_with_no_labels_at_all(self, dialog):
        """Trial 4 carries nothing, and "missing the sequence" must include it."""
        assert _ask(dialog, "1-2", "order", invert=True) == {"2", "3", "4"}

    def test_no_pattern_asks_nothing(self, dialog):
        assert _ask(dialog, "", "present") is None
        assert dialog.apply_btn.isEnabled() is False


class TestFilter:
    def test_applying_narrows_the_trials_table(self, dialog):
        _ask(dialog, "1-2", "partial")
        dialog._apply()
        assert dialog.app_state.trials == ["2"]

    def test_it_stacks_on_the_column_filters(self, dialog):
        """A label question about wild-types must not drop the genotype filter."""
        trials = dialog.meta.trials_widget
        trials.apply_column_filters([{"column": "genotype", "values": ["wt"]}])
        assert dialog.app_state.trials == ["1", "2"]
        _ask(dialog, "1-2", "present")  # trials 1 and 3 carry both
        dialog._apply()
        assert dialog.app_state.trials == ["1"]  # 3 is ko, filtered out by metadata

    def test_clearing_leaves_the_column_filters_alone(self, dialog):
        trials = dialog.meta.trials_widget
        trials.apply_column_filters([{"column": "genotype", "values": ["wt"]}])
        _ask(dialog, "1-2", "present")
        dialog._apply()
        dialog._clear()
        assert trials.label_filter_active() is False
        assert dialog.app_state.trials == ["1", "2"]  # still wild-type only

    def test_an_empty_answer_cannot_be_applied(self, dialog):
        """The table never shows nothing, so refuse rather than mislead."""
        assert _ask(dialog, "9", "present") == set()
        assert dialog.apply_btn.isEnabled() is False
