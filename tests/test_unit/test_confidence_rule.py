"""The confidence knob in the grid: preview touches only the tiles, Apply reaches the labels with undo.

What earns a test: a preview must change the entries a curve exists for and
no others, closing without Apply must put every value back, and Apply must
record one undo step per trial it touched and hand the new frame to
``replace_all_labels`` — the two calls that make it an ordinary label edit.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ethograph.gui.dialog_label_gridview import ConfidenceHistogramsDialog, ConfidenceRuleController, FrameEntry
from ethograph.labels.intervals import LABELING_AUTOMATED, LABELING_MANUAL


def _bump(t, c, h, w=0.03):
    return h * np.exp(-0.5 * ((t - c) / w) ** 2)


def _curves():
    t = np.arange(0, 4.0, 0.01)
    return {"1": (t, {31: _bump(t, 2.0, 0.9), 32: _bump(t, 2.0, 0.9) + _bump(t, 0.8, 0.85)})}


def _entry(trial, label, method, confidence):
    return FrameEntry(
        trial=trial,
        camera=None,
        label_id=label,
        name=str(label),
        event_type="point",
        boundary="point",
        t_rel=2.0,
        onset_s=2.0,
        offset_s=float("nan"),
        confidence=confidence,
        labeling_method=method,
    )


class _State:
    """The slice of app_state the controller reads and writes."""

    def __init__(self, df):
        self._all_labels_df = df
        self.nc_file_path = "unused.nc"
        self.recorded = []
        self.replaced = None
        self.grid_confidence_rule = "product"
        self.grid_confidence_alpha = 0.5
        self.grid_confidence_window_ms = 100.0

    def get_with_default(self, key):
        return getattr(self, key)

    def record_label_edit(self, description, trial=None):
        self.recorded.append(trial)

    def replace_all_labels(self, df):
        self.replaced = df
        self._all_labels_df = df


@pytest.fixture
def setup(qapp):
    df = pd.DataFrame(
        {
            "trial": [1, 1, 1],
            "labels": [31, 32, 31],
            "onset_s": [2.0, 2.0, 1.0],
            "offset_s": [np.nan] * 3,
            "confidence": [0.5, 0.5, 1.0],
            "labeling_method": [LABELING_AUTOMATED, LABELING_AUTOMATED, LABELING_MANUAL],
        }
    )
    state = _State(df)
    entries = [
        _entry(1, 31, LABELING_AUTOMATED, 0.5),
        _entry(1, 32, LABELING_AUTOMATED, 0.5),
        _entry(1, 31, LABELING_MANUAL, 1.0),
    ]
    meta = SimpleNamespace(app_state=state, labels_widget=None)
    controller = ConfidenceRuleController(meta, entries_fn=lambda: entries)
    controller._curves = _curves()  # no session file on disk in a unit test
    return state, entries, controller


class TestPreview:
    def test_preview_changes_only_automated_entries_with_a_curve(self, setup):
        state, entries, controller = setup
        controller._originals = {id(e): e.confidence for e in entries}
        controller.preview("ratio", 0.5, 100.0)
        assert entries[0].confidence == pytest.approx(1.0)
        assert entries[1].confidence < 0.1
        assert entries[2].confidence == 1.0  # manual, untouched
        assert state.replaced is None  # nothing reached the labels
        controller.revert()
        assert [e.confidence for e in entries] == [0.5, 0.5, 1.0]


class TestApply:
    def test_apply_records_an_undo_step_per_trial_and_replaces_the_frame(self, setup):
        state, entries, controller = setup
        controller._originals = {id(e): e.confidence for e in entries}
        controller.preview("ratio", 0.5, 100.0)
        controller.apply()
        assert state.recorded == [1]
        assert state.replaced is not None
        assert state.replaced.loc[0, "confidence"] == pytest.approx(1.0)
        assert state.replaced.loc[1, "confidence"] < 0.1
        assert state.replaced.loc[2, "confidence"] == 1.0
        assert (state.grid_confidence_rule, state.grid_confidence_window_ms) == ("ratio", 100.0)
        controller.revert()  # after apply the previewed values are the originals
        assert entries[0].confidence == pytest.approx(1.0)


class TestHistogramPopup:
    def test_the_rule_lives_in_the_histogram_and_previews_until_applied(self, setup):
        state, entries, controller = setup
        controller.begin()
        dialog = ConfidenceHistogramsDialog([], 0.0, rule_controller=controller)
        assert not dialog.alpha_slider.isEnabled()  # product: no slider
        assert "2 of 2" in dialog.coverage.text()  # both automated labels have a curve; the manual one is not counted
        dialog.rule_combo.setCurrentIndex(dialog.rule_combo.findData("ratio"))
        assert entries[0].confidence == pytest.approx(1.0) and entries[1].confidence < 0.1  # previewed
        assert state.replaced is None  # not applied
        dialog.close()  # closing without Apply puts the originals back
        assert [e.confidence for e in entries] == [0.5, 0.5, 1.0]

    def test_apply_confirms_and_closing_afterwards_keeps_it(self, setup):
        state, entries, controller = setup
        controller.begin()
        dialog = ConfidenceHistogramsDialog([], 0.0, rule_controller=controller)
        dialog.rule_combo.setCurrentIndex(dialog.rule_combo.findData("custom"))
        assert dialog.alpha_slider.isEnabled()
        dialog.alpha_slider.setValue(100)  # custom at alpha = 1 is ratio
        dialog.apply_btn.click()
        assert state.recorded == [1] and state.replaced.loc[0, "confidence"] == pytest.approx(1.0)
        dialog.close()
        assert entries[0].confidence == pytest.approx(1.0)  # kept

    def test_copy_puts_the_infer_lines_on_the_clipboard(self, setup, qapp):
        state, entries, controller = setup
        controller.begin()
        dialog = ConfidenceHistogramsDialog([], 0.0, rule_controller=controller)
        dialog.rule_combo.setCurrentIndex(dialog.rule_combo.findData("ratio"))
        dialog.window_spin.setValue(60.0)
        dialog.copy_btn.click()
        assert qapp.clipboard().text() == "infer:\n  confidence: ratio\n  focus_window_ms: 60\n"
        dialog.close()

    def test_without_curves_the_histogram_has_no_rule_panel(self, qapp):
        meta = SimpleNamespace(app_state=_State(pd.DataFrame()), labels_widget=None)
        controller = ConfidenceRuleController(meta, entries_fn=list)
        controller._curves = {}
        dialog = ConfidenceHistogramsDialog([], 0.0, rule_controller=controller)
        assert not hasattr(dialog, "rule_combo")
        dialog.close()
