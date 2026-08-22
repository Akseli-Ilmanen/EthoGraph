"""Predict dialog wiring: combinatorial metadata filters and the review hand-off.

The model itself is covered by ``test_onset_model.py``; these tests only touch
the dialog, so no classifier is ever fitted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_label_frames import LabelFramesDialog
from ethograph.gui.dialog_onset_model import PredictOnsetDialog
from ethograph.io.catalog import XarrayLoader
from ethograph.labels import onset_model as om

MAPPINGS = {
    1: {"name": "approach", "event_type": "point", "color": (1.0, 0.0, 0.0)},
    2: {"name": "touch", "event_type": "point", "color": (0.0, 1.0, 0.0)},
}

METADATA = pd.DataFrame(
    {
        "trial": [1, 2, 3, 4, 5, 6],
        "genotype": ["wt", "wt", "ko", "ko", "wt", "ko"],
        "stimulus": ["tone", "light", "tone", "light", "tone", "tone"],
    }
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path / ".ethograph"))


class _LabelsStub:
    def __init__(self, mappings):
        self._mappings = mappings

    def refresh_labels_shapes_layer(self):
        pass


class _Meta:
    def __init__(self, app_state):
        self.app_state = app_state
        self.labels_widget = _LabelsStub(MAPPINGS)
        self.navigation_widget = None
        self.data_widget = None
        self.io_widget = None


@pytest.fixture()
def predict_dialog(qapp, tmp_path):
    t = np.arange(0.0, 9.0, 0.02)
    ds = xr.Dataset(
        {"signal": (("time", "chan"), np.column_stack([np.sin(t), np.cos(t)]))},
        coords={"time": t, "chan": ["a", "b"]},
    )
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state.data_loader = XarrayLoader(ds)
    state.metadata_df = METADATA
    om.save_config(om.OnsetModelConfig(name="m", targets={1: "approach"}, features={"signal": {}}))
    dialog = PredictOnsetDialog(_Meta(state))
    yield dialog
    if dialog._review_dialog is not None:
        dialog._review_dialog.close()
    dialog.close()


class TestCombinatorialFilters:
    def test_every_condition_column_gets_a_button(self, predict_dialog):
        assert set(predict_dialog._filter_buttons) == {"genotype", "stimulus"}

    def test_no_filters_predicts_everything(self, predict_dialog):
        assert predict_dialog._allowed_trials() is None

    def test_columns_combine(self, predict_dialog):
        """Two filters intersect — wild-type *and* tone, not either."""
        predict_dialog._filters["genotype"] = {"wt"}
        assert predict_dialog._allowed_trials() == {"1", "2", "5"}
        predict_dialog._filters["stimulus"] = {"tone"}
        assert predict_dialog._allowed_trials() == {"1", "5"}

    def test_a_cleared_column_means_any(self, predict_dialog):
        predict_dialog._filters["genotype"] = set()
        predict_dialog._filters["stimulus"] = {"tone"}
        assert predict_dialog._allowed_trials() == {"1", "3", "5", "6"}

    def test_summary_counts_the_surviving_trials(self, predict_dialog):
        predict_dialog._filters["genotype"] = {"ko"}
        predict_dialog._filters["stimulus"] = {"tone"}
        predict_dialog._refresh_filter_summary()
        assert predict_dialog.filter_summary.text().startswith("2 trials match")

    def test_summary_is_blank_without_filters(self, predict_dialog):
        predict_dialog._refresh_filter_summary()
        assert predict_dialog.filter_summary.text() == ""


class TestReviewHandOff:
    def test_review_is_disabled_until_something_is_predicted(self, predict_dialog):
        assert not predict_dialog.review_btn.isEnabled()

    def test_review_opens_the_grid_on_what_was_predicted(self, predict_dialog):
        predict_dialog.app_state._all_labels_df = pd.DataFrame(
            {
                "trial": [2, 4],
                "labels": [1, 1],
                "onset_s": [0.5, 1.5],
                "offset_s": [np.nan, np.nan],
                "individual": ["a", "a"],
                "individual_rec": ["", ""],
                "confidence": [0.4, 0.8],
            }
        )
        predict_dialog._reviewable = ([1], {"2", "4"})
        predict_dialog.review_btn.setEnabled(True)
        predict_dialog._review()

        dialog = predict_dialog._review_dialog
        assert isinstance(dialog, LabelFramesDialog)
        ticked = [
            dialog.label_list.item(i).data(Qt.UserRole)
            for i in range(dialog.label_list.count())
            if dialog.label_list.item(i).checkState() == Qt.Checked
        ]
        assert ticked == [1]  # only the class that was predicted
        assert dialog._restrict_trials == {"2", "4"}

    def test_review_does_nothing_with_no_predictions(self, predict_dialog):
        predict_dialog._review()
        assert predict_dialog._review_dialog is None


class TestRestrictedGrid:
    def test_restriction_narrows_the_entries(self, qapp, tmp_path):
        """A handed-over trial set narrows the grid on top of the filters."""
        labels = pd.DataFrame(
            {
                "trial": [1, 2, 3],
                "labels": [1, 1, 1],
                "onset_s": [0.5, 1.5, 2.5],
                "offset_s": [np.nan, np.nan, np.nan],
                "individual": ["a", "a", "a"],
                "individual_rec": ["", "", ""],
                "confidence": [0.3, 0.9, 0.5],
            }
        )
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels
        state.metadata_df = pd.DataFrame({"trial": [1, 2, 3], "genotype": ["wt", "ko", "wt"]})
        dialog = LabelFramesDialog(_Meta(state), label_ids=[1], trials={"1", "3"})
        try:
            from ethograph.gui.dialog_label_frames import build_frame_entries

            entries = build_frame_entries(labels, MAPPINGS, [1], [None], dialog._restrict_trials)
            assert [entry.trial for entry in entries] == [1, 3]
            assert [entry.confidence for entry in entries] == [0.3, 0.5]
        finally:
            dialog.close()

    def test_no_restriction_keeps_every_trial(self, qapp, tmp_path):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = pd.DataFrame()
        dialog = LabelFramesDialog(_Meta(state))
        try:
            assert dialog._restrict_trials is None
        finally:
            dialog.close()
