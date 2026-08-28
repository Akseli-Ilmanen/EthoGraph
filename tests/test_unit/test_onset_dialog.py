"""Predict dialog wiring: the trials-table scope note and the review hand-off.

The model itself is covered by ``test_onset_model.py``; these tests only touch
the dialog, so no classifier is ever fitted here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_label_gridview import LabelGridViewDialog
from ethograph.gui.dialog_onset_model import (
    FeatureTree,
    LabelInputTree,
    PredictOnsetDialog,
    ensure_curve_run_chosen,
)
from ethograph.gui.widgets_curation import CurationPanel
from ethograph.io.catalog import XarrayLoader
from ethograph.labels import onset_curves
from ethograph.labels import onset_model as om
from ethograph.labels.label_inputs import LabelInput

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
    def __init__(self, mappings, app_state):
        self._mappings = mappings
        self.curation_panel = CurationPanel(app_state, self)

    def refresh_labels_shapes_layer(self):
        pass


class _CollapsibleStub:
    def __init__(self):
        self.expanded = False

    def expand(self, animate: bool = True) -> None:
        self.expanded = True


class _Meta:
    def __init__(self, app_state):
        self.app_state = app_state
        self.labels_widget = _LabelsStub(MAPPINGS, app_state)
        self.navigation_widget = None
        self.data_widget = None
        self.io_widget = None
        # Data / Labels / Nav, matching grid_section_container._SHORT_LABELS order.
        self.collapsible_widgets = [_CollapsibleStub(), _CollapsibleStub(), _CollapsibleStub()]


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
    dialog.close()


class TestTrialsScope:
    """The dialog has no trial filters of its own: the trials table decides."""

    def test_no_filter_controls(self, predict_dialog):
        assert not hasattr(predict_dialog, "_filters")
        assert not hasattr(predict_dialog, "_allowed_trials")

    def test_note_follows_the_trials_table(self, predict_dialog):
        predict_dialog.app_state.trials = [1, 2, 5]
        assert predict_dialog.trials_note.text().startswith("Runs over the 3 trial(s) the trials table")
        predict_dialog.app_state.trials = [1]
        assert predict_dialog.trials_note.text().startswith("Runs over the 1 trial(s)")


class TestReviewHandOff:
    def test_review_is_disabled_until_something_is_predicted(self, predict_dialog):
        assert not predict_dialog.review_btn.isEnabled()

    def test_review_drops_predictions_into_curation_scope_and_opens_labels_tab(self, predict_dialog):
        predict_dialog._reviewable = ([1], {"2", "4"})
        predict_dialog.review_btn.setEnabled(True)
        predict_dialog._review()

        curation_panel = predict_dialog.meta.labels_widget.curation_panel
        assert curation_panel.scope_area.ids() == [1]
        assert curation_panel.app_state.curation_label_ids == [1]
        assert curation_panel.app_state.curation_active is True
        assert predict_dialog.meta.collapsible_widgets[1].expanded is True

    def test_review_does_nothing_with_no_predictions(self, predict_dialog):
        predict_dialog._review()
        curation_panel = predict_dialog.meta.labels_widget.curation_panel
        assert curation_panel.scope_area.ids() == []
        assert curation_panel.app_state.curation_active is False
        assert predict_dialog.meta.collapsible_widgets[1].expanded is False


class _FakeButton:
    def __init__(self, role):
        self.role = role


class _FakeMessageBox:
    """Stands in for QMessageBox so a test never blocks on a real exec().

    Set ``picked_role`` on the class before use to say which button the
    "click" lands on.
    """

    AcceptRole = "accept"
    RejectRole = "reject"
    Question = "question"
    picked_role: str | None = None

    def __init__(self, parent=None):
        self._buttons: list[_FakeButton] = []

    def setIcon(self, *_a):
        pass

    def setWindowTitle(self, *_a):
        pass

    def setText(self, *_a):
        pass

    def addButton(self, _label, role):
        btn = _FakeButton(role)
        self._buttons.append(btn)
        return btn

    def exec(self):
        return 0

    def clickedButton(self):
        return next((b for b in self._buttons if b.role == self.picked_role), None)


class TestEnsureCurveRunChosen:
    """dialog_onset_model.ensure_curve_run_chosen: picked once, remembered.

    Frame-by-frame review (widgets_curation.CurationPanel) never asks itself
    — it only reads app_state.curve_run_path. This is where the ask happens,
    right after a predict run, and only when it actually needs to.
    """

    def _write_run(self, session, timestamp):
        onset_curves.write_curves(
            onset_curves.run_dir(session, timestamp) / onset_curves.CURVES_FILE,
            {"0": (np.linspace(0.0, 1.0, 5), {1: np.full(5, 0.5)})},
        )

    def test_no_session_path_is_a_noop(self, predict_dialog):
        predict_dialog.app_state.nc_file_path = ""
        ensure_curve_run_chosen(predict_dialog, predict_dialog.app_state)
        assert predict_dialog.app_state.curve_run_path is None

    def test_no_runs_is_a_noop(self, predict_dialog, tmp_path):
        predict_dialog.app_state.nc_file_path = str(tmp_path / "session.nc")
        ensure_curve_run_chosen(predict_dialog, predict_dialog.app_state)
        assert predict_dialog.app_state.curve_run_path is None

    def test_one_run_is_used_without_asking(self, predict_dialog, tmp_path, monkeypatch):
        session = tmp_path / "session.nc"
        predict_dialog.app_state.nc_file_path = str(session)
        self._write_run(session, "20260824_120000")
        monkeypatch.setattr(
            "ethograph.gui.dialog_onset_model.QMessageBox",
            lambda *a, **k: pytest.fail("must not ask with only one run"),
        )
        ensure_curve_run_chosen(predict_dialog, predict_dialog.app_state)
        assert predict_dialog.app_state.curve_run_path is not None
        assert Path(predict_dialog.app_state.curve_run_path).is_file()

    def test_an_existing_choice_is_left_alone(self, predict_dialog, tmp_path, monkeypatch):
        session = tmp_path / "session.nc"
        predict_dialog.app_state.nc_file_path = str(session)
        self._write_run(session, "20260824_120000")
        self._write_run(session, "20260825_090000")
        chosen = onset_curves.run_dirs(session)[0] / onset_curves.CURVES_FILE
        predict_dialog.app_state.curve_run_path = str(chosen)
        monkeypatch.setattr(
            "ethograph.gui.dialog_onset_model.QMessageBox",
            lambda *a, **k: pytest.fail("must not ask when curve_run_path is already valid"),
        )
        ensure_curve_run_chosen(predict_dialog, predict_dialog.app_state)
        assert predict_dialog.app_state.curve_run_path == str(chosen)

    def test_several_runs_prompts_and_remembers_the_pick(self, predict_dialog, tmp_path, monkeypatch):
        session = tmp_path / "session.nc"
        predict_dialog.app_state.nc_file_path = str(session)
        self._write_run(session, "20260824_120000")
        self._write_run(session, "20260825_090000")
        picked = onset_curves.run_dirs(session)[1] / onset_curves.CURVES_FILE
        _FakeMessageBox.picked_role = _FakeMessageBox.AcceptRole
        monkeypatch.setattr("ethograph.gui.dialog_onset_model.QMessageBox", _FakeMessageBox)
        monkeypatch.setattr("ethograph.gui.dialog_onset_model.browse_open_file", lambda *a, **k: str(picked))
        ensure_curve_run_chosen(predict_dialog, predict_dialog.app_state)
        assert predict_dialog.app_state.curve_run_path == str(picked)

    def test_declining_leaves_it_unset(self, predict_dialog, tmp_path, monkeypatch):
        session = tmp_path / "session.nc"
        predict_dialog.app_state.nc_file_path = str(session)
        self._write_run(session, "20260824_120000")
        self._write_run(session, "20260825_090000")
        _FakeMessageBox.picked_role = _FakeMessageBox.RejectRole
        monkeypatch.setattr("ethograph.gui.dialog_onset_model.QMessageBox", _FakeMessageBox)
        ensure_curve_run_chosen(predict_dialog, predict_dialog.app_state)
        assert predict_dialog.app_state.curve_run_path is None


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
        dialog = LabelGridViewDialog(_Meta(state), label_ids=[1], trials={"1", "3"})
        try:
            from ethograph.gui.dialog_label_gridview import build_frame_entries

            entries = build_frame_entries(labels, MAPPINGS, [1], [None], dialog._restrict_trials)
            assert [entry.trial for entry in entries] == [1, 3]
            assert [entry.confidence for entry in entries] == [0.3, 0.5]
        finally:
            dialog.close()

    def test_no_restriction_keeps_every_trial(self, qapp, tmp_path):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = pd.DataFrame()
        dialog = LabelGridViewDialog(_Meta(state))
        try:
            assert dialog._restrict_trials is None
        finally:
            dialog.close()


class _CatalogStub:
    def __init__(self, feature_dims: dict[str, dict[str, list[str]]]):
        self._feature_dims = feature_dims

    def feature_choices(self):
        return list(self._feature_dims)


class _LoaderStub:
    """Enough of a DataLoader for FeatureTree.populate_from_loader: a
    catalog listing features, and feature_dims(feature) -> {dim: [values]}."""

    def __init__(self, feature_dims: dict[str, dict[str, list[str]]]):
        self.catalog = _CatalogStub(feature_dims)
        self._feature_dims = feature_dims

    def feature_dims(self, feature):
        return self._feature_dims[feature]


class TestFeatureTree:
    """A dim with only one possible value is nothing to choose between, so it
    is never drawn as a row — it just travels along whenever its feature is
    checked."""

    def _tree(self, feature_dims):
        state = ObservableAppState()
        state.data_loader = _LoaderStub(feature_dims)
        tree = FeatureTree()
        tree.populate_from_loader(state)
        return tree

    def test_single_valued_dim_draws_no_row(self, qapp):
        tree = self._tree({"speed": {"individual": ["a"], "keypoint": ["nose", "tail"]}})
        item = tree.topLevelItem(0)
        assert item.text(0) == "speed"
        # Only "keypoint" (2 values) is a row; "individual" (1 value) is not.
        assert item.childCount() == 1
        assert item.child(0).text(0) == "keypoint"

    def test_feature_with_only_single_valued_dims_has_no_children(self, qapp):
        tree = self._tree({"heading": {"individual": ["a"]}})
        item = tree.topLevelItem(0)
        assert item.childCount() == 0

    def test_implicit_dim_is_included_when_the_feature_is_checked(self, qapp):
        tree = self._tree({"speed": {"individual": ["a"], "keypoint": ["nose", "tail"]}})
        item = tree.topLevelItem(0)
        keypoint_dim = item.child(0)
        # Ticking one value (not the whole feature) is enough — auto-tristate
        # bubbles the feature row to at least partially checked.
        keypoint_dim.child(0).setCheckState(0, Qt.Checked)  # "nose"
        assert tree.selected_features() == {"speed": {"individual": ["a"], "keypoint": ["nose"]}}

    def test_feature_of_only_implicit_dims_needs_no_further_ticking(self, qapp):
        tree = self._tree({"heading": {"individual": ["a"]}})
        tree.topLevelItem(0).setCheckState(0, Qt.Checked)
        assert tree.selected_features() == {"heading": {"individual": ["a"]}}

    def test_populate_from_config_also_hides_single_valued_dims(self, qapp):
        """Display-only: a frozen model's tree is disabled, selected_features()
        is never called on it — this just checks what gets drawn."""
        tree = FeatureTree()
        tree.populate_from_config(_frozen_config())
        item = tree.topLevelItem(0)
        assert item.childCount() == 1
        assert item.child(0).text(0) == "keypoint"


def _approach_input() -> LabelInput:
    return LabelInput(label=5, name="approach", event_type="state", individuals=["Freddy"])


def _frozen_config(derivatives=(), label_inputs=()):
    return om.OnsetModelConfig(
        name="frozen",
        targets={1: "approach"},
        features={"speed": {"individual": ["a"], "keypoint": ["nose", "tail"]}},
        derivatives=list(derivatives),
        label_inputs=list(label_inputs),
    )


class TestDerivativeTick:
    """Each feature row carries its own d/dt tick, drawn as a checkbox widget
    because an auto-tristate item with children reports its children's check
    state in every column — the item's own would never render."""

    def _tree(self, feature_dims):
        state = ObservableAppState()
        state.data_loader = _LoaderStub(feature_dims)
        tree = FeatureTree()
        tree.populate_from_loader(state)
        return tree

    def test_a_feature_with_dim_rows_still_shows_its_tick(self, qapp):
        tree = self._tree({"speed": {"keypoint": ["nose", "tail"]}})
        item = tree.topLevelItem(0)
        assert item.childCount() == 1
        assert tree.itemWidget(item, 1) is not None

    def test_ticked_only_when_the_feature_itself_is_ticked(self, qapp):
        tree = self._tree({"speed": {"keypoint": ["nose", "tail"]}})
        item = tree.topLevelItem(0)
        tree.itemWidget(item, 1).setChecked(True)
        # The feature is not part of the model at all yet.
        assert tree.selected_derivatives() == []
        item.setCheckState(0, Qt.Checked)
        assert tree.selected_derivatives() == ["speed"]

    def test_frozen_config_shows_the_derivatives_it_was_created_with(self, qapp):
        tree = FeatureTree()
        tree.populate_from_config(_frozen_config(derivatives=["speed"]))
        assert tree.itemWidget(tree.topLevelItem(0), 1).isChecked()


class TestIndividualGuard:
    """A label belongs to one (actor, receiver) pair and the overlay draws only
    the selected pair, so events written for somebody this session has never
    heard of are stored and never seen. The usual cause is a workflow copied
    between animals with its Predict onsets step's individual left behind.
    """

    def _capture(self, monkeypatch):
        from ethograph.gui import dialog_onset_model as mod

        messages: list[str] = []
        monkeypatch.setattr(mod, "notify", lambda msg, **kw: messages.append(msg))
        return mod, messages

    def test_an_individual_this_session_does_not_have_is_refused(self, predict_dialog, monkeypatch):
        mod, messages = self._capture(monkeypatch)
        assert predict_dialog.app_state.label_individuals() == ["default"]

        outcome = mod.predict_onsets(predict_dialog.meta, "m", individual="Poppy", min_confidence=0.0)

        assert outcome is None
        assert "Poppy" in messages[0] and "default" in messages[0]
        assert predict_dialog.app_state._all_labels_df is None

    def test_a_known_individual_gets_past_the_guard(self, predict_dialog, monkeypatch):
        mod, messages = self._capture(monkeypatch)

        mod.predict_onsets(predict_dialog.meta, "m", individual="default", min_confidence=0.0)

        assert not any("never be drawn" in msg for msg in messages)


class _LabelStateStub:
    """Enough app_state for LabelInputTree.populate: the mapping and the
    individuals the labels name."""

    def __init__(self, mappings, individuals):
        self._label_mappings = mappings
        self._individuals = individuals

    def label_individuals(self):
        return self._individuals


INPUT_MAPPINGS = {
    1: {"name": "approach", "event_type": "state"},
    2: {"name": "cue", "event_type": "point"},
}


class TestLabelInputTree:
    """Existing labels are ticked like features — and a class the model
    predicts is greyed out, because it cannot be its own input."""

    def _tree(self, individuals=("Freddy",)):
        tree = LabelInputTree()
        tree.populate(_LabelStateStub(INPUT_MAPPINGS, list(individuals)))
        return tree

    def test_one_individual_draws_no_children(self, qapp):
        tree = self._tree()
        assert tree.topLevelItemCount() == 2
        assert tree.topLevelItem(0).childCount() == 0

    def test_several_individuals_become_rows(self, qapp):
        tree = self._tree(["Freddy", "Ivy"])
        item = tree.topLevelItem(0)
        assert [item.child(i).text(0) for i in range(item.childCount())] == ["Freddy", "Ivy"]

    def test_the_class_row_is_that_class_all_toggle(self, qapp):
        tree = self._tree(["Freddy", "Ivy"])
        tree.topLevelItem(0).setCheckState(0, Qt.Checked)
        assert tree.selected_inputs()[0].individuals == ["Freddy", "Ivy"]

    def test_one_individual_ticked_is_one_column(self, qapp):
        tree = self._tree(["Freddy", "Ivy"])
        tree.topLevelItem(0).child(1).setCheckState(0, Qt.Checked)
        assert tree.selected_inputs()[0].individuals == ["Ivy"]

    def test_the_event_type_comes_from_the_mapping(self, qapp):
        tree = self._tree()
        tree.set_all_checked(True)
        assert [(i.label, i.event_type) for i in tree.selected_inputs()] == [(1, "state"), (2, "point")]

    def test_a_target_is_unticked_and_greyed_out(self, qapp):
        tree = self._tree()
        tree.set_all_checked(True)
        tree.set_excluded({1})
        assert tree.topLevelItem(0).isDisabled()
        assert tree.topLevelItem(0).checkState(0) == Qt.Unchecked
        assert [i.label for i in tree.selected_inputs()] == [2]

    def test_select_all_leaves_the_targets_alone(self, qapp):
        tree = self._tree()
        tree.set_excluded({1})
        tree.set_all_checked(True)
        assert [i.label for i in tree.selected_inputs()] == [2]

    def test_a_frozen_config_shows_the_inputs_it_was_created_with(self, qapp):
        tree = LabelInputTree()
        tree.populate_from_config(_frozen_config(label_inputs=[_approach_input()]))
        assert tree.topLevelItemCount() == 1
        assert tree.selected_inputs() == [_approach_input()]
