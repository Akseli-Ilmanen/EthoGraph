"""Ctrl+Z takes back the last label placed, moved or deleted.

The snapshot is per trial, not per table: a session holds every trial's labels
in one DataFrame, and an undo stack that copied all of them per click would
cost with the size of the dataset instead of the size of the edit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ethograph.labels.intervals import INTERVAL_COLUMNS
from ethograph.labels.tsv_store import (
    LabelHistory,
    get_trial_from_tsv,
    get_trial_meta,
    set_trial_in_tsv,
)


def _all_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": [1, 1, 2],
            "individual": ["crow1", "crow1", "crow1"],
            "individual_rec": ["", "", ""],
            "labels": [1, 2, 1],
            "onset_s": [0.4, 0.9, 0.2],
            "offset_s": [0.8, 1.0, 0.5],
            "event_type": ["state", "point", "state"],
            "confidence": [1.0, 1.0, 1.0],
            "human_verified": [0, 0, 0],
            "changepoint_corrected": [0, 0, 0],
            "prediction_source": ["", "", ""],
            "n_samples": [0, 0, 0],
        }
    )


def _place_point(all_df: pd.DataFrame, trial, onset: float) -> pd.DataFrame:
    trial_df = get_trial_from_tsv(all_df, trial)
    row = {
        "onset_s": onset,
        "offset_s": float("nan"),
        "labels": 3,
        "individual": "crow1",
        "individual_rec": "",
        "event_type": "point",
        "confidence": 1.0,
    }
    trial_df = pd.concat([trial_df[INTERVAL_COLUMNS], pd.DataFrame([row])], ignore_index=True)
    return set_trial_in_tsv(all_df, trial, trial_df)


class TestLabelHistory:
    def test_undo_takes_back_a_placed_point(self):
        history = LabelHistory()
        before = _all_labels()

        history.record(before, 1, "place point")
        after = _place_point(before, 1, 1.5)
        assert len(get_trial_from_tsv(after, 1)) == 3

        restored, edit = history.undo(after)
        assert edit.description == "place point"
        assert edit.trial == 1
        assert sorted(get_trial_from_tsv(restored, 1)["onset_s"]) == [0.4, 0.9]

    def test_undo_takes_back_a_deleted_label(self):
        history = LabelHistory()
        before = _all_labels()

        history.record(before, 1, "delete label")
        after = set_trial_in_tsv(before, 1, get_trial_from_tsv(before, 1).iloc[:1])
        assert len(get_trial_from_tsv(after, 1)) == 1

        restored, _ = history.undo(after)
        assert sorted(get_trial_from_tsv(restored, 1)["onset_s"]) == [0.4, 0.9]

    def test_undo_takes_back_a_moved_point(self):
        history = LabelHistory()
        before = _all_labels()

        history.record(before, 1, "move point")
        moved = get_trial_from_tsv(before, 1)
        moved.loc[1, "onset_s"] = 5.0
        after = set_trial_in_tsv(before, 1, moved)

        restored, _ = history.undo(after)
        assert sorted(get_trial_from_tsv(restored, 1)["onset_s"]) == [0.4, 0.9]

    def test_other_trials_are_untouched_by_an_undo(self):
        history = LabelHistory()
        before = _all_labels()

        history.record(before, 1, "place point")
        after = _place_point(_place_point(before, 1, 1.5), 2, 2.5)

        restored, _ = history.undo(after)
        # Trial 2's later edit survives — the snapshot only ever held trial 1.
        assert sorted(restored[restored["trial"] == 2]["onset_s"]) == [0.2, 2.5]

    def test_a_snapshot_holds_only_the_edited_trials_rows(self):
        history = LabelHistory()
        history.record(_all_labels(), 1, "place point")

        assert len(history.peek().rows) == 2  # not 3: trial 2 is not copied

    def test_undo_into_a_trial_that_had_no_labels(self):
        history = LabelHistory()
        before = _all_labels()

        history.record(before, 3, "place point")
        after = _place_point(before, 3, 0.1)
        assert len(get_trial_from_tsv(after, 3)) == 1

        restored, _ = history.undo(after)
        assert restored[restored["trial"] == 3].empty

    def test_undo_restores_the_trial_flags_the_edit_set(self):
        history = LabelHistory()
        before = _all_labels()

        history.record(before, 1, "place point")
        after = _place_point(before, 1, 1.5)
        after.loc[after["trial"] == 1, "human_verified"] = 1

        restored, _ = history.undo(after)
        assert get_trial_meta(restored, 1)["human_verified"] == 0

    def test_edits_undo_newest_first(self):
        history = LabelHistory()
        df = _all_labels()

        history.record(df, 1, "first")
        df = _place_point(df, 1, 1.5)
        history.record(df, 1, "second")
        df = _place_point(df, 1, 2.5)

        df, edit = history.undo(df)
        assert edit.description == "second"
        assert sorted(get_trial_from_tsv(df, 1)["onset_s"]) == [0.4, 0.9, 1.5]

        df, edit = history.undo(df)
        assert edit.description == "first"
        assert sorted(get_trial_from_tsv(df, 1)["onset_s"]) == [0.4, 0.9]

    def test_an_empty_history_undoes_nothing(self):
        assert LabelHistory().undo(_all_labels()) is None

    def test_depth_is_bounded(self):
        history = LabelHistory(max_depth=3)
        for i in range(10):
            history.record(_all_labels(), 1, f"edit {i}")

        assert len(history) == 3
        assert history.peek().description == "edit 9"


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class TestAppStateUndo:
    @pytest.fixture()
    def state(self, qapp, tmp_path):
        from ethograph.gui.app_state import ObservableAppState

        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = _all_labels()
        state.trials = [1, 2]
        state.trials_sel = 1
        state.label_intervals = state.get_trial_intervals(1)
        return state

    def test_undo_restores_the_current_trials_view(self, state):
        state.record_label_edit("place point")
        state.set_trial_intervals(1, _place_point(state._all_labels_df, 1, 1.5)[lambda d: d["trial"] == 1])
        state.label_intervals = state.get_trial_intervals(1)
        assert len(state.label_intervals) == 3

        edit = state.undo_label_edit()

        assert edit.description == "place point"
        assert len(state.label_intervals) == 2
        assert len(state.get_trial_intervals(1)) == 2

    def test_undo_reports_nothing_left(self, state):
        assert state.can_undo_labels() is False
        assert state.undo_label_edit() is None

    def test_clearing_the_history_drops_the_stack(self, state):
        state.record_label_edit("place point")
        assert state.can_undo_labels() is True

        state.clear_label_history()

        assert state.can_undo_labels() is False
        assert state.undo_label_edit() is None
