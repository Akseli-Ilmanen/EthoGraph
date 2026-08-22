"""Frame-by-frame label refinement: seed queue order + boundary commits."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QWidget

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_refine import RefineLabelsDialog, export_refine_log, targets_from_seeds
from ethograph.gui.widgets_navigation import NavigationWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeVideo:
    """Minimal display-clock video: 50 fps, trial basis."""

    fps = 50.0

    def time_to_frame(self, t: float, round_nearest: bool = False) -> int:
        return int(round(t * self.fps)) if round_nearest else int(t * self.fps)

    def frame_to_time(self, frame: int) -> float:
        return frame / self.fps

    def seek_to_frame(self, frame: int) -> None:
        self.sought = frame


class _Meta:
    def __init__(self, app_state, nav, labels_widget):
        self.app_state = app_state
        self.navigation_widget = nav
        self.labels_widget = labels_widget
        self.data_widget = None
        self.io_widget = None


class _LabelsStub(QWidget):
    def __init__(self, mappings):
        super().__init__()
        self._mappings = mappings

    def refresh_labels_shapes_layer(self):
        pass


MAPPINGS = {
    4: {"name": "peck", "color": (1.0, 0.0, 0.0), "event_type": "point"},
    6: {"name": "hop", "color": (0.0, 1.0, 0.0), "event_type": "point"},
    8: {"name": "mount", "color": (0.0, 0.0, 1.0), "event_type": "state"},
}


def _labels_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset_s": [1.0, 2.0, 2.5, 0.5],
            "offset_s": [np.nan, 3.0, np.nan, np.nan],
            "labels": [4, 8, 6, 4],
            "individual": ["a", "a", "a", "a"],
            "individual_rec": ["", "", "", ""],
            "event_type": ["point", "state", "point", "point"],
            "trial": ["0", "0", "0", "1"],
        }
    )


@pytest.fixture()
def dialog(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state._all_labels_df = _labels_df()
    state.video = _FakeVideo()
    nav = NavigationWidget(QWidget(), state)
    dlg = RefineLabelsDialog(_Meta(state, nav, _LabelsStub(MAPPINGS)))
    yield dlg
    dlg.close()
    nav.close()


def _check_all_labels(dlg: RefineLabelsDialog):
    for i in range(dlg.label_list.count()):
        dlg.label_list.item(i).setCheckState(Qt.Checked)


def test_queue_visits_each_trial_once_in_time_order(dialog):
    """Targets are sorted (trial, onset); a state event yields start then end."""
    _check_all_labels(dialog)
    queue = dialog._build_queue(dialog._selected_label_ids())
    got = [(t.inst["trial"], t.inst["labels"], t.field) for t in queue]
    assert got == [
        ("0", 4, "point"),
        ("0", 8, "start"),
        ("0", 8, "end"),
        ("0", 6, "point"),
        ("1", 4, "point"),
    ]


def test_confirm_moves_start_and_sibling_end_target_follows(dialog):
    """Confirming a start writes the on-screen frame's time; the end target of
    the same row (shared inst) sees the updated onset for its row lookup."""
    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()
    assert dialog.refine_group.isVisible() or dialog._targets  # session running

    # Advance to the state event's START target (index 1).
    dialog._idx = 1
    dialog._jump_current()
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")

    # Nudge 3 frames right of the 2.0 s seed (50 fps → 2.06 s) and confirm.
    state.current_frame = _FakeVideo().time_to_frame(2.0, round_nearest=True) + 3
    dialog._confirm()

    df = state._all_labels_df
    row = df[(df["trial"] == "0") & (df["labels"] == 8)].iloc[0]
    assert row["onset_s"] == pytest.approx(2.06)
    assert row["offset_s"] == pytest.approx(3.0)
    # The jump is deferred so the user sees the label land: still on START,
    # a second Enter during the pause is ignored, then the timer advances to
    # the END target of the SAME (updated) instance.
    assert dialog._targets[dialog._idx].field == "start"
    assert dialog._advance_pending
    dialog._confirm()  # Enter mashed during the pause — must not double-commit
    assert dialog._targets[dialog._idx].field == "start"
    dialog._advance_after_pause()
    target = dialog._targets[dialog._idx]
    assert target.field == "end"
    assert target.inst["onset_s"] == pytest.approx(2.06)
    dialog._stop()


def test_confirm_refuses_start_past_end(dialog):
    """A start nudged beyond the end is rejected and nothing advances."""
    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()
    dialog._idx = 1
    dialog._jump_current()
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")

    state.current_frame = _FakeVideo().time_to_frame(3.5, round_nearest=True)
    dialog._confirm()

    df = state._all_labels_df
    row = df[(df["trial"] == "0") & (df["labels"] == 8)].iloc[0]
    assert row["onset_s"] == pytest.approx(2.0)
    assert dialog._targets[dialog._idx].field == "start"  # did not advance
    dialog._stop()


def test_view_window_is_seed_centred_and_decoupled_from_padding(dialog):
    """The refine view is ``refine_window_s`` seconds centred on the seed —
    the navigation Before/After spinners play no part."""
    state = dialog.app_state
    state.refine_window_s = 0.2
    dialog.nav.before_spin.setValue(5.0)  # must not leak into the refine view
    view = dialog._view_rel(1.0)
    assert view.start_s == pytest.approx(0.9)
    assert view.end_s == pytest.approx(1.1)

    # The published restriction spans view ∪ label, so the viewport and the
    # loaders both cover the seed window even when it leaves the label.
    _check_all_labels(dialog)
    dialog._start()
    dialog._idx = 1  # state label 8: 2.0–3.0 s, start seed
    dialog._jump_current()
    rw = state.restrict_window
    assert rw.core_range.start_s == pytest.approx(1.9)
    assert rw.core_range.end_s == pytest.approx(3.0)
    dialog._stop()


def test_enter_shortcut_lives_with_the_session(dialog):
    """Return/Enter are application-wide only while refining, so Confirm works
    after clicking around the main window — and never outside a session."""
    _check_all_labels(dialog)
    assert dialog._enter_shortcuts == []
    # No default/autoDefault buttons: a spinbox ignores Return after committing
    # its edit, and Qt would hand it to the default button — typing a view
    # window then pressing Enter confirmed a boundary as a side effect.
    assert not dialog.confirm_btn.isDefault()
    assert not dialog.confirm_btn.autoDefault()
    dialog._start()
    assert len(dialog._enter_shortcuts) == 2
    assert all(sc.context() == Qt.ApplicationShortcut for sc in dialog._enter_shortcuts)
    dialog._stop()
    assert dialog._enter_shortcuts == []


def test_stop_during_pause_cancels_the_queued_jump(dialog):
    """Stopping between a commit and its delayed jump must not navigate."""
    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")
    state.current_frame = _FakeVideo().time_to_frame(1.0, round_nearest=True)
    dialog._confirm()
    assert dialog._advance_pending
    dialog._stop()
    idx_before = dialog._idx
    dialog._advance_after_pause()  # the timer fires after Stop
    assert dialog._idx == idx_before


def _refine_state_start(dialog, new_time: float):
    """Drive the session to the state event's START seed and confirm at *new_time*."""
    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()
    dialog._idx = 1
    dialog._jump_current()
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")
    state.current_frame = _FakeVideo().time_to_frame(new_time, round_nearest=True)
    dialog._confirm()


def test_refine_log_chains_repeat_refinements(dialog):
    """Every commit lands in refine_log; re-refining the same boundary updates
    the record (orig keeps the very first values) instead of duplicating."""
    state = dialog.app_state
    _refine_state_start(dialog, 2.06)
    dialog._advance_after_pause()  # → END target of the same instance

    log = state.refine_log
    assert len(log) == 1
    rec = log[0]
    assert rec["trial"] == "0" and rec["labels"] == 8 and rec["fields"] == ["start"]
    assert rec["orig_onset_s"] == pytest.approx(2.0)
    assert rec["new_onset_s"] == pytest.approx(2.06)

    # Refine the END of the same instance: old values match the record's new
    # values, so it chains into the same record.
    state.current_frame = _FakeVideo().time_to_frame(2.9, round_nearest=True)
    dialog._confirm()
    log = state.refine_log
    assert len(log) == 1
    rec = log[0]
    assert rec["fields"] == ["start", "end"]
    assert rec["orig_onset_s"] == pytest.approx(2.0)
    assert rec["orig_offset_s"] == pytest.approx(3.0)
    assert rec["new_onset_s"] == pytest.approx(2.06)
    assert rec["new_offset_s"] == pytest.approx(2.9)
    dialog._stop()


def test_resume_returns_to_the_last_seed(qapp, tmp_path):
    """Stop mid-queue, reopen the dialog, Resume → same seed, same filters."""
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state._all_labels_df = _labels_df()
    state.video = _FakeVideo()
    nav = NavigationWidget(QWidget(), state)
    meta = _Meta(state, nav, _LabelsStub(MAPPINGS))

    dlg1 = RefineLabelsDialog(meta)
    _refine_state_start(dlg1, 2.06)
    dlg1._advance_after_pause()  # standing on END of label 8, trial "0"
    dlg1._stop()
    dlg1.close()

    dlg2 = RefineLabelsDialog(meta)  # fresh dialog, as after reopening
    assert dlg2.resume_btn.isEnabled()
    dlg2._resume()
    assert dlg2._session_active
    target = dlg2._targets[dlg2._idx]
    assert str(target.inst["trial"]) == "0"
    assert target.inst["labels"] == 8
    assert target.field == "end"
    assert target.inst["onset_s"] == pytest.approx(2.06)  # the refined onset
    dlg2._stop()
    dlg2.close()
    nav.close()


def test_export_writes_pre_and_post_tsvs(dialog, tmp_path):
    """Export produces {base}_prerefined.tsv / _postrefined.tsv, same row order."""
    _refine_state_start(dialog, 2.06)
    dialog._stop()

    pre_path, post_path = export_refine_log(dialog.app_state.refine_log, tmp_path / "mylabels_refined.tsv")
    assert pre_path.name == "mylabels_refined_prerefined.tsv"
    assert post_path.name == "mylabels_refined_postrefined.tsv"
    pre = pd.read_csv(pre_path, sep="\t")
    post = pd.read_csv(post_path, sep="\t")
    assert len(pre) == len(post) == 1
    assert pre.loc[0, "onset_s"] == pytest.approx(2.0)
    assert post.loc[0, "onset_s"] == pytest.approx(2.06)
    assert pre.loc[0, "labels"] == 8
    assert pre.loc[0, "name"] == "mount"


def test_normal_trial_navigation_pulls_the_session_along(dialog):
    """Navigating trials the ordinary way jumps the session to that trial's
    first seed; a trial change caused by our own jump is ignored."""
    from qtpy.QtTest import QTest

    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()
    assert str(dialog._targets[dialog._idx].inst["trial"]) == "0"

    # Our own jumps must not recurse into the follow handler.
    dialog._jumping = True
    dialog._on_trial_changed()
    dialog._jumping = False

    state.trials_sel = "1"  # the user's normal navigation
    dialog._on_trial_changed()
    QTest.qWait(50)  # the follow-jump is deferred one event-loop tick
    target = dialog._targets[dialog._idx]
    assert str(target.inst["trial"]) == "1"
    assert target.inst["labels"] == 4 and target.field == "point"
    dialog._stop()


def test_unlocking_frees_navigation_and_relocking_snaps_back(dialog, monkeypatch):
    """ "Locked around initial label" off → jumps widen the restriction to the
    normal navigation scope (free pan/zoom); the spinner stops yanking the
    view; ticking it again snaps back to the seed window."""
    calls = {"scope": 0}
    monkeypatch.setattr(dialog.nav, "_apply_slider_scope", lambda: calls.__setitem__("scope", calls["scope"] + 1))

    _check_all_labels(dialog)
    dialog._start()
    assert dialog.lock_checkbox.isChecked()  # locked is the default
    assert calls["scope"] == 0  # locked jumps never touch the nav scope

    dialog.lock_checkbox.setChecked(False)
    assert calls["scope"] == 1  # unlocking frees the restriction immediately
    dialog._advance(+1)
    assert calls["scope"] == 2  # every unlocked jump re-frees it

    seen = []
    monkeypatch.setattr(dialog.nav, "set_view_range", lambda *a: seen.append(a))
    dialog.window_spin.setValue(1.0)  # spinner must not yank a free view
    assert seen == []

    dialog.lock_checkbox.setChecked(True)  # relock snaps back to the seed
    assert len(seen) == 0  # (relock jumps via jump_to_label_instance, not set_view_range)
    assert str(dialog._targets[dialog._idx].inst["trial"]) == "0"
    dialog._stop()


def test_history_filters_scope_the_visible_and_exported_rows(dialog):
    """Funnel filters on Trial/Label/… reduce the table AND what exports."""
    from ethograph.gui.dialog_refine import RefineHistoryDialog

    state = dialog.app_state
    state.refine_log = [
        {
            "trial": "0",
            "labels": 8,
            "name": "mount",
            "individual": "a",
            "individual_rec": "",
            "event_type": "state",
            "orig_onset_s": 2.0,
            "orig_offset_s": 3.0,
            "new_onset_s": 2.06,
            "new_offset_s": 3.0,
            "fields": ["start"],
            "time": "t",
        },
        {
            "trial": "1",
            "labels": 4,
            "name": "peck",
            "individual": "a",
            "individual_rec": "",
            "event_type": "point",
            "orig_onset_s": 0.5,
            "orig_offset_s": None,
            "new_onset_s": 0.46,
            "new_offset_s": None,
            "fields": ["point"],
            "time": "t",
        },
    ]
    hist = RefineHistoryDialog(state)
    assert hist._proxy.rowCount() == 2
    hist._proxy.set_cat_filter(0, {"1"})  # Trial column
    assert hist._proxy.rowCount() == 1
    visible = hist._visible_log()
    assert len(visible) == 1 and visible[0]["trial"] == "1"
    hist.close()


def test_confirm_moves_point_event(dialog):
    """A point event's onset moves to the confirmed frame; NaN offset stays NaN."""
    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()  # starts at index 0: point label 4 @ 1.0 s, trial "0"
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")

    state.current_frame = _FakeVideo().time_to_frame(1.0, round_nearest=True) - 2
    dialog._confirm()

    df = state._all_labels_df
    row = df[(df["trial"] == "0") & (df["labels"] == 4)].iloc[0]
    assert row["onset_s"] == pytest.approx(0.96)
    assert math.isnan(row["offset_s"])
    dialog._stop()


# ----------------------------------------------------------------------
# Confidence: a boundary placed by eye is a hand-made label
# ----------------------------------------------------------------------


def test_confirm_stamps_the_refined_row_as_hand_made(dialog):
    """Committing a boundary sets that row's confidence to 1.0; the rows the
    model also predicted keep their own scores."""
    state = dialog.app_state
    state._all_labels_df = state._all_labels_df.assign(confidence=[0.3, 0.4, 0.5, 0.6])
    _check_all_labels(dialog)
    dialog._start()  # first seed: trial "0", label 4, point at 1.0 s
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")

    state.current_frame = _FakeVideo().time_to_frame(1.1, round_nearest=True)
    dialog._confirm()

    df = state._all_labels_df
    trial0 = df[df["trial"] == "0"]
    assert float(trial0[trial0["labels"] == 4].iloc[0]["confidence"]) == 1.0
    assert float(trial0[trial0["labels"] == 8].iloc[0]["confidence"]) == pytest.approx(0.4)
    dialog._stop()


def test_confirm_adds_confidence_when_the_column_is_absent(dialog):
    """A labels file written before the column existed still records the
    refined row as human — every other row reads as certain anyway."""
    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")
    state.current_frame = _FakeVideo().time_to_frame(1.1, round_nearest=True)
    dialog._confirm()

    df = state._all_labels_df
    assert (df["confidence"] == 1.0).all()
    dialog._stop()


# ----------------------------------------------------------------------
# Backspace: the event should not exist at all
# ----------------------------------------------------------------------


def test_delete_removes_the_row_and_skips_its_other_boundary(dialog):
    """Deleting a state event's START drops the whole label, so its END seed
    leaves the queue with it and the next seed is the following label."""
    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()
    dialog._idx = 1  # START of the state event (label 8)
    dialog._jump_current()
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")

    dialog._delete_current()

    df = state._all_labels_df
    assert df[(df["trial"] == "0") & (df["labels"] == 8)].empty
    assert dialog._n_deleted == 1
    target = dialog._targets[dialog._idx]
    assert (target.inst["labels"], target.field) == (6, "point")
    dialog._stop()


def test_delete_is_logged_as_a_deletion(dialog, tmp_path):
    """The history keeps the original times and no new ones; the export says
    so in a ``deleted`` column."""
    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")
    dialog._delete_current()

    rec = state.refine_log[-1]
    assert rec["deleted"] is True
    assert rec["labels"] == 4 and rec["orig_onset_s"] == pytest.approx(1.0)
    assert rec["new_onset_s"] is None

    pre_path, post_path = export_refine_log(state.refine_log, tmp_path / "labels.tsv")
    post = pd.read_csv(post_path, sep="	")
    assert bool(post["deleted"].iloc[0])
    assert pd.isna(post["onset_s"].iloc[0])
    assert pd.read_csv(pre_path, sep="	")["onset_s"].iloc[0] == pytest.approx(1.0)
    dialog._stop()


def test_delete_refuses_a_seed_from_another_trial(dialog):
    """The row lives in the current trial's table — deleting while the GUI
    sits elsewhere would drop the wrong label."""
    state = dialog.app_state
    _check_all_labels(dialog)
    dialog._start()
    state.trials_sel = "1"  # the first seed belongs to trial "0"
    state.label_intervals = state.get_trial_intervals("1")

    dialog._delete_current()

    assert dialog._n_deleted == 0
    assert len(state._all_labels_df) == 4
    dialog._stop()


def test_delete_shortcuts_live_with_the_session(dialog):
    """Backspace/Delete are application-wide only while refining."""
    _check_all_labels(dialog)
    assert dialog._delete_shortcuts == []
    dialog._start()
    assert len(dialog._delete_shortcuts) == 2
    assert all(sc.context() == Qt.ApplicationShortcut for sc in dialog._delete_shortcuts)
    dialog._stop()
    assert dialog._delete_shortcuts == []


# ----------------------------------------------------------------------
# Queues handed over from the frames grid
# ----------------------------------------------------------------------


def _seed(trial, labels, onset, offset, field, event_type="point"):
    return {
        "trial": trial,
        "labels": labels,
        "onset_s": onset,
        "offset_s": offset,
        "individual": "a",
        "individual_rec": "",
        "event_type": event_type,
        "field": field,
    }


def test_targets_from_seeds_orders_and_shares_the_instance():
    """Seeds arrive in tile order; the queue visits (trial, onset) with START
    before END, and one state event's two boundaries share an inst."""
    seeds = [
        _seed("1", 4, 0.5, math.nan, "point"),
        _seed("0", 8, 2.0, 3.0, "end", "state"),
        _seed("0", 8, 2.0, 3.0, "start", "state"),
    ]
    targets = targets_from_seeds(seeds)
    assert [(t.inst["trial"], t.inst["labels"], t.field) for t in targets] == [
        ("0", 8, "start"),
        ("0", 8, "end"),
        ("1", 4, "point"),
    ]
    assert targets[0].inst is targets[1].inst


def test_start_from_seeds_walks_only_those_boundaries(dialog):
    """The class list and the trials filter play no part — the handed-over
    seeds are the whole queue, and it is not remembered as a resume point."""
    state = dialog.app_state
    assert dialog.start_from_seeds([_seed("1", 4, 0.5, math.nan, "point")], from_grid=True)
    assert dialog._session_active
    assert [(t.inst["trial"], t.inst["labels"]) for t in dialog._targets] == [("1", 4)]
    assert state.refine_resume is None
    # The class list mirrors what is being refined.
    checked = {
        dialog.label_list.item(i).data(Qt.UserRole)
        for i in range(dialog.label_list.count())
        if dialog.label_list.item(i).checkState() == Qt.Checked
    }
    assert checked == {4}
    dialog._stop()


def test_start_from_seeds_needs_a_video(dialog):
    state = dialog.app_state
    state.video = None
    assert not dialog.start_from_seeds([_seed("1", 4, 0.5, math.nan, "point")])
    assert not dialog._session_active
