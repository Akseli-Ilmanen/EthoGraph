"""Session slider scope on a pynapple dataset: display basis, labels, clicks.

Covers the display-basis refactor end to end on real data:
- switching to session scope flips display_basis and gives absolute bounds,
- every trial's labels appear in the display view at session positions,
- a (simulated) label click in another trial's span switches to that trial
  and stores a trial-relative row in the right trial,
- trial navigation puts the marker on the trial's session start.
"""

from __future__ import annotations

import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

SESSION_SCOPE = "Session start → Session end"
TRIAL_SCOPE = "Trial start → Trial end"


def _labels_df(onset: float, label_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset_s": [onset],
            "offset_s": [onset + 0.4],
            "labels": [label_id],
            "individual": ["default"],
        }
    )


@pytest.fixture
def session_gui(moll2025_pynapple_gui, qtbot):
    viewer, meta = moll2025_pynapple_gui
    meta.navigation_widget.scope_combo.setCurrentText(SESSION_SCOPE)
    QApplication.processEvents()
    yield viewer, meta
    # Leave the local settings as trial scope for other tests.
    meta.navigation_widget.scope_combo.setCurrentText(TRIAL_SCOPE)
    QApplication.processEvents()


def test_session_scope_display_basis_and_bounds(session_gui):
    _, meta = session_gui
    state = meta.app_state

    assert state.display_basis == "session"
    sc = state.source_collection
    assert sc is not None and sc.n_trials > 1

    wb = state.window_bounds
    session = sc.session_range
    assert wb is not None and session is not None
    assert wb.start_s == pytest.approx(session.start_s)
    assert wb.end_s == pytest.approx(session.end_s)


def test_display_intervals_show_all_trials_at_session_positions(session_gui):
    _, meta = session_gui
    state = meta.app_state
    sc = state.source_collection
    trials = state.trials
    assert len(trials) >= 2
    t_a, t_b = trials[0], trials[1]

    state.set_trial_intervals(t_a, _labels_df(0.1))
    state.set_trial_intervals(t_b, _labels_df(0.2))
    state.label_intervals = state.get_trial_intervals(state.trials_sel)

    view = state.get_display_intervals()
    by_trial = dict(zip(view["trial"], view["onset_s"]))
    assert by_trial[t_a] == pytest.approx(sc.to_session(t_a, 0.1))
    assert by_trial[t_b] == pytest.approx(sc.to_session(t_b, 0.2))

    # Drawing the session view must not raise.
    meta.data_widget.update_label_plot(state.get_ds_kwargs())


def test_session_click_labels_the_trial_under_it(session_gui, qtbot):
    _, meta = session_gui
    state = meta.app_state
    sc = state.source_collection
    lw = meta.labels_widget

    trials = state.trials
    target_trial = trials[2] if len(trials) > 2 else trials[-1]
    assert state.trials_sel != target_trial

    # Arm a label class the way activate_label would.
    label_id = next((lid for lid in lw._mappings if isinstance(lid, int) and lid != 0), None)
    if label_id is None:
        lw._mappings[1] = {"name": "test", "color": "#ff0000", "branch": 0}
        label_id = 1
    lw.selected_labels = label_id
    lw.ready_for_label_click = True
    lw.first_click = None
    lw.second_click = None

    # Two clicks inside the target trial's span, on the session axis.
    t0 = sc.to_session(target_trial, 0.5)
    t1 = sc.to_session(target_trial, 1.0)
    lw._on_plot_clicked({"x": t0, "button": Qt.LeftButton})
    QApplication.processEvents()
    lw._on_plot_clicked({"x": t1, "button": Qt.LeftButton})
    QApplication.processEvents()

    assert state.trials_sel == target_trial
    stored = state.get_trial_intervals(target_trial)
    match = stored[stored["labels"] == label_id]
    assert len(match) >= 1
    # Stored trial-relative, not session-absolute.
    assert match["onset_s"].iloc[0] == pytest.approx(0.5, abs=0.05)
    assert match["offset_s"].iloc[0] == pytest.approx(1.0, abs=0.05)


def test_click_in_intertrial_gap_places_nothing(session_gui):
    _, meta = session_gui
    state = meta.app_state
    sc = state.source_collection
    lw = meta.labels_widget

    trials = state.trials
    # A time just after trial 0's end, before trial 1's start (the gap).
    end_of_first = sc.trial_range(0).end_s
    start_of_second = sc.trial_range(1).start_s
    if start_of_second - end_of_first < 0.2:
        pytest.skip("No inter-trial gap in this dataset")
    gap_t = (end_of_first + start_of_second) / 2.0

    lw._mappings.setdefault(1, {"name": "test", "color": "#ff0000", "branch": 0})
    lw.selected_labels = 1
    lw.ready_for_label_click = True
    lw.first_click = None
    before = {t: len(state.get_trial_intervals(t)) for t in trials[:3]}

    lw._on_plot_clicked({"x": gap_t, "button": Qt.LeftButton})
    QApplication.processEvents()

    assert lw.first_click is None
    after = {t: len(state.get_trial_intervals(t)) for t in trials[:3]}
    assert after == before


def test_trial_change_centers_marker_on_session_start(session_gui):
    _, meta = session_gui
    state = meta.app_state
    sc = state.source_collection
    nav = meta.navigation_widget

    target = state.trials[3] if len(state.trials) > 3 else state.trials[-1]
    nav.trials_combo.setCurrentText(str(target))
    QApplication.processEvents()

    assert state.trials_sel == target
    expected_start = sc.to_session(target, 0.0)
    marker_t = nav._current_time()
    assert marker_t == pytest.approx(expected_start, abs=0.5)


def test_camera_view_blanking(qapp):
    """The black 'no input' cover toggles without touching the decoder."""
    from ethograph.gui.pygfx_video import CameraView

    view = CameraView()
    assert not view.is_blanked
    view.set_blanked(True)
    assert view.is_blanked
    view.set_blanked(False)
    assert not view.is_blanked
    view.deleteLater()
