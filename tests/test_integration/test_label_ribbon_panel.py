"""The label timeline is an ordinary panel instance.

It is created from the add-panel popup like any other type, takes part in
the label overlay, survives a layout round trip, and is opened on its own
only when the Labels tab asks for it and nothing else is on screen.
"""

import pytest
from qtpy.QtWidgets import QApplication

from ethograph.gui.app_constants import LABELLING_MODE_FRAME

pytestmark = pytest.mark.usefixtures("gui")


def _ribbons(pc):
    return pc.panels_of_type("labels")


def test_ribbon_is_a_panel_instance_that_carries_labels(moll2025_gui):
    _viewer, meta = moll2025_gui
    pc = meta.plot_container

    ribbon = pc.add_panel("labels")
    QApplication.processEvents()

    assert ribbon in pc._get_all_plots(), "the label overlay reaches it"
    assert ribbon in list(pc._visible_plots()), "the time marker and x-sync reach it"
    assert {"type": "labels"} in pc.layout_state()["panels"]

    pc.remove_panel(ribbon)
    assert _ribbons(pc) == []


def test_layout_round_trip_recreates_the_ribbon(moll2025_gui):
    _viewer, meta = moll2025_gui
    pc = meta.plot_container
    pc.add_panel("labels")
    state = pc.layout_state()

    pc.apply_layout_state(state)
    QApplication.processEvents()

    assert len(_ribbons(pc)) == 1


def test_ribbon_opens_only_when_nothing_else_is_shown(moll2025_gui):
    _viewer, meta = moll2025_gui
    pc = meta.plot_container

    meta.app_state.labelling_mode = LABELLING_MODE_FRAME
    meta.app_state.label_ribbon_auto = True
    meta.ensure_label_ribbon()
    assert _ribbons(pc) == [], "a session with panels gets nothing extra"

    for plot in list(pc._dyn_panels):
        pc.remove_panel(plot)
    assert not pc.has_open_plots()

    meta.app_state.label_ribbon_auto = False
    meta.ensure_label_ribbon()
    assert _ribbons(pc) == []

    meta.app_state.label_ribbon_auto = True
    meta.ensure_label_ribbon()
    assert len(_ribbons(pc)) == 1


def test_frame_mode_places_a_label_from_the_video_frame(moll2025_gui):
    """The key twice around a frame step is the whole placement."""
    _viewer, meta = moll2025_gui
    state = meta.app_state
    labels = meta.labels_widget
    video = state.video
    assert video is not None

    combo = labels.labelling_mode_combo
    combo.setCurrentIndex(combo.findData(LABELLING_MODE_FRAME))
    assert state.labelling_mode == LABELLING_MODE_FRAME

    state.label_intervals = state.label_intervals.iloc[0:0]
    state.set_trial_intervals(state.trials_sel, state.label_intervals)
    key_id = next(lid for lid, m in labels._mappings.items() if isinstance(lid, int) and lid != 0)
    labels._mappings[key_id]["event_type"] = "state"

    video.seek_to_frame(10)
    labels.activate_label(key_id)
    assert labels.first_click is not None
    assert labels.ready_for_label_click is False

    video.seek_to_frame(40)
    labels.activate_label(key_id)
    QApplication.processEvents()

    df = state.label_intervals
    assert len(df) == 1
    _trial, t_start = state.from_display(video.frame_to_time(10))
    _trial, t_end = state.from_display(video.frame_to_time(40))
    assert df.iloc[0]["onset_s"] == pytest.approx(t_start)
    assert df.iloc[0]["offset_s"] == pytest.approx(t_end)
    assert int(df.iloc[0]["labels"]) == key_id
