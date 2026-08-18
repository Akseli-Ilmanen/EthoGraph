"""Ad-hoc repro: two-click state label on the birdpark (xarray) dataset."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from ethograph.labels.intervals import find_interval_at


def test_birdpark_label_two_clicks(birdpark_gui):
    _, meta = birdpark_gui
    lw = meta.labels_widget
    state = meta.app_state

    print("trials_sel:", state.trials_sel)
    print("display_basis:", state.display_basis)
    print("window_bounds:", state.window_bounds)
    print("mappings:", dict(lw._mappings))
    print("active branch:", state._active_branch, "editable:", state.editable_label_ids)

    lw.activate_label(1)
    assert lw.ready_for_label_click, "label did not arm"
    print("selected_labels:", lw.selected_labels)

    t_start, t_end = 0.5, 1.0
    lw._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
    print("after click 1: first_click =", lw.first_click)
    assert lw.first_click is not None, "first click did not register"

    lw._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
    QApplication.processEvents()
    print("after click 2: first_click =", lw.first_click, "second =", lw.second_click)

    df = state.label_intervals
    print("label_intervals:\n", df)
    assert df is not None and not df.empty, "No intervals after label creation"
    idx = find_interval_at(df, (t_start + t_end) / 2, lw._current_individual())
    assert idx is not None, "Interval not found at midpoint"
