"""Horizontal reference lines, driven from the sidebar the way a user does.

A line belongs to the panel it was drawn on and lives as long as that panel
does — navigating to another trial re-renders the plot but must not wipe it.
"""

from qtpy.QtWidgets import QApplication


def _settings(meta, plot):
    meta.plot_container.active_feature_plot = plot
    ps = meta.plot_settings_widget
    ps.sync_axes_to_active_plot()
    return ps


def test_a_line_added_from_the_sidebar_survives_a_trial_change(moll2025_gui):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    ps = _settings(meta, plot)

    ps.hline_value_edit.setText("0.5")
    ps.add_hline_button.click()

    assert plot.hline_values() == [0.5]
    assert ps.hline_value_edit.text() == ""
    assert "0.5" in ps.hline_list_label.text()

    meta.navigation_widget.next_trial()
    QApplication.processEvents()

    assert plot.hline_values() == [0.5]


def test_lines_belong_to_the_panel_they_were_drawn_on(moll2025_gui):
    _, meta = moll2025_gui
    first = meta.plot_container.line_plots[0]
    second = meta.plot_container.add_lineplot()
    QApplication.processEvents()

    ps = _settings(meta, first)
    ps.hline_value_edit.setText("1")
    ps.add_hline_button.click()
    ps.hline_value_edit.setText("2")
    ps.add_hline_button.click()

    assert first.hline_values() == [1.0, 2.0]
    assert second.hline_values() == []

    # Switching the sidebar to the other panel shows that panel's lines.
    ps = _settings(meta, second)
    assert ps.hline_list_label.text() == ""
    assert not ps.clear_hlines_button.isEnabled()

    ps = _settings(meta, first)
    ps.clear_hlines_button.click()
    assert first.hline_values() == []
