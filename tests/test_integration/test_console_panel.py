"""The console panel: click a plot, transform what it shows, plot the result.

The invariant under test is the one that makes the feature comprehensible —
the console binds *what the panel renders* (a numpy array for the panel's own
feature, selections and window), never the DataArray behind it — and anything
assigned becomes a feature the add-panel popup offers.
"""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from ethograph.gui.plots_console import to_identifier
from ethograph.io.derived import DerivedLoader

#: The payload every plot emits with ``plot_clicked``.
_CLICK = {"x": None, "button": Qt.NoButton}


def _console(meta):
    meta._add_console_panel()
    QApplication.processEvents()
    return meta.plot_container.console_panel


def test_the_transcript_starts_with_the_bound_panel_not_a_banner(moll2025_gui):
    """Help lives behind the ? button; the transcript is a record of the work."""
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)

    lines = [line for line in console.output.toPlainText().splitlines() if line.strip()]
    assert len(lines) == 1, lines
    assert lines[0].startswith(to_identifier(plot._effective_feature()))
    assert "shape" in lines[0]

    assert not console.help_label.isVisibleTo(console)
    console.help_button.setChecked(True)
    assert console.help_label.isVisibleTo(console)
    assert "stack(" in console.help_label.text()
    console.help_button.setChecked(False)
    assert not console.help_label.isVisibleTo(console)
    # Toggling help never touches the transcript.
    assert len([line for line in console.output.toPlainText().splitlines() if line.strip()]) == 1


def test_console_binds_the_active_panel_as_a_plain_array(moll2025_gui):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)

    name = to_identifier(plot._effective_feature())
    bound = console.ns.get(name)
    assert isinstance(bound, np.ndarray), f"{name!r} not bound; ns={list(console.ns)}"
    assert bound.ndim in (1, 2)
    # It matches the rendered window, not the whole underlying variable.
    t0, t1 = plot.get_current_xlim()
    assert t0 <= bound.eto_time[0] and bound.eto_time[-1] <= t1


def test_sidebar_feature_change_rebinds_the_console(moll2025_gui):
    """Switching the active panel's feature in the right sidebar must rebind —
    active_changed never fires, so the console used to keep the old variable."""
    _, meta = moll2025_gui
    pc = meta.plot_container
    plot = pc.line_plots[0]
    pc.active_feature_plot = plot
    console = _console(meta)

    feats = [str(f) for f in pc._available_features()]
    first = plot._effective_feature()
    other = next(f for f in feats if f != first)

    meta.data_widget.apply_panel_control("features", other)
    QApplication.processEvents()

    assert plot._effective_feature() == other
    assert to_identifier(other) in console.ns, list(console.ns)
    assert to_identifier(other) in console.output.toPlainText()


def test_reclicking_the_active_panel_rebinds(moll2025_gui):
    _, meta = moll2025_gui
    pc = meta.plot_container
    plot = pc.line_plots[0]
    pc.active_feature_plot = plot
    console = _console(meta)

    feats = [str(f) for f in pc._available_features()]
    other = next(f for f in feats if f != plot._effective_feature())
    # Change the panel behind the console's back, then re-click the SAME panel.
    plot.set_panel_control("features", other)
    plot.update_plot()
    plot.plot_clicked.emit(_CLICK)
    QApplication.processEvents()

    assert to_identifier(other) in console.ns, list(console.ns)


def test_every_click_reports_the_name_and_shape(moll2025_gui):
    _, meta = moll2025_gui
    pc = meta.plot_container
    plot = pc.line_plots[0]
    pc.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    before = console.output.toPlainText().count(f"{name}: shape")
    for _ in range(3):
        plot.plot_clicked.emit(_CLICK)
    QApplication.processEvents()

    after = console.output.toPlainText().count(f"{name}: shape")
    assert after == before + 3


def test_assignment_becomes_an_addable_feature(moll2025_gui):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"theta = np.deg2rad({name})")
    QApplication.processEvents()

    loader = meta.app_state.data_loader
    assert isinstance(loader, DerivedLoader)
    assert loader.is_derived("theta")
    assert "theta" in loader.catalog.feature_choices()
    # The canonical list feeds the popup and the sidebar combo alike.
    assert "theta" in meta.plot_container._available_features()


def test_derived_panel_renders_the_transform(moll2025_gui):
    _, meta = moll2025_gui
    pc = meta.plot_container
    plot = pc.line_plots[0]
    pc.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"theta = np.deg2rad({name})")
    console._on_submitted("wave = np.cos(theta)")
    QApplication.processEvents()

    meta._create_panel_for_source("feature", "wave", "Lineplot")
    QApplication.processEvents()

    derived_panel = pc.line_plots[-1]
    assert derived_panel._effective_feature() == "wave"
    t0, t1 = derived_panel.get_current_xlim()
    plot_data = meta.app_state.data_loader.select("wave", {}, t0=t0, t1=t1)
    assert plot_data is not None
    assert np.all(np.abs(plot_data.data[~np.isnan(plot_data.data)]) <= 1.0 + 1e-9)


def test_stack_renders_two_named_curves_in_one_panel(moll2025_gui):
    """The multi-column workflow: one panel, one legend, a colour per column."""
    _, meta = moll2025_gui
    pc = meta.plot_container
    plot = pc.line_plots[0]
    pc.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"rad = np.deg2rad({name})")
    console._on_submitted("rings = stack(sin=np.sin(rad), cos=np.cos(rad))")
    QApplication.processEvents()
    assert meta.app_state.data_loader.is_derived("rings")

    meta._create_panel_for_source("feature", "rings", "Lineplot")
    QApplication.processEvents()

    panel = pc.line_plots[-1]
    assert panel._effective_feature() == "rings"
    t0, t1 = panel.get_current_xlim()
    plot_data = meta.app_state.data_loader.select("rings", {}, t0=t0, t1=t1)
    assert plot_data.data.ndim == 2 and plot_data.data.shape[1] == 2
    assert plot_data.dim_labels == ["sin", "cos"]
    # Two curves, each its own colour, with a legend naming them.
    curves = [item for item in panel.plot_items if hasattr(item, "opts")]
    assert len(curves) == 2, f"expected 2 curves, got {len(curves)}"
    assert panel.plot_item.legend is not None
    pens = {str(item.opts.get("pen")) for item in curves}
    assert len(pens) == 2, "both curves drawn in the same colour"


def test_stack_takes_already_defined_variables_positionally(moll2025_gui):
    """The natural workflow: define sin and cos, then stack(sin, cos)."""
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"rad = np.deg2rad({name})")
    console._on_submitted("sin = np.sin(rad)")
    console._on_submitted("cos = np.cos(rad)")
    console._on_submitted("sin_cos = stack(sin, cos)")
    QApplication.processEvents()

    loader = meta.app_state.data_loader
    assert loader.is_derived("sin_cos"), console.output.toPlainText()
    plot_data = loader.select("sin_cos", {}, t0=None, t1=None)
    # Column names come from the variables themselves — nothing repeated.
    assert plot_data.dim_labels == ["sin", "cos"]
    assert plot_data.data.shape[1] == 2
    # sin and cos remain features in their own right.
    assert loader.is_derived("sin") and loader.is_derived("cos")


def test_a_stacked_feature_drops_as_a_working_space_plot(moll2025_gui):
    """Space plots pick X and Y from a dim; a derived feature exposing none
    left both combos empty and nothing was drawn."""
    from ethograph.io.derived import DERIVED_COLUMN_DIM

    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"rad = np.deg2rad({name})")
    console._on_submitted("xy = stack(x=np.cos(rad), y=np.sin(rad))")
    QApplication.processEvents()

    n_before = len(meta.data_widget.space_plots)
    meta._create_panel_for_source("feature", "xy", "Space (2D)")
    QApplication.processEvents()

    assert len(meta.data_widget.space_plots) == n_before + 1
    sp = meta.data_widget.space_plots[-1]
    assert sp.feature_combo.currentText() == "xy"
    assert sp.space_dim_combo.currentText() == DERIVED_COLUMN_DIM
    axes = {sp.x_combo.currentText(), sp.y_combo.currentText()}
    assert axes == {"x", "y"}, axes


def test_stack_offers_the_multi_column_plot_types(moll2025_gui):
    """A 2-column derived feature must be droppable as Heatmap / Space too."""
    from ethograph.gui.source_popup import allowed_plot_types, feature_ncols

    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"rad = np.deg2rad({name})")
    console._on_submitted("rings = stack(sin=np.sin(rad), cos=np.cos(rad))")
    QApplication.processEvents()

    assert feature_ncols(meta.app_state, "rings") == 2
    assert "Heatmap" in allowed_plot_types("feature", "rings", meta.app_state)


def test_gradient_without_an_axis_is_explained_not_swallowed(moll2025_gui):
    """np.gradient returns one array PER AXIS — the user must be told why
    nothing appeared, and told what to type instead."""
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"pos = np.column_stack([{name}, {name}, {name}])")
    console._on_submitted("grad = np.gradient(pos)")
    QApplication.processEvents()

    text = console.output.toPlainText()
    assert "grad" in text and "arrays, not one" in text, text
    assert "axis=0" in text
    assert not meta.app_state.data_loader.is_derived("grad")


def test_gradient_along_time_becomes_a_feature(moll2025_gui):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"pos = np.column_stack([{name}, {name}, {name}])")
    console._on_submitted("grad = np.gradient(pos, t, axis=0)")  # t = per second
    QApplication.processEvents()

    loader = meta.app_state.data_loader
    assert loader.is_derived("grad"), console.output.toPlainText()
    assert loader.derived["grad"].is_snapshot, "np.gradient is not a ufunc"
    assert loader.derived["grad"].n_columns == 3

    t0, t1 = plot.get_current_xlim()
    plot_data = loader.select("grad", {}, t0=t0, t1=t1)
    assert plot_data.data.shape[1] == 3


def test_the_time_vector_is_bound_alongside(moll2025_gui):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)

    name = to_identifier(plot._effective_feature())
    assert np.allclose(console.ns["t"], console.ns[name].eto_time)


def test_forget_removes_the_feature_again(moll2025_gui):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"theta = np.deg2rad({name})")
    QApplication.processEvents()
    assert "theta" in meta.app_state.data_loader.catalog.feature_choices()

    console._on_submitted("forget('theta')")
    QApplication.processEvents()
    assert "theta" not in meta.app_state.data_loader.catalog.feature_choices()


def test_clear_wipes_the_transcript_but_keeps_the_variables(moll2025_gui):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"theta = np.deg2rad({name})")
    QApplication.processEvents()
    assert console.output.toPlainText()

    console._on_submitted("clear")  # bare, no parentheses
    QApplication.processEvents()
    assert console.output.toPlainText().strip() == ""
    # The work survives: variable still bound, feature still addable.
    assert "theta" in console.ns
    assert "theta" in meta.app_state.data_loader.catalog.feature_choices()


def test_clear_all_resets_the_session(moll2025_gui):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"theta = np.deg2rad({name})")
    QApplication.processEvents()

    console._on_submitted("clear()")  # the called form still works
    QApplication.processEvents()
    assert console.output.toPlainText().strip() == ""
    assert "theta" in console.ns

    console._on_submitted("clear(all=True)")
    QApplication.processEvents()
    assert "theta" not in console.ns
    assert name not in console.ns
    assert "theta" not in meta.app_state.data_loader.catalog.feature_choices()
    # The helpers are still there — it is a reset, not a broken namespace.
    assert callable(console.ns["clear"])
    assert console.ns["np"] is np

    console._on_submitted("theta = 1")  # namespace still works afterwards
    QApplication.processEvents()
    assert console.ns["theta"] == 1


def test_ctrl_l_clears_the_transcript(moll2025_gui):

    from qtpy.QtGui import QKeyEvent

    _, meta = moll2025_gui
    console = _console(meta)
    console.write("noise")
    assert console.output.toPlainText()

    console.input.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_L, Qt.ControlModifier))
    QApplication.processEvents()
    assert console.output.toPlainText().strip() == ""


def test_derived_features_die_with_the_trial(moll2025_gui):
    """They describe one trial's panel, so they must not follow the user out."""
    _, meta = moll2025_gui
    trials = list(meta.app_state.trials)
    if len(trials) < 2:
        import pytest

        pytest.skip("needs a multi-trial dataset")

    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"theta = np.deg2rad({name})")
    QApplication.processEvents()
    assert "theta" in meta.app_state.data_loader.catalog.feature_choices()

    # Exactly what the bottom bar's next/prev-trial buttons do.
    other = next(t for t in trials if t != meta.app_state.trials_sel)
    meta.app_state.trials_sel = other
    meta.app_state.trial_changed.emit()
    QApplication.processEvents()

    assert "theta" not in console.ns
    assert not meta.app_state.data_loader.derived
    assert "theta" not in meta.app_state.data_loader.catalog.feature_choices()
    # The transcript survives, with a line explaining where they went.
    assert "trial changed" in console.output.toPlainText()


def test_a_panel_on_a_dropped_feature_is_closed(moll2025_gui):
    """A panel whose feature no longer exists would render blank forever."""
    _, meta = moll2025_gui
    pc = meta.plot_container
    plot = pc.line_plots[0]
    pc.active_feature_plot = plot
    console = _console(meta)
    name = to_identifier(plot._effective_feature())

    console._on_submitted(f"theta = np.deg2rad({name})")
    QApplication.processEvents()
    meta._create_panel_for_source("feature", "theta", "Lineplot")
    QApplication.processEvents()
    assert any(p._effective_feature() == "theta" for p in pc.line_plots)
    n_before = len(pc.line_plots)

    console._on_submitted("forget('theta')")
    QApplication.processEvents()

    assert not any(p._effective_feature() == "theta" for p in pc.line_plots)
    assert len(pc.line_plots) == n_before - 1
    # Panels on real features are untouched.
    assert any(p._effective_feature() == plot._effective_feature() for p in pc.line_plots)


def test_console_is_a_singleton_dock(moll2025_gui):
    """One namespace per session — re-adding must not start a second one."""
    _, meta = moll2025_gui
    first = _console(meta)
    first.ns["marker"] = 1
    second = _console(meta)
    assert second is first
    assert second.ns.get("marker") == 1
