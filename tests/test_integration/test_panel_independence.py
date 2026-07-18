"""Dropping feature B as a new panel while a feature-A panel is open: the new
panel must show B (not A), the sidebar must follow, and A's panel is untouched.

Guards the canonical-feature-list invariant: the left sidebar, panel creation
(`_available_features`), and the features combo all use
``catalog.feature_choices()`` — a feature offered anywhere is displayable
everywhere, so the dropdown can never silently show another plot's feature.
"""

from qtpy.QtWidgets import QApplication

from ethograph.utils.qt import get_combo_value


def test_heatmap_drop_uses_dropped_feature(moll2025_gui):
    _, meta = moll2025_gui
    pc = meta.plot_container
    dw = meta.data_widget
    feats = pc._available_features()
    assert len(feats) >= 2, feats
    feat_a, feat_b = str(feats[0]), str(feats[1])

    # A lineplot showing feat_a is open and active; sidebar edited it.
    lp = pc.line_plots[0]
    pc.active_feature_plot = lp
    dw.apply_panel_control("features", feat_a)
    QApplication.processEvents()
    assert lp._effective_feature() == feat_a

    # Drop feat_b as Heatmap (the exact drop code path).
    meta._create_panel_for_source("feature", feat_b, "Heatmap")
    QApplication.processEvents()

    hm = pc.heatmap_plot
    assert hm._effective_feature() == feat_b, (
        f"heatmap feature leaked: {hm._effective_feature()!r} != {feat_b!r}; "
        f"panel_state={hm.panel_state}"
    )
    assert pc.active_feature_plot is hm, f"active is {pc.active_feature_plot}"
    combo = dw.combos.get("features")
    assert get_combo_value(combo) == feat_b, (
        f"sidebar combo shows {get_combo_value(combo)!r}, expected {feat_b!r}"
    )
    # The original lineplot is untouched.
    assert lp._effective_feature() == feat_a


def test_lineplot_drop_uses_dropped_feature(moll2025_gui):
    _, meta = moll2025_gui
    pc = meta.plot_container
    dw = meta.data_widget
    feats = pc._available_features()
    feat_a, feat_b = str(feats[0]), str(feats[1])

    lp = pc.line_plots[0]
    pc.active_feature_plot = lp
    dw.apply_panel_control("features", feat_a)
    QApplication.processEvents()

    n_before = len(pc.line_plots)
    meta._create_panel_for_source("feature", feat_b, "Lineplot")
    QApplication.processEvents()
    assert len(pc.line_plots) == n_before + 1
    new_plot = pc.line_plots[-1]
    assert new_plot._effective_feature() == feat_b, (
        f"lineplot feature leaked: {new_plot._effective_feature()!r} != {feat_b!r}; "
        f"panel_state={new_plot.panel_state}"
    )
    assert pc.active_feature_plot is new_plot
    combo = dw.combos.get("features")
    assert get_combo_value(combo) == feat_b, (
        f"sidebar combo shows {get_combo_value(combo)!r}, expected {feat_b!r}"
    )
    assert lp._effective_feature() == feat_a


def test_sidebar_and_creation_use_canonical_feature_list(birdpark_gui):
    """`_available_features` must be the curated combo list, not raw
    ds.data_vars (which includes label bookkeeping vars like onset_s)."""
    _, meta = birdpark_gui
    pc = meta.plot_container
    feats = pc._available_features()
    combo = meta.data_widget.combos.get("features")
    combo_values = [combo.itemData(i) or combo.itemText(i) for i in range(combo.count())]
    assert feats == combo_values, (feats, combo_values)
    assert "onset_s" not in feats


def test_sync_adds_missing_feature_to_combo(birdpark_gui):
    """Even for a feature outside the combo (e.g. injected/derived), the
    sidebar must display the active plot's true feature — never another
    plot's value."""
    _, meta = birdpark_gui
    pc = meta.plot_container
    dw = meta.data_widget
    plot = pc.line_plots[0]
    pc.active_feature_plot = plot
    plot.set_panel_control("features", "some_injected_feature")
    dw.sync_sidebar_from_active_plot()
    combo = dw.combos.get("features")
    assert get_combo_value(combo) == "some_injected_feature"
