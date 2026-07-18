"""Integration tests for automatic panel-layout persistence.

The ``gui`` fixture disables ``app_state._layout_snapshot_provider`` for
hermeticity, so the auto-save → snapshot → local_settings.yaml path is
otherwise untested. These tests re-enable it against a loaded dataset and
drive the exact code path the 30s auto-save timer and the close-save use.
"""

from __future__ import annotations

import yaml


def _redirect_settings_paths(app_state, tmp_path, monkeypatch):
    local_path = tmp_path / "local_settings.yaml"
    global_path = tmp_path / "gui_settings.yaml"
    monkeypatch.setattr(type(app_state), "_local_settings_path", lambda self: local_path)
    monkeypatch.setattr(type(app_state), "_global_settings_path", lambda self: global_path)
    return local_path, global_path


def test_panel_layout_persisted_by_autosave(moll2025_gui, tmp_path, monkeypatch):
    viewer, meta = moll2025_gui
    app_state = meta.app_state
    assert app_state.ready

    app_state._layout_snapshot_provider = meta._snapshot_layouts
    local_path, global_path = _redirect_settings_paths(app_state, tmp_path, monkeypatch)

    assert app_state.save_to_yaml()

    local_state = yaml.safe_load(local_path.read_text(encoding="utf-8"))
    layout = local_state.get("panel_layout")
    assert layout, "panel_layout missing from local_settings.yaml"
    assert layout.get("dock_state_b64")
    panel_types = {p["type"] for p in layout["panels"]}
    assert "lineplot" in panel_types or "heatmap" in panel_types

    global_state = yaml.safe_load(global_path.read_text(encoding="utf-8"))
    assert "panel_layout" not in global_state, "panel_layout is per-dataset (local) state"
    assert global_state.get("window_state"), "window_state missing from gui_settings.yaml"


def test_space_dock_position_restored(moll2025_gui, qtbot):
    """A space dock moved by the user must come back in the same dock area
    after the layout is re-applied (regression: restoreDockWidget on a
    late-created dock left it squished top-right and hidden). Placement is
    explicit per dock — a full QMainWindow restoreState blob reparents the
    native pygfx/GL canvases and crashes on Windows."""
    from qtpy.QtCore import Qt

    viewer, meta = moll2025_gui
    shell = meta.shell
    dw = meta.data_widget

    for existing in list(dw.space_plots):  # dataset load may auto-create one
        dw.remove_space_plot(existing)

    sp = dw.add_space_plot(focus=False)
    qtbot.wait(50)
    shell.addDockWidget(Qt.BottomDockWidgetArea, sp.dock_widget)
    qtbot.wait(10)

    meta._snapshot_layouts()
    layout = meta.app_state.panel_layout
    assert len(layout["space_plots"]) == 1
    entry = layout["space_plots"][0]
    assert entry.get("dock_area") == int(Qt.BottomDockWidgetArea.value)
    assert entry.get("dock_size")

    dw.remove_space_plot(sp)
    assert not dw.space_plots

    # Call the class method directly: the gui fixture wraps the instance
    # attribute with a panel_layout-nulling guard for hermeticity.
    type(meta).apply_saved_panel_layout(meta)
    # The dock-state restore is deferred to the shell's first show (mirrors
    # the cover-page flow, where loads happen before shell.show()).
    shell.show()
    qtbot.wait(100)

    assert len(dw.space_plots) == 1
    dock = dw.space_plots[0].dock_widget
    assert shell.dockWidgetArea(dock) == Qt.BottomDockWidgetArea
    assert not dock.isHidden()
    shell.hide()


def test_save_survives_snapshot_failure(moll2025_gui, tmp_path, monkeypatch):
    """A raising layout snapshot must not kill the whole save — the auto-save
    QTimer slot would swallow the exception and every save would silently
    stop happening (regression: no settings written for the rest of the
    session)."""
    viewer, meta = moll2025_gui
    app_state = meta.app_state

    def broken_snapshot():
        raise RuntimeError("wrapped C/C++ object deleted")

    app_state._layout_snapshot_provider = broken_snapshot
    local_path, global_path = _redirect_settings_paths(app_state, tmp_path, monkeypatch)

    assert app_state.save_to_yaml()
    assert local_path.exists()
    assert global_path.exists()
