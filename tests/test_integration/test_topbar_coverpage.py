"""Integration tests for the top menu bar and the IO cover page.

Exercises the two GUI layout-refactor deliverables against the real
``EthographMainWindow``/``MetaWidget`` (via the ``gui`` fixture) and against a
loaded template dataset (``birdpark_gui``), so the menu actions and the
drag&drop alignment builder are driven end-to-end rather than only imported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ethograph.io.validation import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS




def _menu(shell, title: str):
    for action in shell.menuBar().actions():
        if action.text().replace("&", "") == title:
            return action.menu()
    raise AssertionError(f"menu {title!r} not found")


def _find_action(menu, label: str):
    for action in menu.actions():
        if action.text().replace("&", "").startswith(label):
            return action
        if action.menu() is not None:
            for sub in action.menu().actions():
                if sub.text().replace("&", "").startswith(label):
                    return sub
    raise AssertionError(f"action {label!r} not found")


# ---------------------------------------------------------------------------
# Top bar (no dataset needed)
# ---------------------------------------------------------------------------



def test_video_area_survives_attach(gui):
    """setCentralWidget(plots) must not delete the video area / CameraView.

    Regression: the video used to be the central widget, and making the plots
    central deleted the CameraView C++ object (crash in _add_labels_shapes_layer).
    """
    from qtpy.QtWidgets import QLabel

    shell, meta = gui
    canvas = shell.canvas_widget()
    assert canvas is not None
    # Touch the CameraView and build a child QLabel on the canvas exactly like
    # the label overlay does — raises RuntimeError if the C++ object was deleted.
    assert shell.video_area.primary.objectName() is not None
    QLabel(canvas)


def test_sidebars_span_full_height(gui):
    from qtpy.QtCore import Qt

    shell, meta = gui
    assert shell.corner(Qt.TopLeftCorner) == Qt.LeftDockWidgetArea
    assert shell.corner(Qt.BottomLeftCorner) == Qt.LeftDockWidgetArea
    assert shell.corner(Qt.TopRightCorner) == Qt.RightDockWidgetArea
    assert shell.corner(Qt.BottomRightCorner) == Qt.RightDockWidgetArea


def test_every_panel_has_a_close_button(gui):
    shell, meta = gui
    pc = meta.plot_container
    # Every fixed panel gets its own ✕ close button.
    assert len(pc._panel_close_buttons) == 7
    pc.set_audiotrace_visible(True)
    btn = next(b for b in pc._panel_close_buttons if b.parentWidget() is pc.audio_trace_plot)
    btn.click()
    assert not pc.audio_trace_plot.isVisible()


def test_top_bar_has_expected_menus(gui):
    shell, meta = gui
    titles = [a.text().replace("&", "") for a in shell.menuBar().actions()]
    assert titles == ["File", "Layout", "Changepoints", "Neural", "Help"]



def test_show_changepoints_menu_action_syncs_state(gui):
    shell, meta = gui
    cp_menu = _menu(shell, "Changepoints")
    action = _find_action(cp_menu, "Show changepoints")
    assert action.isCheckable()

    checkbox = meta.changepoints_widget.show_cp_checkbox
    start = checkbox.isChecked()
    action.trigger()  # simulate the user clicking the menu item
    assert checkbox.isChecked() != start
    assert meta.app_state.show_changepoints == checkbox.isChecked()



def test_changepoints_popup_borrows_and_returns_detached_widget(gui):
    shell, meta = gui
    cp = meta.changepoints_widget
    holder = meta._detached_holder
    assert cp.parent() is holder  # detached widget parked in the holder

    builder = shell._top_bar
    builder._popup_section("cp", "Changepoint correction", cp)
    dlg = builder._open_popups["cp"]
    assert cp.parent() is not holder  # borrowed into the dialog
    dlg.close()
    assert cp.parent() is holder  # returned to the holder


def test_overlay_checkboxes_relocated(birdpark_gui):
    shell, meta = birdpark_gui
    conf = meta.data_widget.show_confidence_checkbox
    env = meta.data_widget.show_envelope_checkbox
    # Confidence moved under the predictions importer (io_widget).
    assert meta.io_widget.isAncestorOf(conf)
    # Envelope moved under the energy (audio-trace) group.
    assert meta.data_panel.energy_group.isAncestorOf(env)


def test_overlay_group_moved_to_labels(gui):
    shell, meta = gui
    ov = meta.data_panel.overlays_groupbox
    assert ov.title() == "Label overlay"
    assert meta.labels_widget.isAncestorOf(ov)


def test_file_menu_has_no_label_table(gui):
    shell, meta = gui
    file_menu = _menu(shell, "File")
    texts = [a.text() for a in file_menu.actions() if a.text()]
    assert not any("Label table" in t for t in texts)


def test_sidebar_has_three_sections(gui):
    shell, meta = gui
    labels = [b.text().split()[0] for b in meta._buttons]
    assert labels == ["Data", "Labels", "Nav"]


def test_plot_click_shows_only_relevant_sections(birdpark_gui):
    """Clicking a plot shows only that plot's sections (minimal sidebar)."""
    meta = birdpark_gui[1]
    ctx = meta.context_panel
    ps = meta.plot_settings_widget
    dp = meta.data_panel

    # Audio trace → energy/envelope + shared, NOT spectrogram.
    meta._on_plot_focus("audiotrace")
    assert ctx.current_context() == "audiotrace"
    assert dp.energy_group.isVisibleTo(ctx)
    assert not ps.spectrogram_panel.isVisibleTo(ctx)
    assert not dp.coords_groupbox.isVisibleTo(ctx)

    # Spectrogram → only the spectrogram panel, NOT energy.
    meta._on_plot_focus("spectrogram")
    assert ctx.current_context() == "spectrogram"
    assert ps.spectrogram_panel.isVisibleTo(ctx)
    assert not dp.energy_group.isVisibleTo(ctx)

    # Feature (lineplot) → xarray coords, NOT spectrogram.
    meta._on_plot_focus("feature")
    assert ctx.current_context() in ("lineplot", "heatmap")
    assert dp.coords_groupbox.isVisibleTo(ctx)
    assert not ps.spectrogram_panel.isVisibleTo(ctx)


def test_trials_table_hidden_without_metadata(gui):
    import pandas as pd

    shell, meta = gui
    # Bare trial numbers → no metadata → table hidden.
    meta.trials_widget.setup(pd.DataFrame({"trial": [1, 2, 3]}))
    assert meta.trials_widget.isHidden()
    # Real metadata columns → table shown.
    meta.trials_widget.setup(pd.DataFrame({"trial": [1, 2], "condition": ["a", "b"]}))
    assert not meta.trials_widget.isHidden()


def test_nav_tab_contains_trials_table(birdpark_gui):
    shell, meta = birdpark_gui
    # trials_widget (with its table) is parented into the Navigation section.
    assert meta.trials_widget.parent() is not None
    # Playback moved out of navigation into the video context panel.
    assert meta.context_panel.isAncestorOf(meta.navigation_widget.playback_group)


def test_video_context_shows_playback_and_pose(birdpark_gui):
    shell, meta = birdpark_gui
    meta.focus_video_context()
    assert meta.context_panel.current_context() == "video"
    assert meta.navigation_widget.playback_group.isVisibleTo(meta.context_panel)



def test_zen_mode_toggle(gui, qtbot):
    shell, meta = gui
    shell.show()
    qtbot.waitExposed(shell)
    shell.set_zen_mode(True)
    assert not shell._sidebar_dock.isVisible()
    assert meta.app_state.zen_mode is True
    shell.set_zen_mode(False)
    assert shell._sidebar_dock.isVisible()
    assert meta.app_state.zen_mode is False


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Left sidebar (drag-drop panel creation)
# ---------------------------------------------------------------------------


def test_allowed_plot_types_gating():
    from ethograph.gui.left_sidebar import allowed_plot_types

    class _State:
        ds = None  # feature_ncols falls back to 1

    audio = allowed_plot_types("audio", "mic1", _State())
    assert audio == ["Audio Trace", "Spectrogram Trace"]
    feat = allowed_plot_types("feature", "speed", _State())
    assert feat == ["Lineplot"]  # N==1 -> no heatmap/space


def test_left_sidebar_lists_sources_after_load(birdpark_gui):
    shell, meta = birdpark_gui
    meta.left_sidebar.refresh()
    lst = meta.left_sidebar._list
    texts = [lst.item(i).text().strip() for i in range(lst.count())]
    assert "Media" in texts and "Features" in texts
    # At least one draggable data item exists below the headers.
    assert len(texts) > 2


def test_drop_feature_creates_extra_lineplot(birdpark_gui):
    shell, meta = birdpark_gui
    feats = meta.plot_container._available_features()
    assert feats, "expected at least one feature in birdpark"
    n_before = len(meta.plot_container.extra_line_plots)
    meta._create_panel_for_source("feature", feats[0], "Lineplot")
    assert len(meta.plot_container.extra_line_plots) == n_before + 1


def test_extra_lineplot_behaves_like_main_lineplot(birdpark_gui):
    shell, meta = birdpark_gui
    pc = meta.plot_container
    feats = pc._available_features()
    plot = pc.add_extra_lineplot(feature=feats[0])
    assert plot is not None
    # No per-plot feature dropdown (simplified — same as the main line plot).
    assert not hasattr(plot, "_feature_combo")
    # Clicking it switches the sidebar to the lineplot context.
    meta.context_panel.set_context("audiotrace")
    pc._emit_feature_focus()
    assert meta.context_panel.current_context() in ("lineplot", "heatmap")


def test_cover_page_classify_files():
    from ethograph.gui.cover_page import classify_files

    buckets = classify_files(
        ["a.mp4", "b.h5", "c.wav", "d.dat", "s.nc", "labels.tsv", "junk.xyz"]
    )
    assert buckets["video"] == ["a.mp4"]
    assert buckets["pose"] == ["b.h5"]
    assert buckets["audio"] == ["c.wav"]
    assert buckets["ephys"] == ["d.dat"]
    assert buckets["session"] == ["s.nc"]
    assert buckets["labels"] == ["labels.tsv"]
    assert buckets["unknown"] == ["junk.xyz"]



def test_cover_page_builds_single_trial_alignment(gui, birdpark_data_dir):
    """Drag&drop path: build a real alignment.tmp.nwb from birdpark media."""
    from ethograph.gui.cover_page import CoverPage
    from ethograph.io.nwb_alignment import NWBAlignment

    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    videos = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    audio_only = AUDIO_EXTENSIONS - VIDEO_EXTENSIONS
    audios = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in audio_only)
    if not videos:
        pytest.skip("birdpark has no video files to build an alignment from")

    page = CoverPage(shell, meta.io_widget)
    cam_map = [(str(videos[0]), None)]
    audio_files = [str(audios[0])] if audios else []
    nwb_path = page._build_tmp_alignment(cam_map, audio_files)

    assert nwb_path.exists()
    align = NWBAlignment(nwb_path)  # must be a readable alignment NWB
    rate = align.get_stream_rate("video", "cam-1")
    assert rate and rate > 0  # real fps, not a hardcoded fallback


# ---------------------------------------------------------------------------
# End-to-end: load a template dataset, then drive a menu action
# ---------------------------------------------------------------------------



def test_birdpark_load_then_toggle_changepoints_via_menu(birdpark_gui):
    from qtpy.QtWidgets import QApplication

    shell, meta = birdpark_gui
    assert meta.app_state.ready

    cp_menu = _menu(shell, "Changepoints")
    action = _find_action(cp_menu, "Show changepoints")

    before = meta.app_state.show_changepoints
    action.trigger()
    QApplication.processEvents()
    assert meta.app_state.show_changepoints != before
