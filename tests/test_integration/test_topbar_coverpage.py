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


def test_every_panel_has_close_and_move_buttons(gui):
    from qtpy.QtWidgets import QPushButton

    shell, meta = gui
    pc = meta.plot_container
    # Every fixed panel is a dock with a slim title bar holding move + ✕
    # buttons (dynamic panels — incl. neo — get theirs per-instance in
    # add_panel). Fixed panels are the Phy trace + raster.
    assert set(pc._panel_docks) == {"ephys", "raster"}
    meta.app_state.has_audio = True
    plot = pc.add_audio_panel("audiotrace")
    dock = pc._dyn_docks[plot]
    assert not dock.isHidden()
    assert dock.titleBarWidget().findChild(QPushButton, "panel_move_btn") is not None
    close_btn = dock.titleBarWidget().findChild(QPushButton, "panel_close_btn")
    close_btn.click()
    assert plot not in pc.audio_trace_plots
    assert plot not in pc._dyn_docks


def test_top_bar_has_expected_menus(gui):
    shell, meta = gui
    titles = [a.text().replace("&", "") for a in shell.menuBar().actions()]
    assert titles == ["File", "Changepoints", "Neural", "Help"]


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


def test_io_subpanel_popups_are_separate(gui):
    """Import labels / Import predictions / Export labels each pop up alone.

    The popup hosts only its own sub-panel — none of the other sections —
    and the widget returns home on close.
    """
    shell, meta = gui
    io = meta.io_widget
    builder = shell._top_bar

    builder._popup_section("import_labels", "Import labels", io.labels_group)
    dlg = builder._open_popups["import_labels"]
    assert dlg.isAncestorOf(io.labels_group)
    assert not dlg.isAncestorOf(io.pred_group)  # predictions stay separate
    dlg.close()
    assert io.isAncestorOf(io.labels_group)

    builder._popup_section("import_predictions", "Import predictions", io.pred_group)
    dlg = builder._open_popups["import_predictions"]
    assert dlg.isAncestorOf(io.pred_group)
    assert not dlg.isAncestorOf(io.labels_group)
    dlg.close()
    assert io.isAncestorOf(io.pred_group)

    builder._popup_section("export_labels", "Export labels", io.export_panel)
    dlg = builder._open_popups["export_labels"]
    assert dlg.isAncestorOf(io.export_panel)
    dlg.close()
    assert io.isAncestorOf(io.export_panel)
    assert io.export_panel.isHidden()  # only shown while borrowed by its popup


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

    # Feature (lineplot) → xarray coords (feature dims), NOT spectrogram.
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
# Add-panel popup (drag-drop / Enter panel creation)
# ---------------------------------------------------------------------------


def test_allowed_plot_types_gating():
    from ethograph.gui.source_popup import allowed_plot_types

    class _State:
        ds = None  # feature_ncols falls back to 1

    audio = allowed_plot_types("audio", "mic1", _State())
    assert audio == ["Audio Trace", "Spectrogram Trace"]
    feat = allowed_plot_types("feature", "speed", _State())
    assert feat == ["Lineplot"]  # N==1 -> no heatmap/space


def test_add_panel_button_wired(gui):
    """The bottom bar's ➕ button opens the popup (guarded before a load)."""
    shell, meta = gui
    btn = shell.bottom_bar.add_panel_btn
    assert not meta.app_state.ready
    btn.click()  # no dataset yet → warning notify, popup stays hidden
    assert not meta.source_popup.isVisible()


def test_source_popup_lists_filters_and_navigates(qtbot):
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QKeyEvent

    from ethograph.gui.source_popup import SourcePopup

    class _Catalog:
        @staticmethod
        def feature_choices():
            return ["speed", "position", "beak_angle"]

    class _State:
        nwb_alignment = None
        ds = None

    popup = SourcePopup(_State())
    qtbot.addWidget(popup)
    popup.refresh(catalog=_Catalog())
    lst = popup._list
    texts = [lst.item(i).text().strip() for i in range(lst.count())]
    assert "Media" in texts and "Features" in texts
    # At least one draggable data item exists below the headers.
    assert len(texts) > 2

    # Filtering hides non-matching entries (and empty group headers).
    rows = popup._visible_data_rows()
    assert rows, "expected at least one source entry"
    target = lst.item(rows[0]).text().strip()
    popup._filter.setText(target)
    visible = [lst.item(i).text().strip() for i in popup._visible_data_rows()]
    assert target in visible
    popup._filter.setText("zzz-no-such-feature")
    assert popup._visible_data_rows() == []
    popup._filter.clear()
    assert popup._visible_data_rows() == rows

    # ↑/↓ typed in the filter box move the list selection (wrapping).
    popup._select_first_visible()
    down = QKeyEvent(QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier)
    popup.eventFilter(popup._filter, down)
    assert lst.currentRow() == rows[1 % len(rows)]
    up = QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier)
    popup.eventFilter(popup._filter, up)
    assert lst.currentRow() == rows[0]


def test_drop_feature_creates_lineplot(birdpark_gui):
    shell, meta = birdpark_gui
    feats = meta.plot_container._available_features()
    assert feats, "expected at least one feature in birdpark"
    n_before = len(meta.plot_container.line_plots)
    meta._create_panel_for_source("feature", feats[0], "Lineplot")
    assert len(meta.plot_container.line_plots) == n_before + 1


def test_clicking_feature_plot_sets_active_and_context(birdpark_gui):
    shell, meta = birdpark_gui
    pc = meta.plot_container
    p = pc.add_lineplot(feature=pc._available_features()[0])
    reg = meta.active_panels.registration_for(p)
    assert reg is not None  # every line plot auto-registers with the manager
    meta.active_panels.set_active(reg)
    assert pc.active_feature_plot is p
    assert meta.context_panel.current_context() in ("lineplot", "heatmap")


def test_lineplots_have_independent_state(birdpark_gui):
    shell, meta = birdpark_gui
    pc = meta.plot_container
    feats = pc._available_features()
    first = pc.line_plots[0]
    p = pc.add_lineplot(feature=feats[0])
    first_before = first._effective_feature()
    # Editing one plot's own state must not change any other line plot.
    pc.active_feature_plot = p
    target = feats[1] if len(feats) > 1 else feats[0]
    p.set_panel_control("features", target)
    p.set_panel_control("keypoint", "beakTip")
    assert p._effective_feature() == target
    assert p._effective_selections().get("keypoint") == "beakTip"
    assert first._effective_feature() == first_before  # peer untouched


def test_lineplot_axes_are_independent(birdpark_gui):
    shell, meta = birdpark_gui
    pc = meta.plot_container
    p = pc.add_lineplot(feature=pc._available_features()[0])
    pc.active_feature_plot = p
    ps = meta.plot_settings_widget
    ps.ymin_edit.setText("1.0")
    ps.ymax_edit.setText("5.0")
    ps._on_axes_edited()
    assert p.panel_state.get("ymin") == 1.0
    assert p.panel_state.get("ymax") == 5.0
    # Global axes (which drive the main plot) were not touched.
    assert getattr(meta.app_state, "ymin", None) != 1.0


def test_active_panel_gets_green_edge(birdpark_gui):
    shell, meta = birdpark_gui
    mgr = meta.active_panels
    pc = meta.plot_container
    p = pc.add_lineplot(feature=pc._available_features()[0])
    p_reg = mgr.registration_for(p)
    mgr.set_active(p_reg)
    assert mgr.active is p_reg
    assert "2ecc71" in p.styleSheet().lower()
    # Activating another panel moves the green edge.
    first = pc.line_plots[0]
    mgr.set_active(mgr.registration_for(first))
    assert "2ecc71" not in p.styleSheet().lower()
    assert "2ecc71" in first.styleSheet().lower()


def test_video_click_activates_green_edge_and_pose_playback(birdpark_gui):
    shell, meta = birdpark_gui
    mgr = meta.active_panels
    primary = shell.video_area.primary
    reg = mgr.registration_for(primary)
    assert reg is not None
    primary.clicked.emit()  # simulate a click on the video
    assert mgr.active is reg
    assert "2ecc71" in primary.styleSheet().lower()  # green edge
    assert meta.context_panel.current_context() == "video"  # pose + playback


def test_extra_camera_views_are_individual_docks(gui, qtbot):
    """Each extra camera view is its own closable shell dock — closing one
    view's ✕ removes only that view, never its siblings (even duplicates of
    the same camera)."""
    from qtpy.QtWidgets import QDockWidget

    shell, meta = gui
    area = shell.video_area

    v1 = area.add_extra("cam")
    v2 = area.add_extra("cam")  # duplicate view of the same camera
    v3 = area.add_extra("other")
    assert len(area.extras) == 3

    docks = [v.dock_widget for v in (v1, v2, v3)]
    assert all(isinstance(d, QDockWidget) for d in docks)
    assert len({d.objectName() for d in docks}) == 3  # unique per instance

    mgr = meta.active_panels
    assert mgr.registration_for(v2) is not None

    # Close ONE duplicate — the other view of the same camera must survive.
    v2.dock_widget.close()
    qtbot.waitUntil(lambda: len(area.extras) == 2, timeout=2000)
    assert v1 in area.extras.values()
    assert v3 in area.extras.values()
    assert mgr.registration_for(v2) is None
    assert mgr.registration_for(v1) is not None

    # Programmatic removal by camera name still removes every view of it.
    meta.data_widget.video_mgr.remove_camera("cam")
    assert list(area.extras.values()) == [v3]


def test_space_plot_registers_and_activates(moll2025_gui):
    shell, meta = moll2025_gui
    meta.app_state.space_plot_type = "Space Plot"
    meta.data_widget.update_space_plot()
    sp = meta.data_widget.space_plot
    assert sp is not None
    mgr = meta.active_panels
    reg = mgr.registration_for(sp)
    assert reg is not None  # space plot registered with the manager
    sp.clicked.emit()  # simulate a click
    assert mgr.active is reg
    assert "2ecc71" in sp.styleSheet().lower()  # green edge
    assert meta.context_panel.current_context() == "space"


def test_all_panel_types_registered(birdpark_gui):
    shell, meta = birdpark_gui
    mgr = meta.active_panels
    pc = meta.plot_container
    # Video, audio trace, spectrogram, line plots all register.
    for w in (pc.audio_trace_plot, pc.spectrogram_plot, pc.line_plots[0]):
        assert mgr.registration_for(w) is not None
    assert mgr.registration_for(shell.video_area.primary) is not None
    # Heatmaps are dynamic instances like every other panel: they register
    # on creation and unregister on removal.
    hm = pc.add_panel("heatmap")
    assert mgr.registration_for(hm) is not None
    pc.remove_panel(hm)
    assert mgr.registration_for(hm) is None


def test_panel_control_only_affects_active_plot(birdpark_gui):
    shell, meta = birdpark_gui
    dw = meta.data_widget
    pc = meta.plot_container
    feats = pc._available_features()
    p1 = pc.add_lineplot(feature=feats[0])
    p2 = pc.add_lineplot(feature=feats[0])
    # Any data-panel control routes only to the active plot (generic path).
    pc.active_feature_plot = p1
    dw.apply_panel_control("keypoint", "beakTip")
    assert p1._effective_selections().get("keypoint") == "beakTip"
    assert p2._effective_selections().get("keypoint") != "beakTip"


def test_added_lineplot_behaves_like_any_lineplot(birdpark_gui):
    shell, meta = birdpark_gui
    pc = meta.plot_container
    feats = pc._available_features()
    plot = pc.add_lineplot(feature=feats[0])
    assert plot is not None
    # No per-plot feature dropdown (all line plots share the sidebar controls).
    assert not hasattr(plot, "_feature_combo")
    # Clicking it switches the sidebar to the lineplot context.
    meta.context_panel.set_context("audiotrace")
    meta.active_panels.set_active(meta.active_panels.registration_for(plot))
    assert meta.context_panel.current_context() in ("lineplot", "heatmap")


def test_cover_page_borrows_and_returns_load_panel(gui, qtbot):
    """The IO load panel (path fields + Load) is hosted on the cover page
    while it is shown, and handed back to the IO tab when it closes."""
    from ethograph.gui.cover_page import CoverPage

    shell, meta = gui
    io = meta.io_widget
    page = CoverPage(shell, io)
    page.show()
    qtbot.waitExposed(page)
    assert page._load_host.isAncestorOf(io.load_panel)  # borrowed
    assert io.load_buttons_row.isHidden()  # no duplicate wizard/template buttons
    page.close()
    assert io.isAncestorOf(io.load_panel)  # returned
    assert not io.load_buttons_row.isHidden()


def test_cover_page_custom_load_accepts_page(gui, qtbot, monkeypatch):
    """Custom set-up path: clicking the shared Load bar's button closes
    (accepts) the cover page once the dataset is loaded — otherwise the
    modal dialog stays open forever and the main window never appears."""
    from qtpy.QtWidgets import QDialog

    from ethograph.gui.cover_page import CoverPage

    shell, meta = gui
    io = meta.io_widget
    page = CoverPage(shell, io)
    page.show()
    qtbot.waitExposed(page)
    assert io.load_button.isHidden()  # replaced by the shared Load bar

    monkeypatch.setattr(
        io.data_widget,
        "on_load_clicked",
        lambda: setattr(meta.app_state, "ready", True),
    )
    page._shared_load_btn.click()

    assert page.result() == QDialog.Accepted
    assert not page.isVisible()
    assert io.isAncestorOf(io.load_panel)  # panel handed back on close


def test_cover_page_classify_files():
    from ethograph.gui.cover_page import classify_files

    buckets = classify_files(["a.mp4", "b.h5", "c.wav", "d.dat", "s.nc", "labels.tsv", "arena.png", "junk.xyz"])
    assert buckets["video"] == ["a.mp4"]
    assert buckets["pose"] == ["b.h5"]
    assert buckets["audio"] == ["c.wav"]
    assert buckets["ephys"] == ["d.dat"]
    assert buckets["session"] == ["s.nc"]
    assert buckets["labels"] == ["labels.tsv"]
    assert buckets["image"] == ["arena.png"]
    assert buckets["unknown"] == ["junk.xyz"]


def test_cover_page_image_only_drop_rejected(gui):
    """An image alone has no time axis — the drop must fail with guidance."""
    from ethograph.gui.cover_page import CoverPage, classify_files

    shell, meta = gui
    page = CoverPage(shell, meta.io_widget)
    buckets = classify_files(["arena.png"])
    with pytest.raises(RuntimeError, match="no time axis"):
        page._populate_io_from_buckets(buckets, {"data_sr": None, "source_software": None, "pose_fps": None})


def test_cover_page_pose_without_image_or_video_rejected(gui):
    """A pose file needs either a video or a background image to display on."""
    from ethograph.gui.cover_page import CoverPage, classify_files

    shell, meta = gui
    page = CoverPage(shell, meta.io_widget)
    buckets = classify_files(["tracking.slp"])
    with pytest.raises(RuntimeError, match="background image"):
        page._populate_io_from_buckets(buckets, {"data_sr": None, "source_software": None, "pose_fps": 30.0})


def test_cover_page_builds_single_trial_alignment(gui, birdpark_data_dir):
    """Drag&drop path: build a real alignment.tmp.nwb from birdpark media."""
    from ethograph.gui.cover_page import CoverPage
    from ethograph.io.nwb_alignment import NWBAlignment

    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    videos = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    audio_only = AUDIO_EXTENSIONS - VIDEO_EXTENSIONS
    audios = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in audio_only)
    if not videos:
        pytest.skip("birdpark has no video files to build an alignment from")

    page = CoverPage(shell, meta.io_widget)
    page._drop_tmp_dir = page._prepare_drop_dir()
    cam_map = [(str(videos[0]), None)]
    audio_files = [str(audios[0])] if audios else []
    nwb_path = page._build_tmp_alignment(cam_map, audio_files)

    assert nwb_path.exists()
    align = NWBAlignment(nwb_path)  # must be a readable alignment NWB
    rate = align.get_stream_rate("video", "cam-1")
    assert rate and rate > 0  # real fps, not a hardcoded fallback


def test_cover_page_session_plus_media_builds_alignment(gui, birdpark_data_dir):
    """Dropping a session .nc together with media synthesises a tmp alignment
    (nwb_file_path override) so the media is loadable — the session file's
    folder has no .ethograph sidecar describing the dropped files."""
    from ethograph.gui.cover_page import CoverPage, classify_files
    from ethograph.io.nwb_alignment import NWBAlignment

    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    videos = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    session = next(iter(sorted(data_dir.glob("*.nc"))), None)
    if not videos or session is None:
        pytest.skip("birdpark is missing a video or .nc session file")

    page = CoverPage(shell, meta.io_widget)
    buckets = classify_files([str(session), str(videos[0])])
    page._populate_io_from_buckets(buckets, {"data_sr": None, "source_software": None})

    app_state = meta.app_state
    assert app_state.nc_file_path == str(session)
    assert app_state.nwb_file_path and app_state.nwb_file_path.endswith(".tmp.nwb")
    align = NWBAlignment(app_state.nwb_file_path)
    assert align.cameras == ["cam-1"]
    assert app_state.video_folder == str(videos[0].parent)


def test_cover_page_audio_only_alignment(gui, birdpark_data_dir):
    """Audio-only drops build an alignment too (duration from the audio file)."""
    from ethograph.gui.cover_page import CoverPage
    from ethograph.io.nwb_alignment import NWBAlignment

    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    audio_only = AUDIO_EXTENSIONS - VIDEO_EXTENSIONS
    audios = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in audio_only)
    if not audios:
        pytest.skip("birdpark has no audio-only files")

    page = CoverPage(shell, meta.io_widget)
    page._drop_tmp_dir = page._prepare_drop_dir()
    nwb_path = page._build_tmp_alignment([], [str(audios[0])])

    align = NWBAlignment(nwb_path)
    assert align.mics == ["mic-1"]
    assert align.cameras == []
    assert align.stop_time(1) and align.stop_time(1) > 0
    # Stream-based alignments have no trials-table filename columns —
    # the GUI must resolve audio via the ImageSeries external_file.
    assert align.get_media(1, "audio", "mic-1") is None
    resolved = align.resolve_media_path(1, "audio", device="mic-1")
    assert resolved and Path(resolved).name == audios[0].name


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
