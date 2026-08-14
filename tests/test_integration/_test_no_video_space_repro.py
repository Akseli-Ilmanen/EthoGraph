"""Ad-hoc repro: without a video folder, can space/radial plots be added?"""

import pytest
from qtpy.QtWidgets import QApplication

from ethograph.datasets import resolve_dataset_paths
from ethograph.gui.source_popup import allowed_plot_types
from tests.conftest import _ensure_alignment_nwb, _skip_if_not_downloaded


@pytest.fixture
def moll2025_no_video_gui(gui, qtbot):
    _skip_if_not_downloaded("moll2025")
    viewer, meta = gui
    _ensure_alignment_nwb("moll2025")
    resolved = resolve_dataset_paths("moll2025")

    io = meta.io_widget
    io._clear_all_line_edits()
    io.nc_file_path_edit.setText(resolved["nc_file_path"])
    meta.app_state.nc_file_path = resolved["nc_file_path"]
    # Deliberately NO video folder / audio folder / pose folder.
    meta.app_state.video_folder = None
    meta.app_state.audio_folder = None
    meta.app_state.pose_folder = None
    io.downsample_checkbox.setChecked(False)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    return viewer, meta


def test_no_video_space_and_radial(moll2025_no_video_gui):
    viewer, meta = moll2025_no_video_gui
    print("READY:", meta.app_state.ready)
    assert meta.app_state.ready

    catalog = meta.app_state.data_loader.catalog
    feats = catalog.feature_choices()
    print("FEATURES:", feats)

    for f in feats:
        print(f, "->", allowed_plot_types("feature", f, meta.app_state))

    # Try to add a space plot for a multi-column feature.
    target = None
    for f in feats:
        if "Space (2D)" in allowed_plot_types("feature", f, meta.app_state):
            target = f
            break
    print("SPACE TARGET:", target)
    assert target is not None, "no feature offers Space (2D)"

    n_before = len(meta.data_widget.space_plots)
    meta._create_panel_for_source("feature", target, "Space (2D)")
    QApplication.processEvents()
    print("SPACE PLOTS:", len(meta.data_widget.space_plots))
    assert len(meta.data_widget.space_plots) == n_before + 1
    sp = meta.data_widget.space_plots[-1]
    print("dock:", sp.dock_widget, "visible:", sp.dock_widget.isVisible() if sp.dock_widget else None)
    assert sp.dock_widget is not None

    # Radial via a stored heading, if the dataset has one.
    if "angles" in feats:
        opts = allowed_plot_types("feature", "angles", meta.app_state)
        print("ANGLES OPTS:", opts)
        assert "Radial" in opts
        meta._create_panel_for_source("feature", "angles", "Radial")
        QApplication.processEvents()
        print("RADIAL PLOTS:", len(meta.data_widget.radial_plots))
        assert meta.data_widget.radial_plots
