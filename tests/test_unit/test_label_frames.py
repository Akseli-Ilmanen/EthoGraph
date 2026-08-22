"""Pure logic of the label-frames grid (Tools ▸ Labels: Show frames as PDF…)."""

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QApplication, QCheckBox, QLabel, QListWidget, QListWidgetItem, QWidget

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_label_frames import (
    FrameEntry,
    LabelFramesConfigDialog,
    LabelFramesGridDialog,
    _draw_disc,
    _entry_info,
    _entry_title,
    allowed_trials_from_metadata,
    build_frame_entries,
    capture_panel_images,
    crop_thumbnail,
    decode_entry_images,
    draw_pose_points,
    open_gui_panels,
)
from ethograph.gui.pose_render import PoseRenderData

MAPPINGS = {
    1: {"name": "peck", "event_type": "point", "color": (1.0, 0.0, 0.0)},
    2: {"name": "hop", "event_type": "state", "color": (0.0, 1.0, 0.0)},
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def labels_df():
    return pd.DataFrame(
        {
            "trial": [1, 1, 2, 3],
            "labels": [1, 2, 2, 1],
            "onset_s": [0.5, 1.0, 2.0, 3.0],
            "offset_s": [np.nan, 1.8, 2.6, np.nan],
            "individual": ["a", "a", "b", "a"],
            "individual_rec": ["", "", "", ""],
        }
    )


class TestBuildFrameEntries:
    def test_point_one_entry_state_two(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1, 2], [None])
        by_label = {}
        for e in entries:
            by_label.setdefault(e.label_id, []).append(e)
        # 2 point rows -> 1 entry each; 2 state rows -> 2 entries each.
        assert len(by_label[1]) == 2
        assert len(by_label[2]) == 4
        assert all(e.boundary == "point" for e in by_label[1])
        assert [e.boundary for e in by_label[2]] == ["start", "end", "start", "end"]

    def test_state_times_are_onset_and_offset(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [2], [None])
        start, end = entries[0], entries[1]
        assert start.t_rel == 1.0 and end.t_rel == 1.8
        assert start.onset_s == end.onset_s == 1.0

    def test_label_filter(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1], [None])
        assert {e.label_id for e in entries} == {1}

    def test_cameras_multiply_entries(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1], ["cam-1", "cam-2"])
        assert len(entries) == 4
        assert {e.camera for e in entries} == {"cam-1", "cam-2"}

    def test_allowed_trials_filter(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1, 2], [None], allowed_trials={"1"})
        assert {str(e.trial) for e in entries} == {"1"}

    def test_unmapped_label_falls_back_to_state(self, labels_df):
        df = labels_df.assign(labels=[7, 7, 7, 7])
        entries = build_frame_entries(df, MAPPINGS, [7], [None])
        # Finite-offset rows become state pairs; NaN offsets stay points.
        assert [e.boundary for e in entries if str(e.trial) == "1"] == ["point", "start", "end"]

    def test_empty_df(self):
        assert build_frame_entries(pd.DataFrame(), MAPPINGS, [1], [None]) == []
        assert build_frame_entries(None, MAPPINGS, [1], [None]) == []

    def test_mapping_color_carried(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1], [None])
        assert entries[0].color_hex == "#ff0000"


class TestAllowedTrials:
    def test_no_filters_means_none(self):
        mdf = pd.DataFrame({"trial": [1, 2], "genotype": ["wt", "ko"]})
        assert allowed_trials_from_metadata(mdf, {}) is None
        assert allowed_trials_from_metadata(mdf, {"genotype": set()}) is None
        assert allowed_trials_from_metadata(None, {"genotype": {"wt"}}) is None

    def test_single_column(self):
        mdf = pd.DataFrame({"trial": [1, 2, 3], "genotype": ["wt", "ko", "wt"]})
        assert allowed_trials_from_metadata(mdf, {"genotype": {"wt"}}) == {"1", "3"}

    def test_columns_intersect(self):
        mdf = pd.DataFrame(
            {
                "trial": [1, 2, 3],
                "genotype": ["wt", "wt", "ko"],
                "treatment": ["sal", "drug", "sal"],
            }
        )
        allowed = allowed_trials_from_metadata(mdf, {"genotype": {"wt"}, "treatment": {"sal"}})
        assert allowed == {"1"}


class TestTitles:
    def test_point_has_no_boundary_suffix(self):
        entry = FrameEntry(
            trial=2,
            camera="cam-1",
            label_id=1,
            name="peck",
            event_type="point",
            boundary="point",
            t_rel=0.5,
            onset_s=0.5,
            offset_s=float("nan"),
            individual="a",
        )
        assert _entry_title(entry) == "peck (1)"
        assert _entry_info(entry) == "trial 2  ·  cam-1  ·  a  ·  0.500 s"

    def test_state_names_boundary(self):
        entry = FrameEntry(
            trial=1,
            camera=None,
            label_id=2,
            name="hop",
            event_type="state",
            boundary="end",
            t_rel=1.8,
            onset_s=1.0,
            offset_s=1.8,
        )
        assert _entry_title(entry) == "hop (2) — END"
        assert _entry_info(entry) == "trial 1  ·  1.800 s"

    def test_cropped_entry_says_so(self):
        entry = FrameEntry(
            trial=1,
            camera="cam-1",
            label_id=1,
            name="peck",
            event_type="point",
            boundary="point",
            t_rel=0.5,
            onset_s=0.5,
            offset_s=float("nan"),
            cropped=True,
        )
        assert _entry_info(entry) == "trial 1  ·  cam-1  ·  0.500 s  ·  cropped"


class TestCropThumbnail:
    def test_crop_slices_source_pixels(self):
        image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        out = crop_thumbnail(image, (4, 2, 10, 8), scale=1.0)
        assert out.shape == (6, 6, 3)
        assert np.array_equal(out, image[2:8, 4:10])

    def test_crop_rect_divided_by_decode_scale(self):
        # Source 40x60 decoded at half size: rect in source pixels lands halved.
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        out = crop_thumbnail(image, (10, 4, 30, 16), scale=2.0)
        assert out.shape == (6, 10, 3)

    def test_crop_clamps_to_image(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        out = crop_thumbnail(image, (-5, -5, 100, 100), scale=1.0)
        assert out.shape == (20, 30, 3)

    def test_degenerate_crop_returns_unchanged(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        out = crop_thumbnail(image, (10, 10, 10, 10), scale=1.0)
        assert out.shape == (20, 30, 3)


class _LabelsStub(QWidget):
    def __init__(self, mappings):
        super().__init__()
        self._mappings = mappings


class _Meta:
    def __init__(self, app_state, labels_widget, nav=None, data_widget=None):
        self.app_state = app_state
        self.labels_widget = labels_widget
        self.navigation_widget = nav
        self.data_widget = data_widget


class TestConfigDialog:
    @pytest.fixture()
    def dialog(self, qapp, tmp_path, labels_df):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        state.metadata_df = pd.DataFrame({"trial": [1, 2, 3], "genotype": ["wt", "ko", "wt"]})
        dlg = LabelFramesConfigDialog(_Meta(state, _LabelsStub({0: {"name": "none"}, **MAPPINGS})))
        yield dlg
        if dlg._grid_dialog is not None:
            dlg._grid_dialog.close()
        dlg.close()

    def test_label_list_skips_background(self, dialog):
        ids = [dialog.label_list.item(i).data(Qt.UserRole) for i in range(dialog.label_list.count())]
        assert ids == [1, 2]

    def test_metadata_columns_get_filter_buttons(self, dialog):
        assert set(dialog._filter_buttons) == {"genotype"}

    def test_generate_opens_grid_with_placeholder_tiles(self, dialog):
        """No resolvable video (EmpytAlignment) → every tile carries an error,
        and the grid still opens so the user sees what went wrong."""
        dialog.label_list.item(0).setCheckState(Qt.Checked)  # label 1: two points
        dialog._filters["genotype"] = {"wt"}  # trials 1 + 3
        dialog._generate()
        grid = dialog._grid_dialog
        assert grid is not None
        assert [str(e.trial) for e in grid._entries] == ["1", "3"]
        assert all(e.image is None and e.error == "video not found" for e in grid._entries)
        assert len(grid._cells) == 2


def _write_video(path, n_frames=20, fps=10, size=32):
    """Synthetic video whose frame *i* is a uniform gray of value ``i * 10``."""
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(np.full((size, size, 3), i * 10, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


class _AlignmentStub:
    """Per-camera video paths; no rates, no offsets, no pose."""

    def __init__(self, paths):
        self.paths = paths

    def resolve_media_path(self, trial, stream, device=None, fallback_folder=None):
        return str(self.paths[device]) if stream == "video" and device in self.paths else None

    def get_stream_rate(self, stream, device=None):
        return None

    def stream_offset_for_trial(self, trial, stream, device=None):
        return 0.0


class TestDecodeEntryImages:
    def test_two_cameras_decode_in_parallel_groups(self, qapp, tmp_path):
        paths = {f"cam-{k}": _write_video(tmp_path / f"cam{k}.mp4") for k in (1, 2)}
        entries = [_point_entry(1, "cam-1", 0.5), _point_entry(1, "cam-2", 1.2)]
        decode_entry_images(
            entries,
            alignment=_AlignmentStub(paths),
            video_folder=None,
            pose_folder=None,
            source_software=None,
            pose_color_by="keypoint",
            camera_crops={"cam-2": (0, 0, 16, 16)},
        )
        assert [e.error for e in entries] == [None, None]
        # fps 10, offset 0: t=0.5 → frame 5 (gray 50), t=1.2 → frame 12 (gray 120).
        assert entries[0].frame_idx == 5 and entries[1].frame_idx == 12
        assert abs(float(entries[0].image.mean()) - 50) < 15
        assert abs(float(entries[1].image.mean()) - 120) < 15
        assert entries[0].image.shape == (32, 32, 3)
        assert entries[1].cropped and entries[1].image.shape == (16, 16, 3)

    def test_unresolvable_video_marks_errors_without_raising(self, qapp):
        entries = [_point_entry(1, "cam-1", 0.5)]
        decode_entry_images(
            entries,
            alignment=_AlignmentStub({}),
            video_folder=None,
            pose_folder=None,
            source_software=None,
            pose_color_by="keypoint",
        )
        assert entries[0].image is None and entries[0].error == "video not found"


class _NavStub:
    def __init__(self):
        self.jumps = []

    def jump_to_label_instance(self, inst, **kwargs):
        self.jumps.append((inst, kwargs))


def _point_entry(trial, camera, t_rel):
    return FrameEntry(
        trial=trial,
        camera=camera,
        label_id=1,
        name="peck",
        event_type="point",
        boundary="point",
        t_rel=t_rel,
        onset_s=t_rel,
        offset_s=float("nan"),
    )


class TestPanelCapture:
    def test_no_open_panels_for_stub_meta(self):
        assert open_gui_panels(_Meta(None, None)) == []

    def test_cameras_share_one_capture(self, qapp):
        """Two cameras of the same boundary → one GUI jump, shared shots."""
        panel = QLabel("plot")
        panel.setFixedSize(60, 40)
        entries = [_point_entry(1, "cam-1", 2.0), _point_entry(1, "cam-2", 2.0)]
        nav = _NavStub()
        visited = capture_panel_images(entries, nav=nav, panels=[("Lineplot", panel)], window_s=4.0)
        assert visited == 1
        assert len(nav.jumps) == 1
        inst, kwargs = nav.jumps[0]
        assert kwargs["seek_rel"] == 2.0
        assert kwargs["view_rel"].start_s == 0.0 and kwargs["view_rel"].end_s == 4.0
        assert entries[0].panels and entries[0].panels is entries[1].panels
        title, image = entries[0].panels[0]
        assert title == "Lineplot"
        assert isinstance(image, QImage) and not image.isNull()

    def test_y_mode_lock_freezes_and_restores_autorange(self, qapp):
        """'lock' disables y autorange for the capture and puts the panel's
        own view state back afterwards; 'autoscale' does the reverse."""
        plot = pg.PlotWidget()
        plot.plot([0.0, 1.0, 2.0], [0.0, 5.0, 1.0])
        vb = plot.getViewBox()
        vb.enableAutoRange(y=True)  # the user's own setting

        seen = {}

        class _AxisNav(_NavStub):
            def jump_to_label_instance(self, inst, **kwargs):
                seen["auto_y"] = bool(vb.autoRangeEnabled()[1])
                super().jump_to_label_instance(inst, **kwargs)

        capture_panel_images(
            [_point_entry(1, "cam-1", 2.0)], nav=_AxisNav(), panels=[("P", plot)], window_s=1.0, y_mode="lock"
        )
        assert seen["auto_y"] is False
        assert bool(vb.autoRangeEnabled()[1]) is True  # restored

        vb.enableAutoRange(y=False)  # now the user has a manual range
        capture_panel_images(
            [_point_entry(1, "cam-1", 2.0)], nav=_AxisNav(), panels=[("P", plot)], window_s=1.0, y_mode="autoscale"
        )
        assert seen["auto_y"] is True
        assert bool(vb.autoRangeEnabled()[1]) is False  # restored
        plot.close()

    def test_skip_video_suppresses_loading_during_capture(self, qapp, tmp_path):
        """With 'skip video' ticked, trial jumps run with suppress_video_load
        set, and the media reloads exactly once afterwards."""

        class _VideoMgrStub:
            def __init__(self):
                self.synced = 0

            def sync_proxies(self):
                self.synced += 1

        class _DataWidgetStub:
            def __init__(self):
                self.suppress_video_load = False
                self.video_mgr = _VideoMgrStub()
                self.calls = []

            def update_video(self):
                self.calls.append("video")

            def _init_or_update_extra_cameras(self):
                self.calls.append("extras")

            def update_pose(self):
                self.calls.append("pose")

        class _FlagNav(_NavStub):
            def __init__(self, data_widget):
                super().__init__()
                self.data_widget = data_widget
                self.flags = []

            def jump_to_label_instance(self, inst, **kwargs):
                self.flags.append(self.data_widget.suppress_video_load)
                super().jump_to_label_instance(inst, **kwargs)

        data_widget = _DataWidgetStub()
        nav = _FlagNav(data_widget)
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        dlg = LabelFramesConfigDialog(_Meta(state, _LabelsStub(MAPPINGS), nav=nav, data_widget=data_widget))
        try:
            panel = QLabel("plot")
            panel.setFixedSize(40, 30)
            dlg.panel_list = QListWidget()
            item = QListWidgetItem("Lineplot")
            item.setData(Qt.UserRole, panel)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            dlg.panel_list.addItem(item)
            dlg.skip_video_cb = QCheckBox()
            dlg.skip_video_cb.setChecked(True)
            dlg.window_spin = None

            entries = [_point_entry(1, "cam-1", 2.0)]
            dlg._capture_panels(entries)

            assert nav.flags == [True]  # suppressed while jumping to the seed
            assert data_widget.suppress_video_load is False
            assert data_widget.calls == ["video", "extras", "pose"]
            assert data_widget.video_mgr.synced == 1
            assert entries[0].panels
        finally:
            dlg.close()

    def test_grid_cell_stacks_panel_snapshots(self, qapp):
        entry = _point_entry(1, "cam-1", 2.0)
        entry.image = np.zeros((20, 30, 3), dtype=np.uint8)
        entry.panels = [("Lineplot — speed", QImage(40, 20, QImage.Format_RGB888))]
        dlg = LabelFramesGridDialog(_Meta(ObservableAppState(), None), [entry])
        try:
            # Frame pixmap + one panel pixmap, both rescalable on relayout.
            assert len(dlg._cells[0]._pix_labels) == 2
        finally:
            dlg.close()


class TestPoseDrawing:
    def test_draw_disc_clips_at_edges(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        _draw_disc(image, 0, 0, 2, (255, 0, 0))
        assert image[0, 0, 0] == 255
        _draw_disc(image, 50, 50, 2, (255, 0, 0))  # fully outside: no raise

    def test_draw_pose_points_only_target_frame(self):
        # rows: [frame, y, x] — frame 3 lands inside, frame 4 does not.
        data = np.array([[3.0, 5.0, 5.0], [4.0, 20.0, 20.0]])
        pose = PoseRenderData(
            data=data,
            properties=pd.DataFrame({"keypoint": ["beak", "beak"], "individual": ["a", "a"]}),
            data_not_nan=np.array([True, True]),
            file_name="pose.nc",
        )
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        draw_pose_points(image, pose, frame_idx=3, scale=1.0, color_by="keypoint")
        assert image[5, 5].any()
        assert not image[20, 20].any()

    def test_scale_divides_coordinates(self):
        data = np.array([[0.0, 10.0, 10.0]])
        pose = PoseRenderData(
            data=data,
            properties=pd.DataFrame({"keypoint": ["beak"]}),
            data_not_nan=np.array([True]),
            file_name="pose.nc",
        )
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        draw_pose_points(image, pose, frame_idx=0, scale=2.0, color_by="keypoint")
        assert image[5, 5].any()
        assert not image[10, 10].any()

    def test_nan_rows_skipped(self):
        data = np.array([[0.0, np.nan, np.nan]])
        pose = PoseRenderData(
            data=data,
            properties=pd.DataFrame({"keypoint": ["beak"]}),
            data_not_nan=np.array([False]),
            file_name="pose.nc",
        )
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        draw_pose_points(image, pose, frame_idx=0, scale=1.0, color_by="keypoint")
        assert not image.any()
