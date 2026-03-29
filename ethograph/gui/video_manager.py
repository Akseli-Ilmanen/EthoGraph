"""Video layer lifecycle management — setup, teardown, camera switching, multi-camera display."""

import os
from pathlib import Path

from napari._qt.qt_viewer import QtViewer
from napari.components.viewer_model import ViewerModel
from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QSplitter, QVBoxLayout, QWidget

import numpy as np
from napari_pyav._reader import FastVideoReader

from .video_sync import NapariVideoSync

MAX_EXTRA_CAMERAS = 4


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


class TrialVideoSlice:
    """Wraps a FastVideoReader to expose only frames within a trial's time range.

    Napari sees ``shape[0]`` as ``end_frame - start_frame`` and the dims
    slider stays within the trial.  All frame indices are remapped so that
    index 0 corresponds to ``start_frame`` in the underlying reader.
    """

    def __init__(self, reader: FastVideoReader, start_frame: int, end_frame: int):
        self._reader = reader
        self._start = max(0, start_frame)
        self._end = min(end_frame, reader.nframes)
        self._n = max(0, self._end - self._start)

    @property
    def shape(self):
        base = self._reader.shape
        return (self._n, *base[1:])

    @property
    def ndim(self):
        return self._reader.ndim

    @property
    def dtype(self):
        return self._reader.dtype

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def stream(self):
        return self._reader.stream

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self._reader.read_frame(int(index) + self._start)
        if isinstance(index, tuple) and len(np.r_[index]) == 1:
            return self._reader.read_frame(np.r_[index][0] + self._start)[None]
        if isinstance(index, slice):
            frames = [self._reader.read_frame(i + self._start) for i in range(*index.indices(self._n))]
            return np.array(frames)
        raise NotImplementedError(f"Slicing of {type(index)}: {index} not implemented")

    def close(self):
        self._reader.close()

    @property
    def start_frame(self) -> int:
        return self._start

class ExtraCameraWidget(QWidget):
    """Displays an extra camera feed with pose overlay via a napari canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._viewer_model = ViewerModel()
        self._qt_viewer = QtViewer(self._viewer_model)
        layout.addWidget(self._qt_viewer)

        self._hide_dims_slider()

        self._video_layer = None
        self._points_layer = None

    def _hide_dims_slider(self):
        from napari._qt.widgets.qt_dims import QtDims

        for widget in self._qt_viewer.findChildren(QtDims):
            widget.setVisible(False)

    def set_video(self, video_data):
        if self._video_layer is not None:
            old_data = getattr(self._video_layer, "data", None)
            try:
                self._viewer_model.layers.remove(self._video_layer)
            except ValueError:
                pass
            self._video_layer = None
            if hasattr(old_data, "close"):
                try:
                    old_data.close()
                except Exception:
                    pass

        if video_data is not None:
            self._video_layer = self._viewer_model.add_image(video_data, name="video", rgb=True)
            self._hide_dims_slider()
            self.seek_to_frame(0)

    def set_pose_layer(self, data, properties, style_kwargs):
        self.clear_pose()
        if data is not None and len(data) > 0:
            self._points_layer = self._viewer_model.add_points(
                data,
                properties=properties,
                **style_kwargs,
            )

    def seek_to_frame(self, frame: int):
        n_frames = 0
        if self._video_layer is not None:
            shape = self._video_layer.data.shape
            n_frames = shape[0] if len(shape) >= 3 else 0
        if n_frames == 0:
            return
        frame = max(0, min(frame, n_frames - 1))
        self._viewer_model.dims.set_point(0, frame)

    def clear_pose(self):
        if self._points_layer is not None:
            try:
                self._viewer_model.layers.remove(self._points_layer)
            except ValueError:
                pass
            self._points_layer = None

    def clear(self):
        self.clear_pose()
        if self._video_layer is not None:
            old_data = getattr(self._video_layer, "data", None)
            try:
                self._viewer_model.layers.remove(self._video_layer)
            except ValueError:
                pass
            self._video_layer = None
            if hasattr(old_data, "close"):
                try:
                    old_data.close()
                except Exception:
                    pass


class VideoManager:
    """Manages primary and extra video layers, audio path resolution, and frame sync.

    Owns the video layer lifecycle on behalf of DataWidget. Does NOT own
    plot_container, labels, combos, or any UI controls — those stay in DataWidget.

    Supports up to MAX_EXTRA_CAMERAS additional camera views displayed in a
    vertical stack alongside the primary napari viewer.
    """

    def __init__(self, viewer, app_state):
        self.viewer = viewer
        self.app_state = app_state
        self._extra_widgets: dict[str, ExtraCameraWidget] = {}
        self._extra_fps: dict[str, float] = {}
        self._central_splitter: QSplitter | None = None
        self._extra_splitter: QSplitter | None = None
        self._original_central = None
        self._video_format_warned = False

    @property
    def extra_widgets(self) -> dict[str, ExtraCameraWidget]:
        return self._extra_widgets

    def get_camera_fps(self, camera_name: str) -> float:
        return self._extra_fps.get(camera_name, 0.0)

    def update_video(self, plot_container, transform_widget):
        if not self.app_state.ready:
            return
        camera_sel = getattr(self.app_state, 'cameras_sel', None)
        video_file = None
        if camera_sel:
            dt = self.app_state.dt

            video_file = dt.get_media(self.app_state.trials_sel, "video", device=camera_sel)
        if video_file and is_url(video_file):
            self.app_state.video_path = video_file
        elif video_file and self.app_state.video_folder:
            self.app_state.video_path = os.path.normpath(
                os.path.join(self.app_state.video_folder, video_file)
            )
        else:
            self.app_state.video_path = None
        if not self.app_state.video_path:
            return
        restore_frame = max(0, int(getattr(self.app_state, 'current_frame', 0) or 0))
        self._warn_video_format()
        self._cleanup_primary_video()
        self._setup_primary_video(restore_frame)

    def update_audio(self, plot_container, transform_widget):
        if not self.app_state.ready:
            return
        self._update_audio_path()
        self._update_audio_ui(plot_container, transform_widget)

    def _update_audio_path(self) -> None:
        self.app_state.audio_path = None
        if self.app_state.audio_folder and hasattr(self.app_state, 'mics_sel'):
            audio_path, _ = self.app_state.get_audio_source()
            if audio_path:
                self.app_state.audio_path = audio_path

    def _update_audio_ui(self, plot_container, transform_widget):
        has_audio = bool(self.app_state.audio_path)
        if transform_widget:
            transform_widget.set_enabled_state(has_audio=has_audio)
        for w in self._audio_row_widgets:
            w.setVisible(has_audio)
        if has_audio:
            plot_container.update_audio_panels()

    def set_audio_row_widgets(self, widgets):
        self._audio_row_widgets = widgets

    def _warn_video_format(self):
        video_path = self.app_state.video_path
        if not video_path or is_url(video_path):
            return
        ext = Path(video_path).suffix.lower()
        if ext in ('.avi', '.mov') and not self._video_format_warned:
            self._video_format_warned = True
            show_warning(
                f"Video format '{ext}' may have inaccurate frame seeking. "
                f"See https://ethograph.readthedocs.io/en/latest/troubleshooting/"
            )

    def _cleanup_primary_video(self):
        sync = getattr(self.app_state, 'video', None)
        if sync is not None:
            try:
                sync.frame_changed.disconnect(self._on_primary_frame_changed)
                sync.cleanup()
            except (RuntimeError, TypeError):
                pass
            self.app_state.video = None
        for layer in list(self.viewer.layers):
            if layer.name in ["video", "Video Stream", "video_new"]:
                old_data = getattr(layer, "data", None)
                self.viewer.layers.remove(layer)
                if hasattr(old_data, "close"):
                    try:
                        old_data.close()
                    except Exception:
                        pass

    def _setup_primary_video(self, restore_frame: int):

        reader = FastVideoReader(
            self.app_state.video_path, read_format='rgb24',
        )
        _ = reader.shape

        alignment = getattr(self.app_state, 'trial_alignment', None)
        video_time_offset = alignment.video_offset if alignment else 0.0

        # For session-wide videos, slice to the trial's time range so napari
        # only shows frames belonging to the current trial.
        video_data = reader
        if alignment and alignment.trial_range and video_time_offset != 0.0:
            fps = float(reader.stream.guessed_rate) if reader.stream.guessed_rate else 30.0
            trial_start_in_video = -video_time_offset
            start_frame = int(trial_start_in_video * fps)
            end_frame = int((trial_start_in_video + alignment.trial_range.duration) * fps)
            if start_frame > 0 or end_frame < reader.nframes:
                video_data = TrialVideoSlice(reader, start_frame, end_frame)
                video_time_offset = 0.0

        n_frames = int(video_data.shape[0]) if len(video_data.shape) >= 1 else 0
        if n_frames > 0:
            restore_frame = min(restore_frame, n_frames - 1)
        else:
            restore_frame = 0

        video_layer = self.viewer.add_image(video_data, name="video", rgb=True)
        video_index = self.viewer.layers.index(video_layer)
        self.viewer.layers.move(video_index, 0)

        try:
            sync = NapariVideoSync(
                viewer=self.viewer,
                app_state=self.app_state,
                video_source=self.app_state.video_path,
                audio_source=self.app_state.audio_path,
                video_layer=video_layer,
                time_offset=video_time_offset,
            )
            self.app_state.video = sync
            self.app_state.num_frames = sync.total_frames
        except (OSError, ValueError) as e:
            show_warning(f"Failed to initialize video sync: {e}")
            return

        sync.frame_changed.connect(self._on_primary_frame_changed)
        sync.seek_to_frame(restore_frame)
        self.app_state.current_frame = restore_frame

    def set_frame_changed_callback(self, callback):
        self._frame_changed_callback = callback

    def _on_primary_frame_changed(self, frame_number: int):
        self.app_state.current_frame = frame_number
        if hasattr(self, '_frame_changed_callback'):
            self._frame_changed_callback(frame_number)

    def toggle_pause_resume(self, plot_container):
        video = getattr(self.app_state, 'video', None)
        if video:
            video.toggle_pause_resume()
        else:
            plot_container.toggle_pause_resume()

    # ------------------------------------------------------------------
    # Extra cameras
    # ------------------------------------------------------------------

    def add_camera(self, camera_name: str, video_path: str, layout_mgr, meta_widget):
        if camera_name in self._extra_widgets:
            self._update_existing_camera(camera_name, video_path)
            return

        if len(self._extra_widgets) >= MAX_EXTRA_CAMERAS:
            show_warning(f"Maximum {MAX_EXTRA_CAMERAS} extra cameras supported.")
            return

        video_data = FastVideoReader(video_path, read_format='rgb24')
        _ = video_data.shape
        self._extra_fps[camera_name] = float(video_data.stream.guessed_rate)

        widget = ExtraCameraWidget()
        widget.set_video(video_data)
        self._extra_widgets[camera_name] = widget
        self._rebuild_camera_layout(layout_mgr, meta_widget)

        self._connect_extra_sync()
        widget.seek_to_frame(self.viewer.dims.current_step[0])

    def _update_existing_camera(self, camera_name: str, video_path: str):
        video_data = FastVideoReader(video_path, read_format='rgb24')
        _ = video_data.shape
        self._extra_fps[camera_name] = float(video_data.stream.guessed_rate)
        widget = self._extra_widgets[camera_name]
        widget.set_video(video_data)
        widget.show()
        widget.seek_to_frame(self.viewer.dims.current_step[0])

    def remove_camera(self, camera_name: str):
        widget = self._extra_widgets.pop(camera_name, None)
        self._extra_fps.pop(camera_name, None)
        if widget is None:
            return

        self._disconnect_extra_sync()
        widget.clear()
        widget.setParent(None)

        if not self._extra_widgets:
            self._teardown_camera_layout()
        elif self._extra_splitter is not None:
            self._equalize_extra_stack()

        if self._extra_widgets:
            self._connect_extra_sync()

    def remove_all_cameras(self):
        self._disconnect_extra_sync()
        for widget in self._extra_widgets.values():
            widget.clear()
            widget.setParent(None)
        self._extra_widgets.clear()
        self._extra_fps.clear()
        self._teardown_camera_layout()

    def _rebuild_camera_layout(self, layout_mgr, meta_widget):
        qt_window = self.viewer.window._qt_window

        if self._central_splitter is None:
            saved = layout_mgr.save_dock_widths()
            central = qt_window.centralWidget()
            self._original_central = central

            self._central_splitter = QSplitter(Qt.Horizontal)
            self._extra_splitter = QSplitter(Qt.Vertical)

            central.setParent(None)
            self._central_splitter.addWidget(central)
            self._central_splitter.addWidget(self._extra_splitter)
            qt_window.setCentralWidget(self._central_splitter)
            central.show()
            meta_widget.reapply_shortcuts()

            def _settle():
                self._equalize_camera_split()
                layout_mgr.restore_dock_widths(saved)

            QTimer.singleShot(50, _settle)
        else:
            while self._extra_splitter.count():
                w = self._extra_splitter.widget(0)
                w.setParent(None)

        for widget in self._extra_widgets.values():
            self._extra_splitter.addWidget(widget)
            widget.show()

        QTimer.singleShot(50, self._equalize_extra_stack)

    def _teardown_camera_layout(self):
        if self._central_splitter is None or self._original_central is None:
            return
        qt_window = self.viewer.window._qt_window
        self._original_central.setParent(None)
        qt_window.setCentralWidget(self._original_central)
        self._central_splitter = None
        self._extra_splitter = None
        self._original_central = None

    def _equalize_camera_split(self):
        if self._central_splitter is None:
            return
        total = self._central_splitter.width()
        self._central_splitter.setSizes([int(total * 0.6), int(total * 0.4)])
        self._equalize_extra_stack()

    def _equalize_extra_stack(self):
        if self._extra_splitter is None:
            return
        n = self._extra_splitter.count()
        if n > 0:
            h = self._extra_splitter.height()
            self._extra_splitter.setSizes([h // n] * n)

    def _connect_extra_sync(self):
        self._disconnect_extra_sync()
        if self._extra_widgets:
            self.viewer.dims.events.current_step.connect(self._on_extra_frame_sync)

    def _disconnect_extra_sync(self):
        try:
            self.viewer.dims.events.current_step.disconnect(self._on_extra_frame_sync)
        except (RuntimeError, TypeError):
            pass

    def _on_extra_frame_sync(self, event=None):
        if not self._extra_widgets or getattr(self.app_state, 'video', None) is None:
            return

        primary_fps = self.app_state.video_fps
        frame = self.viewer.dims.current_step[0]
        for camera_name, widget in self._extra_widgets.items():
            fps = self._extra_fps.get(camera_name, primary_fps)
            if abs(fps - primary_fps) < 0.01:
                widget.seek_to_frame(frame)
            else:
                widget.seek_to_frame(int(frame / primary_fps * fps))

    def cleanup(self):
        if getattr(self.app_state, 'video', None):
            self.app_state.video.stop()
            self.app_state.video = None
        self._cleanup_primary_video()
        self.remove_all_cameras()

    def _resolve_video_path(self, camera_name: str, video_folder: str | None) -> str | None:
        if is_url(camera_name):
            return camera_name
        if video_folder:
            video_file = self.app_state.dt.get_media(self.app_state.trials_sel, "video", device=camera_name)
            if video_file:
                path = os.path.normpath(os.path.join(video_folder, video_file))
                return path if os.path.isfile(path) else None
        return None
