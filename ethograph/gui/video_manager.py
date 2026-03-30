<<<<<<< HEAD
"""Video layer lifecycle management — setup, teardown, camera switching, secondary video."""
=======
"""Video layer lifecycle management — setup, teardown, camera switching, multi-camera display."""
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

import os
from pathlib import Path

from napari._qt.qt_viewer import QtViewer
from napari.components.viewer_model import ViewerModel
<<<<<<< HEAD
from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QSplitter, QVBoxLayout, QWidget

from napari_pyav._reader import FastVideoReader

from .video_sync import NapariVideoSync

=======
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QSplitter, QVBoxLayout, QWidget

import numpy as np
from napari_pyav._reader import FastVideoReader

from .notify import notify
from .video_sync import NapariVideoSync

MAX_EXTRA_CAMERAS = 4

>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")

<<<<<<< HEAD
class SecondaryVideoWidget(QWidget):
    """Displays a second camera feed with pose overlay via a napari canvas."""
=======

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
            clamped = max(0, min(int(index), self._n - 1))
            return self._reader.read_frame(clamped + self._start)
        if isinstance(index, tuple) and len(np.r_[index]) == 1:
            clamped = max(0, min(int(np.r_[index][0]), self._n - 1))
            return self._reader.read_frame(clamped + self._start)[None]
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
    """Self-contained camera view with pose overlay via a napari canvas.

    Owns its own FPS — the mediator (VideoManager) broadcasts time in seconds
    and each widget converts to frames internally.
    """
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._viewer_model = ViewerModel()
        self._qt_viewer = QtViewer(self._viewer_model)
        layout.addWidget(self._qt_viewer)

        self._hide_dims_slider()

<<<<<<< HEAD
        self._video_layer = None
        self._points_layer = None

=======
        self._fps: float = 0.0
        self._video_layer = None
        self._points_layer = None

    @property
    def fps(self) -> float:
        return self._fps

>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    def _hide_dims_slider(self):
        from napari._qt.widgets.qt_dims import QtDims

        for widget in self._qt_viewer.findChildren(QtDims):
            widget.setVisible(False)

<<<<<<< HEAD
    def set_video(self, video_data):
        """Set the video source (a FastVideoReader or ndarray-like)."""
=======
    def set_video(self, video_data, fps: float = 0.0):
        self._fps = fps
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
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
<<<<<<< HEAD
            self._video_layer = self._viewer_model.add_image(video_data, name="video", rgb=True)
            self._hide_dims_slider()
            self.seek_to_frame(0)

    def set_pose_layer(self, data, properties, style_kwargs):
        """Add or replace the pose Points layer."""
        self.clear_pose()
        if data is not None and len(data) > 0:
            self._points_layer = self._viewer_model.add_points(
                data,
                properties=properties,
                **style_kwargs,
            )
=======
            try:
                self._video_layer = self._viewer_model.add_image(video_data, name="video", rgb=True)
            except StopIteration:
                notify("Video file could not be loaded (frame read failed).", "warning")
                return
            self._hide_dims_slider()
            self.seek_to_frame(0)

    def set_pose(self, data, properties, shown, style_kwargs):
        self.clear_pose()
        if data is None or len(data) == 0:
            return
        self._points_layer = self._viewer_model.add_points(
            data, properties=properties, shown=shown, **style_kwargs,
        )

    def seek_to_time(self, t_seconds: float):
        if self._fps <= 0 or self._video_layer is None:
            return
        frame = int(t_seconds * self._fps)
        n_frames = self._video_layer.data.shape[0]
        frame = max(0, min(frame, n_frames - 1))
        self._viewer_model.dims.set_point(0, frame)
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

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
<<<<<<< HEAD
    """Manages primary and secondary video layers, audio path resolution, and frame sync.

    Owns the video layer lifecycle on behalf of DataWidget. Does NOT own
    plot_container, labels, combos, or any UI controls — those stay in DataWidget.
=======
    """Manages primary and extra video layers, audio path resolution, and frame sync.

    Acts as a mediator: broadcasts time in seconds to extra cameras, each of
    which converts to frames internally using its own FPS.

    Supports up to MAX_EXTRA_CAMERAS additional camera views displayed in a
    vertical stack alongside the primary napari viewer.
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    """

    def __init__(self, viewer, app_state):
        self.viewer = viewer
        self.app_state = app_state
<<<<<<< HEAD
        self._secondary_widget: SecondaryVideoWidget | None = None
        self._secondary_fps: float = 0.0
        self._central_splitter: QSplitter | None = None
=======
        self._extra_widgets: dict[str, ExtraCameraWidget] = {}
        self._central_splitter: QSplitter | None = None
        self._extra_splitter: QSplitter | None = None
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
        self._original_central = None
        self._video_format_warned = False

    @property
<<<<<<< HEAD
    def secondary_widget(self) -> SecondaryVideoWidget | None:
        return self._secondary_widget

    @property
    def secondary_fps(self) -> float:
        return self._secondary_fps
=======
    def extra_widgets(self) -> dict[str, ExtraCameraWidget]:
        return self._extra_widgets
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

    def update_video(self, plot_container, transform_widget):
        if not self.app_state.ready:
            return
<<<<<<< HEAD
        camera_sel = getattr(self.app_state, 'cameras_sel', None)
        video_file = None
        if camera_sel:
            dt = self.app_state.dt

            video_file = dt.get_media(self.app_state.trials_sel, "video", device=camera_sel)
=======
        camera = self.app_state.primary_camera
        video_file = None
        if camera:
            dt = self.app_state.dt
            video_file = dt.get_media(self.app_state.trials_sel, "video", device=camera)
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
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
<<<<<<< HEAD
            show_warning(
                f"Video format '{ext}' may have inaccurate frame seeking. "
                f"See https://ethograph.readthedocs.io/en/latest/troubleshooting/"
=======
            notify(
                f"Video format '{ext}' may have inaccurate frame seeking. "
                f"See https://ethograph.readthedocs.io/en/latest/troubleshooting/",
                "warning",
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
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

<<<<<<< HEAD
        video_data = FastVideoReader(
            self.app_state.video_path, read_format='rgb24',
        )

        _ = video_data.shape
=======
        reader = FastVideoReader(
            self.app_state.video_path, read_format='rgb24',
        )
        _ = reader.shape

        detected_fps = float(reader.stream.guessed_rate) if reader.stream.guessed_rate else None
        if detected_fps is not None and self.app_state.dt is not None:
            camera = self.app_state.primary_camera
            if camera:
                self._store_camera_fps_in_session(camera, detected_fps)
            else:
                self.app_state.dt.set_video_fps(detected_fps)

        alignment = getattr(self.app_state, 'trial_alignment', None)
        video_time_offset = alignment.video_offset if alignment else 0.0

        # Slice the reader to the trial's time range so napari's slider is
        # bounded to trial frames.  This prevents StopIteration crashes when
        # codecs over-report nframes (common with AVI/MOV) and also enforces
        # the trial stop_time for session-wide videos with a non-zero offset.
        video_data = reader
        if alignment and alignment.trial_range:
            fps = self.app_state.video_fps
            trial_start_in_video = -video_time_offset
            start_frame = max(0, int(trial_start_in_video * fps))
            end_frame = int((trial_start_in_video + alignment.trial_range.duration) * fps)
            if start_frame > 0 or end_frame < reader.nframes:
                video_data = TrialVideoSlice(reader, start_frame, end_frame)
                video_time_offset = 0.0

>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
        n_frames = int(video_data.shape[0]) if len(video_data.shape) >= 1 else 0
        if n_frames > 0:
            restore_frame = min(restore_frame, n_frames - 1)
        else:
            restore_frame = 0

<<<<<<< HEAD
        video_layer = self.viewer.add_image(video_data, name="video", rgb=True)
=======
        try:
            video_layer = self.viewer.add_image(video_data, name="video", rgb=True)
        except StopIteration:
            notify("Video file could not be loaded (frame read failed).", "warning")
            return
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
        video_index = self.viewer.layers.index(video_layer)
        self.viewer.layers.move(video_index, 0)

        try:
<<<<<<< HEAD
            alignment = getattr(self.app_state, 'trial_alignment', None)
            video_time_offset = alignment.video_offset if alignment else 0.0
=======
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
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
<<<<<<< HEAD
            show_warning(f"Failed to initialize video sync: {e}")
=======
            notify(f"Failed to initialize video sync: {e}", "warning")
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
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
<<<<<<< HEAD
    # Secondary video
    # ------------------------------------------------------------------

    def show_secondary_video(self, video_path: str, layout_mgr, meta_widget):
        video_data = self._load_secondary_video_data(video_path)

        if self._secondary_widget is None:
            saved = layout_mgr.save_dock_widths()
            self._secondary_widget = SecondaryVideoWidget()
            qt_window = self.viewer.window._qt_window
            central = qt_window.centralWidget()
            self._central_splitter = QSplitter(Qt.Horizontal)
            self._central_splitter.setStretchFactor(0, 1)
            self._central_splitter.setStretchFactor(1, 1)
            self._original_central = central
            central.setParent(None)
            self._central_splitter.addWidget(central)
            self._central_splitter.addWidget(self._secondary_widget)
=======
    # Extra cameras
    # ------------------------------------------------------------------

    def add_camera(self, camera_name: str, video_path: str, layout_mgr, meta_widget):
        if camera_name in self._extra_widgets:
            self._update_existing_camera(camera_name, video_path)
            return

        if len(self._extra_widgets) >= MAX_EXTRA_CAMERAS:
            notify(f"Maximum {MAX_EXTRA_CAMERAS} extra cameras supported.", "warning")
            return

        reader = FastVideoReader(video_path, read_format='rgb24')
        _ = reader.shape
        fps = float(reader.stream.guessed_rate)
        self._store_camera_fps_in_session(camera_name, fps)

        widget = ExtraCameraWidget()
        widget.set_video(reader, fps=fps)
        self._extra_widgets[camera_name] = widget
        self._rebuild_camera_layout(layout_mgr, meta_widget)

        self._connect_extra_sync()
        self._sync_widget_to_current_time(widget)

    def _update_existing_camera(self, camera_name: str, video_path: str):
        reader = FastVideoReader(video_path, read_format='rgb24')
        _ = reader.shape
        fps = float(reader.stream.guessed_rate)
        self._store_camera_fps_in_session(camera_name, fps)
        widget = self._extra_widgets[camera_name]
        widget.set_video(reader, fps=fps)
        widget.show()
        self._sync_widget_to_current_time(widget)

    def _sync_widget_to_current_time(self, widget: ExtraCameraWidget):
        video = getattr(self.app_state, 'video', None)
        if video is not None:
            frame = self.viewer.dims.current_step[0]
            widget.seek_to_time(video.frame_to_time(frame))
        else:
            widget.seek_to_frame(0)

    def remove_camera(self, camera_name: str):
        widget = self._extra_widgets.pop(camera_name, None)
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
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
            qt_window.setCentralWidget(self._central_splitter)
            central.show()
            meta_widget.reapply_shortcuts()

            def _settle():
<<<<<<< HEAD
                self._equalize_video_split_now()
=======
                self._equalize_camera_split()
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
                layout_mgr.restore_dock_widths(saved)

            QTimer.singleShot(50, _settle)
        else:
<<<<<<< HEAD
            self._secondary_widget.show()

            def _settle():
                self._equalize_video_split_now()

            QTimer.singleShot(50, _settle)

        self._secondary_widget.set_video(video_data)

        self._connect_secondary_sync()
        self._secondary_widget.seek_to_frame(self.viewer.dims.current_step[0])

    def _load_secondary_video_data(self, video_path: str):
        video_data = FastVideoReader(video_path, read_format='rgb24')
        _ = video_data.shape
        self._secondary_fps = float(video_data.stream.guessed_rate)
        return video_data

    def hide_secondary_video(self):
        if self._secondary_widget is not None:
            self._disconnect_secondary_sync()
            self._secondary_widget.clear()
            self._secondary_widget.hide()

    def _equalize_video_split_now(self):
        if self._central_splitter is None:
            return
        total = self._central_splitter.width()
        self._central_splitter.setSizes([total // 2, total // 2])

    def _connect_secondary_sync(self):
        self._disconnect_secondary_sync()
        self.viewer.dims.events.current_step.connect(self._on_secondary_frame_sync)

    def _disconnect_secondary_sync(self):
        try:
            self.viewer.dims.events.current_step.disconnect(self._on_secondary_frame_sync)
        except (RuntimeError, TypeError):
            pass

    def _on_secondary_frame_sync(self, event=None):
        if self._secondary_widget is None or getattr(self.app_state, 'video', None) is None:
            return

        primary_fps = self.app_state.video_fps
        frame = self.viewer.dims.current_step[0]
        if abs(self._secondary_fps - primary_fps) < 0.01:
            self._secondary_widget.seek_to_frame(frame)
        else:
            self._secondary_widget.seek_to_frame(int(frame / primary_fps * self._secondary_fps))
            
    def cleanup(self):
        # Centralized cleanup for both primary and secondary video
=======
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
        frame = self.viewer.dims.current_step[0]
        t_seconds = self.app_state.video.frame_to_time(frame)
        for widget in self._extra_widgets.values():
            widget.seek_to_time(t_seconds)

    def _store_camera_fps_in_session(self, camera_name: str, fps: float):
        dt = getattr(self.app_state, 'dt', None)
        if dt is None:
            return
        sess = dt.session
        if sess is not None and "video_fps" in sess:
            da = sess["video_fps"]
            if "cameras" in da.dims and camera_name in da.coords["cameras"].values:
                da.loc[{"cameras": camera_name}] = fps

    def cleanup(self):
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
        if getattr(self.app_state, 'video', None):
            self.app_state.video.stop()
            self.app_state.video = None
        self._cleanup_primary_video()
<<<<<<< HEAD
        self.hide_secondary_video()
        self._secondary_widget = None
        self._central_splitter = None

=======
        self.remove_all_cameras()
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955

    def _resolve_video_path(self, camera_name: str, video_folder: str | None) -> str | None:
        if is_url(camera_name):
            return camera_name
        if video_folder:
<<<<<<< HEAD

=======
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
            video_file = self.app_state.dt.get_media(self.app_state.trials_sel, "video", device=camera_name)
            if video_file:
                path = os.path.normpath(os.path.join(video_folder, video_file))
                return path if os.path.isfile(path) else None
        return None
