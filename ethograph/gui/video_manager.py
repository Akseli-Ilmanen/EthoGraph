"""Video layer lifecycle management — setup, teardown, camera switching, multi-camera display."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from napari._qt.qt_viewer import QtViewer
from napari.components.viewer_model import ViewerModel
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QSplitter, QVBoxLayout, QWidget

import numpy as np
from napari_pyav._reader import FastVideoReader

from .notify import notify
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

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._viewer_model = ViewerModel()
        self._qt_viewer = QtViewer(self._viewer_model)
        layout.addWidget(self._qt_viewer)

        self._hide_dims_slider()

        self._fps: float = 0.0
        self._time_offset: float = 0.0
        self._video_layer = None
        self._points_layer = None
        self._shapes_layer = None

    @property
    def fps(self) -> float:
        return self._fps

    def _hide_dims_slider(self):
        from napari._qt.widgets.qt_dims import QtDims

        for widget in self._qt_viewer.findChildren(QtDims):
            widget.setVisible(False)

    def set_video(self, video_data, fps: float = 0.0, time_offset: float = 0.0):
        self._fps = fps
        self._time_offset = time_offset
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
        frame = int((t_seconds - self._time_offset) * self._fps)
        n_frames = self._video_layer.data.shape[0]
        frame = max(0, min(frame, n_frames - 1))
        self._viewer_model.dims.set_point(0, frame)

    def seek_to_frame(self, frame: int):
        n_frames = 0
        if self._video_layer is not None:
            shape = self._video_layer.data.shape
            n_frames = shape[0] if len(shape) >= 3 else 0
        if n_frames == 0:
            return
        frame = max(0, min(frame, n_frames - 1))
        self._viewer_model.dims.set_point(0, frame)

    def clear_bbox(self):
        if self._shapes_layer is not None:
            try:
                self._viewer_model.layers.remove(self._shapes_layer)
            except ValueError:
                pass
            self._shapes_layer = None

    def clear_pose(self):
        if self._points_layer is not None:
            try:
                self._viewer_model.layers.remove(self._points_layer)
            except ValueError:
                pass
            self._points_layer = None
        self.clear_bbox()

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

    Acts as a mediator: broadcasts time in seconds to extra cameras, each of
    which converts to frames internally using its own FPS.

    Supports up to MAX_EXTRA_CAMERAS additional camera views displayed in a
    vertical stack alongside the primary napari viewer.
    """

    def __init__(self, viewer, app_state):
        self.viewer = viewer
        self.app_state = app_state
        self._extra_widgets: dict[str, ExtraCameraWidget] = {}
        self._central_splitter: QSplitter | None = None
        self._extra_splitter: QSplitter | None = None
        self._original_central = None
        self._video_format_warned = False

    @property
    def extra_widgets(self) -> dict[str, ExtraCameraWidget]:
        return self._extra_widgets

    def update_video(self, plot_container):
        if not self.app_state.ready:
            return
        camera = self.app_state.primary_camera
        sio = getattr(self.app_state, 'nwb_alignment', None)
        if camera and sio is not None:
            self.app_state.video_path = sio.resolve_media_path(
                self.app_state.trials_sel, "video", device=camera,
                fallback_folder=self.app_state.video_folder,
            )
        else:
            self.app_state.video_path = None
        if not self.app_state.video_path:
            return
        restore_frame = max(0, int(getattr(self.app_state, 'current_frame', 0) or 0))
        self._warn_video_format()
        self._cleanup_primary_video()
        self._setup_primary_video(restore_frame)

    def update_audio(self, plot_container):
        if not self.app_state.ready:
            return
        self._update_audio_path()
        self._update_audio_ui(plot_container)

    def _update_audio_path(self) -> None:
        self.app_state.audio_path = None
        if self.app_state.audio_folder and hasattr(self.app_state, 'mics_sel'):
            audio_path, _ = self.app_state.get_audio_source()
            if audio_path:
                self.app_state.audio_path = audio_path

    def _update_audio_ui(self, plot_container):
        has_audio = bool(self.app_state.audio_path)
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
            notify(
                f"Video format '{ext}' may have inaccurate frame seeking. "
                
                f"See https://akseli-ilmanen.github.io/ethograph/user_guide/troubleshooting.html",
                "warning",
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

        detected_fps = float(reader.stream.guessed_rate) if reader.stream.guessed_rate else None
        if detected_fps is not None and self.app_state.dt is not None:
            camera = self.app_state.primary_camera
            self.app_state.nwb_alignment.set_stream_rate(detected_fps, "video", camera)

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

        n_frames = int(video_data.shape[0]) if len(video_data.shape) >= 1 else 0
        if n_frames > 0:
            restore_frame = min(restore_frame, n_frames - 1)
        else:
            restore_frame = 0

        try:
            video_layer = self.viewer.add_image(video_data, name="video", rgb=True)
        except StopIteration:
            notify("Video file could not be loaded (frame read failed).", "warning")
            return
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
            notify(f"Failed to initialize video sync: {e}", "warning")
            return

        sync.frame_changed.connect(self._on_primary_frame_changed)
        sync.seek_to_frame(restore_frame)
        self.app_state.current_frame = restore_frame

        qt_dims = getattr(self.viewer.window, "_qt_viewer", self.viewer.window.qt_viewer).dims
        if qt_dims.slider_widgets:
            qt_dims.slider_widgets[0].play_button.hide()

    def set_frame_changed_callback(self, callback):
        self._frame_changed_callback = callback

    def _on_primary_frame_changed(self, frame_number: int):
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

    def add_camera(self, camera_name: str, video_path: str, layout_mgr, meta_widget, *, reader=None):
        if camera_name in self._extra_widgets:
            self._update_existing_camera(camera_name, video_path, reader=reader)
            return

        if len(self._extra_widgets) >= MAX_EXTRA_CAMERAS:
            notify(f"Maximum {MAX_EXTRA_CAMERAS} extra cameras supported.", "warning")
            return

        if reader is None:
            reader = FastVideoReader(video_path, read_format='rgb24')
            _ = reader.shape
        fps = float(reader.stream.guessed_rate)
        self._store_camera_fps_in_session(camera_name, fps)

        video_data, time_offset = self._prepare_extra_video(reader, fps, camera_name)

        widget = ExtraCameraWidget()
        widget.set_video(video_data, fps=fps, time_offset=time_offset)
        self._extra_widgets[camera_name] = widget
        self._rebuild_camera_layout(layout_mgr, meta_widget)

        self._connect_extra_sync()
        self._sync_widget_to_current_time(widget)

    def _update_existing_camera(self, camera_name: str, video_path: str, *, reader=None):
        if reader is None:
            reader = FastVideoReader(video_path, read_format='rgb24')
            _ = reader.shape
        fps = float(reader.stream.guessed_rate)
        self._store_camera_fps_in_session(camera_name, fps)
        video_data, time_offset = self._prepare_extra_video(reader, fps, camera_name)
        widget = self._extra_widgets[camera_name]
        widget.set_video(video_data, fps=fps, time_offset=time_offset)
        widget.show()
        self._sync_widget_to_current_time(widget)

    def _prepare_extra_video(self, reader, fps: float, camera_name: str):
        """Retrieve per-camera offset and apply trial slicing for an extra camera."""
        sio = getattr(self.app_state, 'nwb_alignment', None)
        time_offset = 0.0
        if sio is not None:
            trial_id = self.app_state.trials_sel
            time_offset = sio.stream_offset_for_trial(trial_id, "video", camera_name)

        alignment = getattr(self.app_state, 'trial_alignment', None)
        video_data = reader
        if alignment and alignment.trial_range:
            trial_start_in_video = -time_offset
            start_frame = max(0, int(trial_start_in_video * fps))
            end_frame = int((trial_start_in_video + alignment.trial_range.duration) * fps)
            if start_frame > 0 or end_frame < reader.nframes:
                video_data = TrialVideoSlice(reader, start_frame, end_frame)
                time_offset = 0.0

        return video_data, time_offset

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
        frame = self.viewer.dims.current_step[0]
        t_seconds = self.app_state.video.frame_to_time(frame)
        for widget in self._extra_widgets.values():
            widget.seek_to_time(t_seconds)

    def _store_camera_fps_in_session(self, camera_name: str, fps: float):
        sio = getattr(self.app_state, 'nwb_alignment', None)
        if sio is None:
            return
        sio.set_stream_rate(fps, "video", camera_name)

    def cleanup(self):
        if getattr(self.app_state, 'video', None):
            self.app_state.video.stop()
            self.app_state.video = None
        self._cleanup_primary_video()
        self.remove_all_cameras()

    @staticmethod
    def open_readers_parallel(paths: dict[str, str]) -> dict[str, FastVideoReader]:
        """Open FastVideoReaders for *paths* concurrently.

        Parameters
        ----------
        paths
            ``{camera_name: video_path}`` mapping.

        Returns
        -------
        dict
            ``{camera_name: reader}`` for every path that opened
            successfully.  Failed opens are silently skipped.
        """
        def _open(video_path: str) -> FastVideoReader | None:
            try:
                reader = FastVideoReader(video_path, read_format="rgb24")
                _ = reader.shape
                return reader
            except Exception:
                return None

        if not paths:
            return {}

        results: dict[str, FastVideoReader] = {}
        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            futures = {name: pool.submit(_open, path) for name, path in paths.items()}
            for name, future in futures.items():
                reader = future.result()
                if reader is not None:
                    results[name] = reader
        return results

    def _resolve_video_path(self, camera_name: str, video_folder: str | None) -> str | None:
        if is_url(camera_name):
            return camera_name
        return self.app_state.nwb_alignment.resolve_media_path(
            self.app_state.trials_sel, "video", device=camera_name,
            fallback_folder=video_folder,
        )
