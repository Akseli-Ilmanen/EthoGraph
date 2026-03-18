"""Video layer lifecycle management — setup, teardown, camera switching, secondary video."""

import os
from pathlib import Path

from napari._qt.qt_viewer import QtViewer
from napari.components.viewer_model import ViewerModel
from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QSplitter, QVBoxLayout, QWidget

from napari_pyav._reader import FastVideoReader

from .video_sync import NapariVideoSync


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")

class SecondaryVideoWidget(QWidget):
    """Displays a second camera feed with pose overlay via a napari canvas."""

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
        """Set the video source (a FastVideoReader or ndarray-like)."""
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
        """Add or replace the pose Points layer."""
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
    """Manages primary and secondary video layers, audio path resolution, and frame sync.

    Owns the video layer lifecycle on behalf of DataWidget. Does NOT own
    plot_container, labels, combos, or any UI controls — those stay in DataWidget.
    """

    def __init__(self, viewer, app_state):
        self.viewer = viewer
        self.app_state = app_state
        self._secondary_widget: SecondaryVideoWidget | None = None
        self._secondary_fps: float = 0.0
        self._central_splitter: QSplitter | None = None
        self._original_central = None
        self._video_format_warned = False

    @property
    def secondary_widget(self) -> SecondaryVideoWidget | None:
        return self._secondary_widget

    @property
    def secondary_fps(self) -> float:
        return self._secondary_fps

    def update_video(self, plot_container, transform_widget):
        if not self.app_state.ready:
            return
        camera_sel = getattr(self.app_state, 'cameras_sel', None)
        video_file = None
        if camera_sel:
            dt = self.app_state.dt
            video_file = dt.get_video(self.app_state.trials_sel, camera_sel)
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

        video_data = FastVideoReader(
            self.app_state.video_path, read_format='rgb24',
        )

        _ = video_data.shape
        n_frames = int(video_data.shape[0]) if len(video_data.shape) >= 1 else 0
        if n_frames > 0:
            restore_frame = min(restore_frame, n_frames - 1)
        else:
            restore_frame = 0

        video_layer = self.viewer.add_image(video_data, name="video", rgb=True)
        video_index = self.viewer.layers.index(video_layer)
        self.viewer.layers.move(video_index, 0)

        try:
            alignment = getattr(self.app_state, 'trial_alignment', None)
            video_time_offset = alignment.video_offset if alignment else 0.0
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
            qt_window.setCentralWidget(self._central_splitter)
            central.show()
            meta_widget.reapply_shortcuts()

            def _settle():
                self._equalize_video_split_now()
                layout_mgr.restore_dock_widths(saved)

            QTimer.singleShot(50, _settle)
        else:
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
        if getattr(self.app_state, 'video', None):
            self.app_state.video.stop()
            self.app_state.video = None
        self._cleanup_primary_video()
        self.hide_secondary_video()
        self._secondary_widget = None
        self._central_splitter = None


    def _resolve_video_path(self, camera_name: str, video_folder: str | None) -> str | None:
        if is_url(camera_name):
            return camera_name
        if video_folder:
        
            video_file = self.app_state.dt.get_video(self.app_state.trials_sel, camera_name)
            if video_file:
                path = os.path.normpath(os.path.join(video_folder, video_file))
                return path if os.path.isfile(path) else None
        return None
