"""Video lifecycle management over pygfx camera views.

``VideoArea`` is the central widget of the main window: a horizontal splitter
holding the primary :class:`CameraView` and a vertical stack of extra camera
views. ``VideoManager`` keeps its old public surface (update_video,
add_camera, extra_widgets, …) but creates pygfx views instead of napari
layers.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import av
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QSplitter, QVBoxLayout, QWidget

from .notify import notify
from .pygfx_video import CameraView
from .video_sync import VideoSync


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


@dataclass
class VideoProbe:
    """Cheap av-based metadata probe (replaces pre-opened FastVideoReaders)."""

    path: str
    fps: float
    nframes: int


def probe_video(video_path: str) -> VideoProbe:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None:
            raise ValueError(f"Cannot determine frame rate of {video_path}")
        fps = float(rate)
        nframes = stream.frames
        if not nframes and stream.duration and stream.time_base:
            nframes = int(float(stream.duration * stream.time_base) * fps)
        if not nframes and container.duration:
            nframes = int(container.duration / av.time_base * fps)
    return VideoProbe(path=str(video_path), fps=fps, nframes=int(nframes))


class VideoArea(QWidget):
    """Primary camera view + vertical stack of extra camera views."""

    #: Emitted on any mouse press inside the video area (→ video context sidebar).
    clicked = Signal()
    #: Emitted with a CameraView when an extra camera is added (for active-panel).
    camera_added = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # One splitter; every camera is an EQUAL panel (pynaviz-style — there is
        # no large "primary" and small "secondary"). `primary` is just the first
        # panel, kept as an attribute for VideoManager compatibility.
        self._splitter = QSplitter(Qt.Horizontal)
        self.primary = CameraView()
        self._splitter.addWidget(self.primary)
        layout.addWidget(self._splitter)
        self._extras: dict[str, CameraView] = {}

    @property
    def extras(self) -> dict[str, CameraView]:
        return self._extras

    def add_extra(self, name: str) -> CameraView:
        if name in self._extras:
            return self._extras[name]
        view = CameraView()
        self._extras[name] = view
        self._splitter.addWidget(view)
        self.camera_added.emit(view)
        self._equalize()
        return view

    def remove_extra(self, name: str) -> None:
        view = self._extras.pop(name, None)
        if view is None:
            return
        view.clear()
        view.setParent(None)
        view.deleteLater()
        self._equalize()

    def _equalize(self) -> None:
        """Give every camera panel an equal share of the width."""
        n = self._splitter.count()
        if n > 0:
            total = max(1, self._splitter.width())
            self._splitter.setSizes([total // n] * n)


class VideoManager:
    """Manages primary and extra camera views, audio path resolution, sync."""

    def __init__(self, video_area: VideoArea, app_state):
        self.video_area = video_area
        self.app_state = app_state
        self._video_format_warned = False
        self._audio_row_widgets: list = []

    @property
    def primary_view(self) -> CameraView:
        return self.video_area.primary

    @property
    def extra_widgets(self) -> dict[str, CameraView]:
        return self.video_area.extras

    # ------------------------------------------------------------------
    # Primary video
    # ------------------------------------------------------------------

    def update_video(self, plot_container):
        if not self.app_state.ready:
            return
        camera = self.app_state.primary_camera
        sio = getattr(self.app_state, "nwb_alignment", None)
        if camera and sio is not None:
            self.app_state.video_path = sio.resolve_media_path(
                self.app_state.trials_sel,
                "video",
                device=camera,
                fallback_folder=self.app_state.video_folder,
            )
        else:
            self.app_state.video_path = None
        if not self.app_state.video_path:
            return
        restore_frame = max(0, int(getattr(self.app_state, "current_frame", 0) or 0))
        self._warn_video_format()
        self._cleanup_primary_video()
        self._setup_primary_video(restore_frame)

    def _cleanup_primary_video(self):
        sync = getattr(self.app_state, "video", None)
        if sync is not None:
            try:
                sync.frame_changed.disconnect(self._on_primary_frame_changed)
            except (RuntimeError, TypeError):
                pass
            sync.cleanup()
            self.app_state.video = None
        self.primary_view.clear()

    def _trial_clip(self, fps: float, time_offset: float, nframes: int) -> tuple[int, int, float]:
        """Compute (start_frame, end_frame, effective_offset) for the trial."""
        alignment = getattr(self.app_state, "trial_alignment", None)
        if alignment and alignment.trial_range:
            trial_start_in_video = -time_offset
            start_frame = max(0, int(trial_start_in_video * fps))
            end_frame = int((trial_start_in_video + alignment.trial_range.duration) * fps)
            end_frame = min(end_frame, nframes)
            if start_frame > 0 or end_frame < nframes:
                return start_frame, end_frame, 0.0
        return 0, nframes, time_offset

    def _setup_primary_video(self, restore_frame: int):
        try:
            probe = probe_video(self.app_state.video_path)
        except (OSError, ValueError, av.AVError) as e:
            notify(f"Video file could not be loaded: {e}", "warning")
            return

        if probe.fps and self.app_state.dt is not None:
            camera = self.app_state.primary_camera
            self.app_state.nwb_alignment.set_stream_rate(probe.fps, "video", camera)

        alignment = getattr(self.app_state, "trial_alignment", None)
        video_time_offset = alignment.video_offset if alignment else 0.0
        fps = self.app_state.video_fps
        start_frame, end_frame, effective_offset = self._trial_clip(
            fps, video_time_offset, probe.nframes
        )

        view = self.primary_view
        try:
            view.set_video(
                self.app_state.video_path,
                fps=fps,
                time_offset=effective_offset,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        except (OSError, ValueError) as e:
            notify(f"Video file could not be loaded: {e}", "warning")
            return

        sync = VideoSync(
            app_state=self.app_state,
            view=view,
            video_source=self.app_state.video_path,
            audio_source=self.app_state.audio_path,
        )
        self.app_state.video = sync
        self.app_state.num_frames = sync.total_frames

        sync.frame_changed.connect(self._on_primary_frame_changed)
        sync.frame_changed.connect(self._sync_extra_cameras)
        restore_frame = min(restore_frame, max(0, sync.total_frames - 1))
        sync.seek_to_frame(restore_frame)
        self.app_state.current_frame = restore_frame

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def update_audio(self, plot_container):
        if not self.app_state.ready:
            return
        self._update_audio_path()
        self._update_audio_ui(plot_container)

    def _update_audio_path(self) -> None:
        self.app_state.audio_path = None
        if self.app_state.audio_folder and hasattr(self.app_state, "mics_sel"):
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
        if ext in (".avi", ".mov") and not self._video_format_warned:
            self._video_format_warned = True
            notify(
                f"Video format '{ext}' may have inaccurate frame seeking. "
                f"See https://akseli-ilmanen.github.io/ethograph/user_guide/troubleshooting.html",
                "warning",
            )

    # ------------------------------------------------------------------
    # Frame sync
    # ------------------------------------------------------------------

    def set_frame_changed_callback(self, callback):
        self._frame_changed_callback = callback

    def _on_primary_frame_changed(self, frame_number: int):
        if hasattr(self, "_frame_changed_callback"):
            self._frame_changed_callback(frame_number)

    def _sync_extra_cameras(self, frame_number: int):
        video = getattr(self.app_state, "video", None)
        if video is None or not self.extra_widgets:
            return
        t_seconds = video.frame_to_time(frame_number)
        for view in self.extra_widgets.values():
            view.seek_to_time(t_seconds)

    def toggle_pause_resume(self, plot_container):
        video = getattr(self.app_state, "video", None)
        if video:
            video.toggle_pause_resume()
        else:
            plot_container.toggle_pause_resume()

    # ------------------------------------------------------------------
    # Extra cameras
    # ------------------------------------------------------------------

    def add_camera(self, camera_name: str, video_path: str, layout_mgr=None, meta_widget=None, *, reader=None):
        if camera_name in self.extra_widgets:
            self._update_existing_camera(camera_name, video_path, reader=reader)
            return

        probe = reader if isinstance(reader, VideoProbe) else None
        if probe is None:
            try:
                probe = probe_video(video_path)
            except (OSError, ValueError, av.AVError) as e:
                notify(f"Could not open camera '{camera_name}': {e}", "warning")
                return
        self._store_camera_fps_in_session(camera_name, probe.fps)

        view = self.video_area.add_extra(camera_name)
        self._load_extra_video(view, camera_name, video_path, probe)
        self._sync_widget_to_current_time(view)

    def _update_existing_camera(self, camera_name: str, video_path: str, *, reader=None):
        probe = reader if isinstance(reader, VideoProbe) else None
        if probe is None:
            try:
                probe = probe_video(video_path)
            except (OSError, ValueError, av.AVError) as e:
                notify(f"Could not open camera '{camera_name}': {e}", "warning")
                return
        self._store_camera_fps_in_session(camera_name, probe.fps)
        view = self.extra_widgets[camera_name]
        self._load_extra_video(view, camera_name, video_path, probe)
        view.show()
        self._sync_widget_to_current_time(view)

    def _load_extra_video(self, view: CameraView, camera_name: str, video_path: str, probe: VideoProbe):
        sio = getattr(self.app_state, "nwb_alignment", None)
        time_offset = 0.0
        if sio is not None:
            trial_id = self.app_state.trials_sel
            time_offset = sio.stream_offset_for_trial(trial_id, "video", camera_name)
        start_frame, end_frame, effective_offset = self._trial_clip(
            probe.fps, time_offset, probe.nframes
        )
        try:
            view.set_video(
                video_path,
                fps=probe.fps,
                time_offset=effective_offset,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        except (OSError, ValueError) as e:
            notify(f"Camera '{camera_name}' failed to load: {e}", "warning")

    def _sync_widget_to_current_time(self, view: CameraView):
        video = getattr(self.app_state, "video", None)
        if video is not None:
            view.seek_to_time(video.frame_to_time(video.current_frame))
        else:
            view.seek_video_frame(0)

    def remove_camera(self, camera_name: str):
        self.video_area.remove_extra(camera_name)

    def remove_all_cameras(self):
        for name in list(self.extra_widgets.keys()):
            self.video_area.remove_extra(name)

    def _store_camera_fps_in_session(self, camera_name: str, fps: float):
        sio = getattr(self.app_state, "nwb_alignment", None)
        if sio is None:
            return
        sio.set_stream_rate(fps, "video", camera_name)

    def cleanup(self):
        if getattr(self.app_state, "video", None):
            self.app_state.video.stop()
            self.app_state.video = None
        self._cleanup_primary_video()
        self.remove_all_cameras()

    @staticmethod
    def open_readers_parallel(paths: dict[str, str]) -> dict[str, VideoProbe]:
        """Probe video metadata for *paths* concurrently.

        Returns ``{camera_name: VideoProbe}`` for every path that probed
        successfully. Failed probes are silently skipped.
        """

        def _probe(video_path: str) -> VideoProbe | None:
            try:
                return probe_video(video_path)
            except Exception:
                return None

        if not paths:
            return {}

        results: dict[str, VideoProbe] = {}
        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            futures = {name: pool.submit(_probe, path) for name, path in paths.items()}
            for name, future in futures.items():
                probe = future.result()
                if probe is not None:
                    results[name] = probe
        return results

    def _resolve_video_path(self, camera_name: str, video_folder: str | None) -> str | None:
        if is_url(camera_name):
            return camera_name
        return self.app_state.nwb_alignment.resolve_media_path(
            self.app_state.trials_sel,
            "video",
            device=camera_name,
            fallback_folder=video_folder,
        )
