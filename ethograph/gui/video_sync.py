"""Video synchronization for napari integration with audio playback support."""

from typing import Optional

import napari
from audioio import AudioLoader, PlayAudio
from napari.utils.notifications import show_error
from qtpy.QtCore import QObject, QTimer, Signal

try:
    from napari._qt._qapp_model.qactions._view import _get_current_play_status
except ImportError:
    _get_current_play_status = None


class NapariVideoSync(QObject):
    """Video player integrated with napari's built-in video controls."""

    frame_changed = Signal(int)

    def __init__(
        self,
        viewer: napari.Viewer,
        app_state,
        video_source: str,
        audio_source: Optional[str] = None,
    ):
        super().__init__()
        self.viewer = viewer
        self.app_state = app_state
        self.video_source = video_source
        self.audio_source = audio_source

        self.qt_viewer = getattr(viewer.window, "_qt_viewer", None)
        self.video_layer = None
        self._audio_player: Optional[PlayAudio] = None
        self._monitor_timer = QTimer()
        self._monitor_end_frame = 0

        self.total_frames = 0
        self.total_duration = 0.0

        self.sr = getattr(app_state.ds, "sr", None) if hasattr(app_state, "ds") else None
        if self.sr is None and audio_source:
            try:
                with AudioLoader(audio_source) as data:
                    self.sr = data.rate
            except Exception:
                self.sr = 44100

        for layer in self.viewer.layers:
            if layer.name == "video" and hasattr(layer, "data"):
                self.video_layer = layer
                break

        if not self.video_layer:
            show_error("Video layer not found. Load video first.")
            return

        if hasattr(self.video_layer.data, "shape"):
            self.total_frames = self.video_layer.data.shape[0]
            self.total_duration = self.total_frames / self.fps

        self.viewer.dims.events.current_step.connect(self._on_napari_step_change)

    @property
    def fps(self) -> float:
        return self.app_state.ds.fps

    @property
    def fps_playback(self) -> float:
        return self.app_state.fps_playback

    @property
    def is_playing(self) -> bool:
        if _get_current_play_status and self.qt_viewer:
            try:
                return _get_current_play_status(self.qt_viewer)
            except Exception:
                return False
        return False

    def _on_napari_step_change(self, event=None):
        if self.viewer.dims.current_step:
            frame = self.viewer.dims.current_step[0]
            self.app_state.current_frame = frame
            self.frame_changed.emit(frame)

    def seek_to_frame(self, frame: int):
        if self.video_layer:
            self.viewer.dims.current_step = (frame,) + self.viewer.dims.current_step[1:]
            self._on_napari_step_change()

    def start(self):
        if not self.is_playing:
            self.qt_viewer.dims.play(fps=self.fps_playback)

    def stop(self):
        if self.is_playing:
            self.qt_viewer.dims.stop()

    def toggle_pause_resume(self):
        self.stop() if self.is_playing else self.start()

    def play_segment(self, start_frame: int, end_frame: int):
        self.seek_to_frame(start_frame)
        self._monitor_end_frame = end_frame

        if self.audio_source and self.sr:
            with AudioLoader(self.audio_source) as data:
                start_sample = int(start_frame / self.fps * self.sr)
                end_sample = int(end_frame / self.fps * self.sr)
                segment = data[start_sample:end_sample]

            if segment.shape[0] > 1:
                segment = segment[:, 0]

            rate = (self.fps_playback / self.fps) * self.sr
            self._audio_player = PlayAudio()
            self._audio_player.play(data=segment, rate=float(rate), blocking=False)

        self.qt_viewer.dims.play(axis=0, fps=self.fps_playback)

        def check_playback():
            should_stop = not self.is_playing or self.app_state.current_frame >= self._monitor_end_frame
            if should_stop:
                self._monitor_timer.stop()
                self.stop()
                if self._audio_player:
                    self._audio_player.stop()
                    self._audio_player.__exit__(None, None, None)
                    self._audio_player = None

        self._monitor_timer.timeout.connect(check_playback)
        self._monitor_timer.start(int(1000 / self.fps_playback / 20))

    def cleanup(self):
        self.viewer.dims.events.current_step.disconnect(self._on_napari_step_change)
        self._monitor_timer.stop()
        self._monitor_timer.deleteLater()
        if self._audio_player:
            self._audio_player.stop()
            self._audio_player.__exit__(None, None, None)
            self._audio_player = None
        self.stop()