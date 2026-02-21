"""Video synchronization for napari integration with audio playback support."""

from typing import Optional

import napari
from audioio import AudioLoader, PlayAudio
from napari.utils.notifications import show_error
from qtpy.QtCore import QObject, QTimer, Signal

from ethograph.utils.audio import get_audio_sr

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
        self._monitor_timer.timeout.connect(self._check_segment_playback)
        self._monitor_end_frame = 0

        self._skip_timer = QTimer()
        self._skip_timer.timeout.connect(self._skip_advance)

        self.total_frames = 0
        self.total_duration = 0.0

        self.audio_sr = get_audio_sr(audio_source)
        
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
    def skip_frames(self) -> bool:
        return self.app_state.skip_frames

    @property
    def is_playing(self) -> bool:
        if self._skip_timer.isActive():
            return True
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
            if self.skip_frames:
                self._start_skip_playback()
            else:
                self.qt_viewer.dims.play(fps=self.fps_playback)

    def stop(self):
        if self._skip_timer.isActive():
            self._skip_timer.stop()
        if _get_current_play_status and self.qt_viewer:
            try:
                if _get_current_play_status(self.qt_viewer):
                    self.qt_viewer.dims.stop()
            except Exception:
                pass

    def toggle_pause_resume(self):
        self.stop() if self.is_playing else self.start()

    def play_segment(self, start_frame: int, end_frame: int):
        self._monitor_timer.stop()
        self._skip_timer.stop()
        self._stop_audio()

        self.seek_to_frame(start_frame)
        self._monitor_end_frame = end_frame

        if self.audio_source and self.audio_sr:
            with AudioLoader(self.audio_source) as data:
                start_sample = int(start_frame / self.fps * self.audio_sr)
                end_sample = int(end_frame / self.fps * self.audio_sr)
                segment = data[start_sample:end_sample]

            if segment.ndim > 1:
                _, channel_idx = self.app_state.get_audio_source()
                n_channels = segment.shape[1]
                channel_idx = min(channel_idx, n_channels - 1)
                segment = segment[:, channel_idx]

            if self.app_state.av_speed_coupled:
                rate = (self.fps_playback / self.fps) * self.audio_sr
            else:
                rate = self.app_state.audio_playback_speed * self.audio_sr
            self._audio_player = PlayAudio()
            self._audio_player.play(data=segment, rate=float(rate), blocking=False)

        if self.skip_frames:
            self._start_skip_playback()
        else:
            self.qt_viewer.dims.play(axis=0, fps=self.fps_playback)
        self._monitor_timer.start(int(1000 / self.fps_playback / 2))

    def _start_skip_playback(self):
        max_render_fps = 30.0
        render_fps = min(self.fps_playback, max_render_fps)
        self._skip_step = max(1, round(self.fps_playback / render_fps))
        interval_ms = int(1000 / render_fps)
        self._skip_timer.start(interval_ms)

    def _skip_advance(self):
        current_frame = self.viewer.dims.current_step[0]
        next_frame = current_frame + self._skip_step
        if next_frame >= self.total_frames:
            self._skip_timer.stop()
            return
        self.viewer.dims.current_step = (next_frame,) + self.viewer.dims.current_step[1:]
        self._on_napari_step_change()

    def _check_segment_playback(self):
        should_stop = not self.is_playing or self.app_state.current_frame >= self._monitor_end_frame
        if should_stop:
            self._monitor_timer.stop()
            self.stop()
            self._stop_audio()

    def _stop_audio(self):
        if self._audio_player:
            self._audio_player.stop()
            self._audio_player.__exit__(None, None, None)
            self._audio_player = None

    def cleanup(self):
        self.viewer.dims.events.current_step.disconnect(self._on_napari_step_change)
        self._monitor_timer.stop()
        self._monitor_timer.deleteLater()
        self._skip_timer.stop()
        self._skip_timer.deleteLater()
        self._stop_audio()
        self.stop()