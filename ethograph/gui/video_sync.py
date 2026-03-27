"""Video synchronization for napari integration with audio playback support."""


from typing import Optional

import napari
from audioio import AudioLoader, PlayAudio
from qtpy.QtCore import QObject, QTimer, Signal
from napari.utils.notifications import show_error

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
        video_layer=None,
        frame_offset: int = 0,
        time_offset: float = 0.0,
    ):
        super().__init__()
        self.viewer = viewer
        self.app_state = app_state
        self.video_source = video_source
        self.audio_source = audio_source
        self._frame_offset = frame_offset
        self._time_offset = time_offset

        self.qt_viewer = getattr(viewer.window, "_qt_viewer", None)
        self.video_layer = video_layer
        self._audio_player: Optional[PlayAudio] = None
        self._segment_end_actual_frame: Optional[int] = None
        self._seg_frame_count: int = 0
        self._seg_last_frame: Optional[int] = None
        self._stall_count: int = 0

        self._skip_timer = QTimer()
        self._skip_timer.timeout.connect(self._skip_advance)

        self._watchdog = QTimer()
        self._watchdog.setInterval(250)
        self._watchdog.timeout.connect(self._check_playback_alive)

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

        self.total_frames = self.video_layer.data.shape[0]
        self.total_duration = self.total_frames / self.fps

        self.viewer.dims.events.current_step.connect(self._on_napari_step_change)

    def frame_to_time(self, frame: int) -> float:
        return frame / self.fps + self._time_offset

    def time_to_frame(self, time_s: float) -> int:
        return int((time_s - self._time_offset) * self.fps)

    @property
    def fps(self) -> float:
        if self.video_layer is not None:
            try:
                return float(self.video_layer.data.stream.guessed_rate)
            except (AttributeError, ZeroDivisionError):
                pass
        return 30.0

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
            return _get_current_play_status(self.qt_viewer)
        return False

    def _on_napari_step_change(self, event=None):
        if self.viewer.dims.current_step:
            actual_frame = self.viewer.dims.current_step[0]
            trial_frame = actual_frame - self._frame_offset
            self.app_state.current_frame = trial_frame

            if self._segment_end_actual_frame is not None:
                self._seg_frame_count += 1
 
            self.frame_changed.emit(trial_frame)

            if (
                self._segment_end_actual_frame is not None
                and actual_frame >= self._segment_end_actual_frame
            ):
                self._stop_segment_playback()



    def seek_to_frame(self, frame: int):
        if self.video_layer:
            actual_frame = frame + self._frame_offset
            actual_frame = max(0, min(actual_frame, self.total_frames - 1))
            self.viewer.dims.current_step = (actual_frame,) + self.viewer.dims.current_step[1:]
            self._on_napari_step_change()

    def start(self):
        if not self.is_playing:
            self._segment_end_actual_frame = None
            self._seg_last_frame = None
            if self.skip_frames:
                self._start_skip_playback()
            else:
                self.qt_viewer.dims.play(fps=self.fps_playback)
            self._watchdog.start()

    def stop(self):
        self._watchdog.stop()
        self._segment_end_actual_frame = None
        if self._skip_timer.isActive():
            self._skip_timer.stop()
        if _get_current_play_status and self.qt_viewer:
            if _get_current_play_status(self.qt_viewer):
                self.qt_viewer.dims.stop()

    def toggle_pause_resume(self):
        self.stop() if self.is_playing else self.start()

    def play_segment(self, start_frame: int, end_frame: int):

        self.stop()
        self._stop_audio()
 

        start_frame = max(0, min(int(start_frame), self.total_frames - 1))
        end_frame = max(0, min(int(end_frame), self.total_frames - 1))
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, self.total_frames - 1)

        self._segment_end_actual_frame = end_frame + self._frame_offset
        self._seg_frame_count = 0
        self._seg_last_frame = None

        self.seek_to_frame(start_frame)

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
        self._watchdog.start()

    def _check_playback_alive(self):
        current = self.viewer.dims.current_step[0]
        napari_says_playing = _get_current_play_status(self.qt_viewer) if _get_current_play_status and self.qt_viewer else None
        skip_active = self._skip_timer.isActive()

        if self._seg_last_frame is not None and current == self._seg_last_frame:
            self._stall_count += 1

            if self._stall_count >= 1 and not skip_active:

                next_frame = current + 1
                if self._segment_end_actual_frame is not None and next_frame >= self._segment_end_actual_frame:
                    self._stop_segment_playback()
                    return
                if next_frame < self.total_frames:
                    self.viewer.dims.current_step = (next_frame,) + self.viewer.dims.current_step[1:]
                self.qt_viewer.dims.play(axis=0, fps=self.fps_playback)
                self._stall_count = 0
        else:
            self._stall_count = 0
        self._seg_last_frame = current

    def _stop_segment_playback(self):
        if self._segment_end_actual_frame is not None:
            self._segment_end_actual_frame = None
            self.stop()


    def _start_skip_playback(self):
        max_render_fps = 30.0
        render_fps = min(self.fps_playback, max_render_fps)
        self._skip_step = max(1, round(self.fps_playback / render_fps))
        interval_ms = int(1000 / render_fps)
        self._skip_timer.start(interval_ms)

    def _skip_advance(self):
        current_frame = self.viewer.dims.current_step[0]
        max_frame = self.total_frames - 1
        if self._segment_end_actual_frame is not None:
            max_frame = min(max_frame, self._segment_end_actual_frame)

        next_frame = current_frame + self._skip_step
        if next_frame >= max_frame:
            self.viewer.dims.current_step = (max_frame,) + self.viewer.dims.current_step[1:]
            self._on_napari_step_change()
            return
        self.viewer.dims.current_step = (next_frame,) + self.viewer.dims.current_step[1:]
        self._on_napari_step_change()



    def _stop_audio(self):
        if self._audio_player:
            self._audio_player.stop()
            self._audio_player.__exit__(None, None, None)
            self._audio_player = None
   

    def cleanup(self):
        self.viewer.dims.events.current_step.disconnect(self._on_napari_step_change)
        self._skip_timer.stop()
        self._skip_timer.deleteLater()
        self._watchdog.stop()
        self._watchdog.deleteLater()
        self._stop_audio()
        self.stop()