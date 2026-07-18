"""Video playback + time synchronization over the pygfx camera view.

``VideoSync`` keeps the exact public API the rest of the GUI relies on
(``seek_to_frame``, ``play_segment``, ``frame_to_time`` …) but drives a
:class:`~ethograph.gui.pygfx_video.CameraView` instead of napari dims.

Playback is a QTimer stepping trial frames: the render rate is capped at
30 fps and frames are skipped to honour ``fps_playback`` (this is the old
"skip frames" behaviour, now the only path — there is no napari dims.play).
"""

from typing import Any, Optional

from qtpy.QtCore import QObject, QTimer, Signal

from ethograph.utils.audio import get_audio_sr

MAX_RENDER_FPS = 30.0


class VideoSync(QObject):
    """Video player synchronized with the plots and audio playback."""

    frame_changed = Signal(int)
    playback_stopped = Signal()

    def __init__(
        self,
        app_state,
        view,
        video_source: str,
        audio_source: Optional[str] = None,
    ):
        super().__init__()
        self.app_state = app_state
        self.view = view
        self.video_source = video_source
        self.audio_source = audio_source
        self._time_offset = view.time_offset

        self._audio_player: Any = None
        self._segment_end_frame: Optional[int] = None
        self._current_frame: int = 0

        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._advance)
        self._step: float = 1.0
        self._frame_accum: float = 0.0

        self.total_frames = view.n_frames
        self.total_duration = self.total_frames / self.fps if self.fps else 0.0
        self.audio_sr = get_audio_sr(audio_source) if audio_source else None

        view.time_changed.connect(self._on_view_time_changed)

    # ------------------------------------------------------------------
    # Time mapping
    # ------------------------------------------------------------------

    def frame_to_time(self, frame: int) -> float:
        return frame / self.fps + self._time_offset

    def time_to_frame(self, time_s: float) -> int:
        return int((time_s - self._time_offset) * self.fps)

    @property
    def fps(self) -> float:
        return self.app_state.video_fps

    @property
    def fps_playback(self) -> float:
        return self.app_state.fps_playback

    @property
    def skip_frames(self) -> bool:
        return self.app_state.skip_frames

    @property
    def is_playing(self) -> bool:
        return self._play_timer.isActive()

    @property
    def current_frame(self) -> int:
        return self._current_frame

    # ------------------------------------------------------------------
    # Seeking
    # ------------------------------------------------------------------

    def seek_to_frame(self, frame: int):
        frame = max(0, min(int(frame), self.total_frames - 1)) if self.total_frames else 0
        self.view.seek_trial_frame(frame)
        self._apply_frame(frame)

    def _apply_frame(self, frame: int):
        """Update state + notify listeners (plots, extra cameras)."""
        self._current_frame = frame
        self.app_state.current_frame = frame
        self.frame_changed.emit(frame)
        if self._segment_end_frame is not None and frame >= self._segment_end_frame:
            self._segment_end_frame = None
            self.stop()

    def _on_view_time_changed(self, video_time: float):
        """User interacted with the video canvas (scroll/keys)."""
        if self.fps <= 0:
            return
        frame = int(round(video_time * self.fps)) - self.view.start_frame
        frame = max(0, min(frame, self.total_frames - 1)) if self.total_frames else 0
        self._apply_frame(frame)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def start(self):
        if self.is_playing:
            return
        self._segment_end_frame = None
        self._start_timer()

    def stop(self):
        self._play_timer.stop()
        self._segment_end_frame = None
        self._stop_audio()
        self.playback_stopped.emit()

    def toggle_pause_resume(self):
        self.stop() if self.is_playing else self.start()

    def _start_timer(self):
        render_fps = min(self.fps_playback, MAX_RENDER_FPS)
        if render_fps <= 0:
            return
        self._step = self.fps_playback / render_fps
        self._frame_accum = float(self._current_frame)
        self._play_timer.start(int(1000 / render_fps))

    def _advance(self):
        self._frame_accum += self._step
        next_frame = int(round(self._frame_accum))
        max_frame = self.total_frames - 1
        if self._segment_end_frame is not None:
            max_frame = min(max_frame, self._segment_end_frame)
        if next_frame >= max_frame:
            next_frame = max_frame
            self.view.seek_trial_frame(next_frame)
            self._apply_frame(next_frame)
            if self._segment_end_frame is None:
                self.stop()
            return
        self.view.seek_trial_frame(next_frame)
        self._apply_frame(next_frame)

    def play_segment(self, start_frame: int, end_frame: int):
        self.stop()

        start_frame = max(0, min(int(start_frame), self.total_frames - 1))
        end_frame = max(0, min(int(end_frame), self.total_frames - 1))
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, self.total_frames - 1)

        self._segment_end_frame = end_frame
        self.view.seek_trial_frame(start_frame, synchronous=True)
        self._apply_frame(start_frame)
        self._segment_end_frame = end_frame  # _apply_frame may have cleared it

        audio_path = self.app_state.audio_path or self.audio_source
        if audio_path:
            try:
                from audioio import AudioLoader, PlayAudio
            except ImportError:
                audio_path = None
        if audio_path:
            with AudioLoader(audio_path) as data:
                audio_sr = data.rate
                start_sample = int(start_frame / self.fps * audio_sr)
                end_sample = int(end_frame / self.fps * audio_sr)
                segment = data[start_sample:end_sample]

            if segment.ndim > 1:
                _, channel_idx = self.app_state.get_audio_source()
                n_channels = segment.shape[1]
                channel_idx = min(channel_idx, n_channels - 1)
                segment = segment[:, channel_idx]

            if self.app_state.av_speed_coupled:
                rate = (self.fps_playback / self.fps) * audio_sr
            else:
                rate = self.app_state.audio_playback_speed * audio_sr
            self._audio_player = PlayAudio()
            self._audio_player.play(data=segment, rate=float(rate), blocking=False)

        self._start_timer()

    def _stop_audio(self):
        if self._audio_player:
            self._audio_player.stop()
            self._audio_player.__exit__(None, None, None)
            self._audio_player = None

    def cleanup(self):
        try:
            self.view.time_changed.disconnect(self._on_view_time_changed)
        except (RuntimeError, TypeError):
            pass
        self._play_timer.stop()
        self._stop_audio()
        self._segment_end_frame = None
        self.playback_stopped.emit()
