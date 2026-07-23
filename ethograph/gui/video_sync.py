"""Video playback + time synchronization over the pygfx camera view.

``VideoSync`` keeps the exact public API the rest of the GUI relies on
(``seek_to_frame``, ``play_segment``, ``frame_to_time`` …) but drives a
:class:`~ethograph.gui.pygfx_video.CameraView` instead of napari dims.

Playback is a QTimer stepping trial frames at ``fps_playback``; there is no
render-rate cap — the async decoder drops stale seek requests, so it is the
natural limiter (no forced frame-skipping, no napari dims.play).
"""

import logging
from typing import Any, Optional

from qtpy.QtCore import QObject, QTimer, Signal
from qtpy.QtGui import QGuiApplication

from ethograph.utils.audio import get_audio_sr

logger = logging.getLogger(__name__)

# The frame timer is capped at the display refresh rate (queried at runtime):
# the monitor can't show more distinct frames than that, and it bounds the
# per-frame storm (frame_changed → plot/marker redraws, pose overlay, GL draws)
# so fast playback can't flood the event loop / GL canvas and crash. When
# fps_playback exceeds the refresh rate, playback steps several source frames
# per tick and the async decoder drops the stale ones. Used only if the screen
# refresh rate can't be queried.
RENDER_FPS_FALLBACK = 60.0


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
        # True (sub-frame) end time of the segment being played, set only when
        # ``app_state.segment_end_continuous_time`` is on (see play_segment()).
        self._segment_end_time_s: Optional[float] = None
        self._current_frame: int = 0

        # Phase 3: audio-master clock (regular playback). When set, playback is
        # driven from real audio output; the marker sits on the exact clock time
        # via ``marker_time_override`` (read by plots_container).
        self._audio_clock: Any = None
        self._clock_start_frame: int = 0
        self._clock_start_marker: float = 0.0
        self.marker_time_override: Optional[float] = None
        # Smooth mode: decode-paced (synchronous) stepping, every frame, no audio.
        self._smooth_mode: bool = False

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

    def time_to_frame(self, time_s: float, *, round_nearest: bool = False) -> int:
        frames = (time_s - self._time_offset) * self.fps
        return round(frames) if round_nearest else int(frames)

    @property
    def fps(self) -> float:
        return self.app_state.video_fps

    @property
    def fps_playback(self) -> float:
        """Render-target FPS: native video FPS scaled by ``playback_speed_pct``."""
        return self.fps * self.app_state.playback_speed_pct / 100.0

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
        # An explicit seek always shows the frame-exact time — clear any
        # leftover marker override from a prior continuous-end segment finish.
        self.marker_time_override = None
        frame = max(0, min(int(frame), self.total_frames - 1)) if self.total_frames else 0
        self.view.seek_trial_frame(frame)
        self._apply_frame(frame)

    def _apply_frame(self, frame: int):
        """Update state + notify listeners (plots, extra cameras)."""
        self._current_frame = frame
        self.app_state.current_frame = frame
        finishing_segment = self._segment_end_frame is not None and frame >= self._segment_end_frame
        if finishing_segment and self._segment_end_time_s is not None:
            # Continuous-end mode: snap the marker to the segment's true
            # (sub-frame) end time rather than the nearest frame's time.
            self.marker_time_override = self._segment_end_time_s
        self.frame_changed.emit(frame)
        if finishing_segment:
            self._segment_end_frame = None
            self.stop(clear_marker_override=self._segment_end_time_s is None)

    def _on_view_time_changed(self, video_time: float):
        """User interacted with the video canvas (scroll/keys)."""
        if self.fps <= 0:
            return
        self.marker_time_override = None
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
        self._segment_end_time_s = None
        self._audio_clock = None
        self._smooth_mode = False
        self.marker_time_override = None

        from .app_constants import PLAYBACK_MODE_SMOOTH, PLAYBACK_MODE_SYNCED

        mode = self.app_state.effective_playback_mode()

        # Audio-synced: drive playback from real audio output so marker/video
        # can't drift. Audio only ever plays in this mode. There is no speed cap
        # — the audio device's max sample rate is the only limit; beyond it the
        # stream simply fails to open and playback falls through silently.
        if mode == PLAYBACK_MODE_SYNCED and self.fps > 0:
            t0, t1 = self._current_frame / self.fps, self.total_frames / self.fps
            clock = self._build_audio_clock(t0, t1)
            if clock is not None and clock.start():
                self._audio_clock = clock
                self._clock_start_frame = self._current_frame
                self._clock_start_marker = self.frame_to_time(self._current_frame)
                render_fps = min(self.fps_playback, self._render_cap())
                if render_fps > 0:
                    self._play_timer.start(int(1000 / render_fps))
                return
            # synced requested but no usable audio device → fall through silent.

        if mode == PLAYBACK_MODE_SMOOTH:
            # Decode-paced: show every frame in order (may run slower than fps).
            self._smooth_mode = True
            render_fps = min(self.fps_playback, self._render_cap()) if self.fps_playback > 0 else self._render_cap()
            self._play_timer.start(int(1000 / render_fps))
            return

        # Skip frames (default): approximate real-time fps by dropping frames.
        self._start_timer()

    def stop(self, clear_marker_override: bool = True):
        """Stop playback.

        ``clear_marker_override=False`` lets a continuous-end segment finish
        (see ``_apply_frame``) leave the marker frozen on the segment's exact
        end time instead of snapping it back to the last frame's time.
        """
        self._play_timer.stop()
        self._segment_end_frame = None
        self._segment_end_time_s = None
        if self._audio_clock is not None:
            self._audio_clock.stop()
            self._audio_clock = None
        self._smooth_mode = False
        if clear_marker_override:
            self.marker_time_override = None
        self._stop_audio()
        self.playback_stopped.emit()

    def _playback_speed(self) -> float:
        # Single speed lever for both video FPS and audio pitch/rate.
        return self.app_state.playback_speed_pct / 100.0

    def toggle_pause_resume(self):
        self.stop() if self.is_playing else self.start()

    def _render_cap(self) -> float:
        """Max frame-timer rate: the display refresh rate (queried at runtime)."""
        screen = QGuiApplication.primaryScreen()
        rate = screen.refreshRate() if screen is not None else 0.0
        return rate if rate and rate > 0 else RENDER_FPS_FALLBACK

    def _start_timer(self):
        # Fire at min(fps_playback, display refresh); at higher playback speeds
        # step several frames per tick (the async decoder drops the stale ones)
        # rather than flooding the event loop / GL canvas with timer ticks.
        render_fps = min(self.fps_playback, self._render_cap())
        if render_fps <= 0:
            return
        self._step = self.fps_playback / render_fps
        self._frame_accum = float(self._current_frame)
        self._play_timer.start(int(1000 / render_fps))

    def _advance(self):
        if self._audio_clock is not None:
            self._advance_from_clock()
            return
        if self._smooth_mode:
            self._advance_smooth()
            return
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

    def _advance_smooth(self):
        """Decode-paced stepping: advance one frame, decoding it synchronously so
        no frame is ever skipped. Under load this runs slower than real-time
        instead of dropping frames (the tradeoff for a smooth review pass)."""
        next_frame = self._current_frame + 1
        max_frame = self.total_frames - 1
        if self._segment_end_frame is not None:
            max_frame = min(max_frame, self._segment_end_frame)
        next_frame = min(next_frame, max_frame)
        self.view.seek_trial_frame(next_frame, synchronous=True)
        self._apply_frame(next_frame)
        if next_frame >= max_frame and self._segment_end_frame is None:
            self.stop()

    def _advance_from_clock(self):
        """Drive video + marker from the audio clock's real playback position."""
        clock = self._audio_clock
        elapsed = clock.elapsed_s()
        max_frame = self.total_frames - 1
        frame = min(self._clock_start_frame + int(round(elapsed * self.fps)), max_frame)
        # The playhead sits on the exact (sub-frame) clock time; plots_container
        # reads this override so it isn't quantized to the displayed frame.
        self.marker_time_override = self._clock_start_marker + elapsed
        self.view.seek_trial_frame(frame)
        self._apply_frame(frame)
        if clock.finished or frame >= max_frame:
            self.stop()

    def _build_audio_clock(self, t0_s: float, t1_s: float):
        """Preload the selected audio span and wrap it in an :class:`AudioClock`.

        Returns ``None`` when audio is unavailable so the caller falls back to
        the frame timer.
        """
        from .audio_clock import AudioClock

        resolved, channel_idx = self.app_state.get_audio_source(self.app_state.playback_mic_selection())
        audio_path = resolved or self.app_state.audio_path or self.audio_source
        if not audio_path:
            return None
        try:
            from audioio import AudioLoader
        except ImportError:
            return None
        try:
            with AudioLoader(audio_path) as data:
                fs = data.rate
                s0 = max(0, int(t0_s * fs))
                s1 = min(len(data), int(t1_s * fs))
                if s1 <= s0:
                    return None
                segment = data[s0:s1]
            if segment.ndim > 1:
                segment = segment[:, min(channel_idx, segment.shape[1] - 1)]
            return AudioClock(segment, fs, speed=self._playback_speed())
        except Exception:
            logger.warning("Could not build audio clock; falling back.", exc_info=True)
            return None

    def play_segment(
        self,
        start_frame: int,
        end_frame: int,
        audio_t0: float | None = None,
        audio_t1: float | None = None,
    ):
        """Play frames ``[start_frame, end_frame]`` with synchronized audio.

        ``audio_t0``/``audio_t1`` are the true (sub-frame) segment bounds in
        seconds; when given, audio is sliced from them so it isn't quantized to
        the video frame grid (Phase 2). Video still shows the nearest frames.

        When ``app_state.segment_end_continuous_time`` is on and ``audio_t1``
        is given, the red marker snaps to that exact end time when the segment
        finishes instead of stopping on the nearest frame's time — see
        docs/advanced/playback.md.
        """
        self.stop()

        start_frame = max(0, min(int(start_frame), self.total_frames - 1))
        end_frame = max(0, min(int(end_frame), self.total_frames - 1))
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, self.total_frames - 1)

        self._segment_end_frame = end_frame
        self._segment_end_time_s = (
            audio_t1 if (audio_t1 is not None and self.app_state.segment_end_continuous_time) else None
        )
        self.view.seek_trial_frame(start_frame, synchronous=True)
        self._apply_frame(start_frame)
        self._segment_end_frame = end_frame  # _apply_frame may have cleared it

        if self.fps > 0:
            if audio_t0 is None or audio_t1 is None:
                audio_t0, audio_t1 = start_frame / self.fps, end_frame / self.fps
            self._start_audio(audio_t0, audio_t1)

        self._start_timer()

    def _start_audio(self, t0_s: float, t1_s: float):
        """Play the selected audio channel over the trial-relative span ``[t0_s, t1_s]``.

        Best-effort: a missing audio path, absent backend, or a failing output
        device leaves playback video-only rather than aborting it.
        """
        # Follow the last-clicked audio panel (playback_mic_key); resolve both
        # the file and channel from it, falling back to the global audio path.
        resolved_path, channel_idx = self.app_state.get_audio_source(self.app_state.playback_mic_selection())
        audio_path = resolved_path or self.app_state.audio_path or self.audio_source
        if not audio_path:
            return
        try:
            from audioio import AudioLoader, PlayAudio
        except ImportError:
            return

        try:
            with AudioLoader(audio_path) as data:
                audio_sr = data.rate
                start_sample = max(0, int(t0_s * audio_sr))
                end_sample = int(t1_s * audio_sr)
                if end_sample <= start_sample:
                    return
                segment = data[start_sample:end_sample]

            if segment.ndim > 1:
                channel_idx = min(channel_idx, segment.shape[1] - 1)
                segment = segment[:, channel_idx]

            rate = self._playback_speed() * audio_sr

            self._audio_player = PlayAudio()
            self._audio_player.play(data=segment, rate=float(rate), blocking=False)
        except Exception:
            logger.warning("Audio playback failed; continuing video-only.", exc_info=True)
            self._audio_player = None

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
