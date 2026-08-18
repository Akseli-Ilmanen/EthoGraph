"""Audio playback controller with time marker synchronization."""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING, Any, Callable

from qtpy.QtCore import QTimer

from ethograph.io.plot_sources import audio_display_offset

if TYPE_CHECKING:
    from .app_state import ObservableAppState


class AudioPlayer:
    """Plays audio in no-video mode and advances the time marker in sync.

    Parameters
    ----------
    app_state
        Shared application state (audio path, playback speed, channel, …).
    get_xlim
        Callable returning ``(t0, t1)`` of the current view.
    get_visible_time
        Callable returning the current time marker position (seconds).
    update_marker
        Callable that moves the time marker to a given time (seconds).
    """

    def __init__(
        self,
        app_state: ObservableAppState,
        *,
        get_xlim: Callable[[], tuple[float, float]],
        get_visible_time: Callable[[], float],
        update_marker: Callable[[float], None],
    ):
        self.app_state = app_state
        self._get_xlim = get_xlim
        self._get_visible_time = get_visible_time
        self._update_marker = update_marker

        self._playing = False
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance)
        # Real (audible) playback clock, when an audio device is available —
        # resamples to a fixed output rate so the marker tracks the exact
        # audible position instead of drifting from a wall-clock estimate.
        # ``None`` means silent playback (no device/backend/audio path): the
        # marker still runs, driven by a wall-clock fallback.
        self._clock: Any = None
        self._clock_start_marker = 0.0
        self._start_time = 0.0
        self._start_wall = 0.0
        # Hard stop boundary for segment playback; ``None`` = open playback to
        # the view edge. ``_advance`` snaps the marker to it exactly on stop.
        self._segment_end: float | None = None
        # Optional hook invoked whenever playing flips (start/stop/auto-stop).
        self.on_state_changed: Callable[[], None] | None = None
        # A mid-playback channel switch rebuilds the clock on the new channel
        # (guarded getattr — plain app-state fakes carry no signal).
        mic_signal = getattr(app_state, "playback_mic_key_changed", None)
        if mic_signal is not None:
            mic_signal.connect(self._on_playback_channel_changed)

    @property
    def playing(self) -> bool:
        return self._playing

    def toggle(self):
        if self._playing:
            self.stop()
        else:
            self.start()

    def start(self):
        current_time = self._get_visible_time()
        xlim = self._get_xlim()
        end_time = xlim[1]
        if current_time >= end_time:
            return

        self._segment_end = None
        self._playing = True
        self._begin_clock(current_time, end_time, marker_start=current_time)
        self._timer.start()
        self._notify_state_changed()

    def _build_clock(self, t0_s: float, t1_s: float):
        """Preload the selected audio span into an :class:`AudioClock`.

        Returns ``None`` when audio is unavailable so the caller falls back to
        a silent, wall-clock-driven marker.
        """
        from .audio_clock import AudioClock
        from .plots_spectrogram import SharedAudioCache

        # Follow the last-clicked audio panel (playback_mic_key); resolve both
        # file and channel from it, falling back to the global audio path.
        resolved_path, channel_idx = self.app_state.get_audio_source(self.app_state.playback_mic_selection())
        audio_path = resolved_path or getattr(self.app_state, "audio_path", None)
        if not audio_path:
            return None

        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            return None

        # t0/t1 are display-clock; index the file relative to its own start.
        file_start = audio_display_offset(self.app_state, self.app_state.playback_mic_selection())
        fs = loader.rate
        start_sample = max(0, int((t0_s - file_start) * fs))
        end_sample = min(len(loader), int((t1_s - file_start) * fs))
        if end_sample <= start_sample:
            return None

        segment = loader[start_sample:end_sample]
        if segment.ndim > 1:
            ch = min(channel_idx, segment.shape[1] - 1)
            segment = segment[:, ch]

        speed = self.app_state.playback_speed_pct / 100.0
        return AudioClock(segment, fs, speed=speed)

    def _begin_clock(self, t0_s: float, t1_s: float, *, marker_start: float):
        """Start the audio clock over ``[t0_s, t1_s]``, falling back to a
        silent wall-clock marker if no audio device is usable."""
        clock = self._build_clock(t0_s, t1_s)
        if clock is not None and clock.start():
            self._clock = clock
            self._clock_start_marker = marker_start
        else:
            self._clock = None
            self._start_time = marker_start
            self._start_wall = _time.perf_counter()

    def _on_playback_channel_changed(self, *_args):
        """Rebuild the clock on the new channel without stopping playback.

        ``_build_clock`` resolves the mic/channel once, at start — without a
        rebuild the audible channel only changed after Stop and the next Play.
        """
        if not self._playing or self._clock is None:
            return
        old, self._clock = self._clock, None
        elapsed = old.elapsed_s()
        old.stop()
        current = self._clock_start_marker + elapsed
        end = self._segment_end if self._segment_end is not None else self._get_xlim()[1]
        self._begin_clock(current, end, marker_start=current)

    def stop(self):
        if self._clock is not None:
            self._clock.stop()
            self._clock = None
        self._timer.stop()
        self._playing = False
        self._segment_end = None
        self._notify_state_changed()

    def _notify_state_changed(self):
        if self.on_state_changed is not None:
            self.on_state_changed()

    def play_segment(self, onset_s: float, offset_s: float):
        """Play a segment, stopping the marker exactly on *offset_s*.

        If audio is available, plays it through the audio-master clock.
        Always drives the time marker from *onset_s* until *offset_s*.
        """
        if offset_s <= onset_s:
            return

        self._segment_end = offset_s
        self._playing = True
        self._begin_clock(onset_s, offset_s, marker_start=onset_s)
        self._update_marker(onset_s)  # snap start exactly onto the boundary
        self._notify_state_changed()
        self._timer.start()

    def _advance(self):
        if self._clock is not None:
            elapsed = self._clock.elapsed_s()
            current = self._clock_start_marker + elapsed
            finished = self._clock.finished
        else:
            speed = self.app_state.playback_speed_pct / 100.0
            elapsed = (_time.perf_counter() - self._start_wall) * speed
            current = self._start_time + elapsed
            finished = False
        end = self._segment_end if self._segment_end is not None else self._get_xlim()[1]

        if finished or current >= end:
            self._update_marker(end)  # land exactly on the boundary, not past it
            self.stop()
            return

        self._update_marker(current)
