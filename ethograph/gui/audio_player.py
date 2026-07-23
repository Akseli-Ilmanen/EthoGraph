"""Audio playback controller with time marker synchronization."""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING, Callable

from qtpy.QtCore import QTimer

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
        self._start_time = 0.0
        self._start_wall = 0.0
        # Hard stop boundary for segment playback; ``None`` = open playback to
        # the view edge. ``_advance`` snaps the marker to it exactly on stop.
        self._segment_end: float | None = None
        # Optional hook invoked whenever playing flips (start/stop/auto-stop).
        self.on_state_changed: Callable[[], None] | None = None

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

        self._start_time = current_time
        self._start_wall = _time.perf_counter()
        self._segment_end = None
        self._playing = True

        self._start_audio_if_available(current_time, end_time)
        self._timer.start()
        self._notify_state_changed()

    def _start_audio_if_available(self, current_time: float, end_time: float):
        try:
            import sounddevice as sd
        except ImportError:
            return

        from .plots_spectrogram import SharedAudioCache

        audio_path = getattr(self.app_state, "audio_path", None)
        if not audio_path:
            return

        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            return

        fs = loader.rate
        _, channel_idx = self.app_state.get_audio_source()

        start_sample = max(0, int(current_time * fs))
        end_sample = min(len(loader), int(end_time * fs))
        if end_sample <= start_sample:
            return

        audio_data = loader[start_sample:end_sample]
        if audio_data.ndim > 1:
            ch = min(channel_idx, audio_data.shape[1] - 1)
            audio_data = audio_data[:, ch]

        speed = self.app_state.audio_playback_speed
        sd.stop()
        sd.play(audio_data, samplerate=int(fs * speed))

    def stop(self):
        try:
            import sounddevice as sd

            sd.stop()
        except ImportError:
            pass
        self._timer.stop()
        self._playing = False
        self._segment_end = None
        self._notify_state_changed()

    def _notify_state_changed(self):
        if self.on_state_changed is not None:
            self.on_state_changed()

    def play_segment(self, onset_s: float, offset_s: float):
        """Play a segment, stopping the marker exactly on *offset_s*.

        If audio is available, plays it via ``sounddevice``.
        Always drives the time marker from *onset_s* until *offset_s*.
        """
        if offset_s <= onset_s:
            return

        self._start_audio_if_available(onset_s, offset_s)

        self._start_time = onset_s
        self._start_wall = _time.perf_counter()
        self._segment_end = offset_s
        self._playing = True
        self._update_marker(onset_s)  # snap start exactly onto the boundary
        self._notify_state_changed()
        self._timer.start()

    def _advance(self):
        elapsed = _time.perf_counter() - self._start_wall
        speed = self.app_state.audio_playback_speed
        current = self._start_time + elapsed * speed
        end = self._segment_end if self._segment_end is not None else self._get_xlim()[1]

        if current >= end:
            self._update_marker(end)  # land exactly on the boundary, not past it
            self.stop()
            return

        self._update_marker(current)
