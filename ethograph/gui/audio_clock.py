"""Audio-output-driven master clock for drift-free playback (Phase 3).

The audio device plays a preloaded mono span through a
:class:`sounddevice.OutputStream`. The number of frames the device has consumed
is the *authoritative* playback position — it is exactly what the listener
hears. Callers drive the timeline marker and the video frame from
:meth:`AudioClock.elapsed_s`, so those can never drift away from the sound
(unlike a wall-clock or a frame-count timer, which accumulate error).

The stream callback runs on PortAudio's thread, so it only touches a preloaded
NumPy array and an integer counter — no disk reads, no Qt, no Python-heavy work.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AudioClock:
    """Play a mono span and expose the true (audible) playback position.

    Parameters
    ----------
    data
        1-D mono audio samples for the span to play.
    samplerate
        The *media* sample rate (Hz) of ``data``.
    speed
        Playback speed multiplier. The stream runs at ``samplerate * speed``;
        :meth:`elapsed_s` still returns media-time seconds.
    """

    def __init__(self, data: np.ndarray, samplerate: float, *, speed: float = 1.0):
        self._data = np.ascontiguousarray(np.asarray(data).ravel(), dtype="float32")
        self._samplerate = float(samplerate)
        self._speed = float(speed) if speed and speed > 0 else 1.0
        self._idx = 0  # media frames handed to the device
        self._finished = False
        self._stream = None
        self._latency_s = 0.0

    @property
    def samplerate(self) -> float:
        return self._samplerate

    @property
    def duration_s(self) -> float:
        return len(self._data) / self._samplerate if self._samplerate else 0.0

    def start(self) -> bool:
        """Open and start the output stream. Returns ``False`` if audio is
        unavailable (no backend, no device, empty span) so the caller can fall
        back to a wall-clock/frame timer."""
        if len(self._data) == 0 or self._samplerate <= 0:
            return False
        try:
            import sounddevice as sd
        except ImportError:
            return False
        try:
            self._stream = sd.OutputStream(
                samplerate=self._samplerate * self._speed,
                channels=1,
                dtype="float32",
                callback=self._callback,
                finished_callback=self._on_finished,
            )
            self._stream.start()
            self._latency_s = float(getattr(self._stream, "latency", 0.0) or 0.0)
            return True
        except Exception:
            logger.warning("AudioClock could not open an output stream; falling back.", exc_info=True)
            self._stream = None
            return False

    def _callback(self, outdata, frames, time_info, status):
        import sounddevice as sd

        i = self._idx
        chunk = self._data[i : i + frames]
        n = len(chunk)
        outdata[:n, 0] = chunk
        if n < frames:
            outdata[n:, 0] = 0.0
            self._idx = i + n
            raise sd.CallbackStop
        self._idx = i + frames

    def _on_finished(self):
        self._finished = True

    @property
    def active(self) -> bool:
        return self._stream is not None and not self._finished

    @property
    def finished(self) -> bool:
        return self._finished

    def elapsed_s(self) -> float:
        """Media-time seconds that are *audible now* (frames played minus the
        output latency). Clamped to ``[0, duration]``."""
        played = self._idx / self._samplerate - self._latency_s * self._speed
        return max(0.0, min(played, self.duration_s))

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._finished = True
