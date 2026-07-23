"""Audio-output-driven master clock for drift-free playback (Phase 3).

The audio device plays a preloaded mono span through a
:class:`sounddevice.OutputStream`. The number of frames the device has consumed
is the *authoritative* playback position — it is exactly what the listener
hears. Callers drive the timeline marker and the video frame from
:meth:`AudioClock.elapsed_s`, so those can never drift away from the sound
(unlike a wall-clock or a frame-count timer, which accumulate error).

The span is resampled once (offline, at preload) to a fixed device-friendly
output rate, so **any** playback speed and **any** source sample rate can play:
the device's maximum rate no longer limits playback. This is what makes fast
pitch-shifted playback and high-rate (e.g. ultrasonic) recordings audible — the
latter via slow "time-expansion" speeds that shift content into the audible band.

The stream callback runs on PortAudio's thread, so it only touches a preloaded
NumPy array and an integer counter — no disk reads, no Qt, no Python-heavy work.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Fixed output sample rate. Every sound card supports 48 kHz and it spans the
# full audible band, so we always resample the span to it rather than driving
# the device at ``media_rate * speed`` (which the device's max rate would cap).
OUTPUT_RATE = 48000.0


class AudioClock:
    """Play a mono span and expose the true (audible) playback position.

    Parameters
    ----------
    data
        1-D mono audio samples for the span to play.
    samplerate
        The *media* sample rate (Hz) of ``data``.
    speed
        Playback speed multiplier (media-seconds per real-second). The span is
        resampled so it plays at ``speed`` regardless of the device rate;
        :meth:`elapsed_s` always returns media-time seconds.
    """

    def __init__(self, data: np.ndarray, samplerate: float, *, speed: float = 1.0):
        self._media = np.ascontiguousarray(np.asarray(data).ravel(), dtype="float32")
        self._fs = float(samplerate)  # media sample rate
        self._speed = float(speed) if speed and speed > 0 else 1.0
        # Output buffer + rate, filled by _prepare_output(); default is the
        # legacy "drive the device at fs*speed" path used when scipy is absent.
        self._data = self._media
        self._out_rate = self._fs * self._speed
        self._idx = 0  # output frames handed to the device
        self._finished = False
        self._stream = None
        self._latency_s = 0.0

    @property
    def duration_s(self) -> float:
        """Length of the span in media-time seconds."""
        return len(self._media) / self._fs if self._fs else 0.0

    def _prepare_output(self) -> None:
        """Resample the span to ``OUTPUT_RATE`` so any speed / source rate plays.

        Falls back to streaming the raw media at ``fs * speed`` when SciPy is
        unavailable (works up to the device's max sample rate).
        """
        if len(self._media) == 0 or self._fs <= 0:
            return
        try:
            from scipy.signal import resample_poly
        except ImportError:
            self._data = self._media
            self._out_rate = self._fs * self._speed
            return

        # Output length so the span lasts (duration / speed) real-seconds at
        # OUTPUT_RATE: resample by OUTPUT_RATE / (fs * speed).
        up = int(round(OUTPUT_RATE))
        down = max(1, int(round(self._fs * self._speed)))
        out = resample_poly(self._media, up, down).astype("float32", copy=False)
        self._data = np.ascontiguousarray(out)
        self._out_rate = OUTPUT_RATE

    def start(self) -> bool:
        """Open and start the output stream. Returns ``False`` if audio is
        unavailable (no backend, no device, empty span) so the caller can fall
        back to a wall-clock/frame timer."""
        if len(self._media) == 0 or self._fs <= 0:
            return False
        try:
            import sounddevice as sd
        except ImportError:
            return False
        self._prepare_output()
        if len(self._data) == 0 or self._out_rate <= 0:
            return False
        try:
            self._stream = sd.OutputStream(
                samplerate=self._out_rate,
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
        """Media-time seconds that are *audible now*, clamped to ``[0, duration]``.

        The device has played ``idx`` output frames = ``idx / out_rate`` real
        seconds; media time advances ``speed``× faster than real time.
        """
        if self._out_rate <= 0:
            return 0.0
        real_s = self._idx / self._out_rate - self._latency_s
        return max(0.0, min(real_s * self._speed, self.duration_s))

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._finished = True
