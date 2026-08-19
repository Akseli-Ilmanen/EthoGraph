"""Audio-output-driven master clock for drift-free playback (Phase 3).

The audio device plays a preloaded mono span through a
:class:`sounddevice.OutputStream`. The *audible* playback position is the
authoritative one — it is exactly what the listener hears. Like Audacity and
mpv, the position comes from the device's own timestamps: the stream callback
anchors "output frame N becomes audible at stream time T" from
``time_info.outputBufferDacTime``, and :meth:`AudioClock.elapsed_s`
extrapolates from that anchor against the stream clock. Callers drive the
timeline marker and the video frame from it, so those can never drift away
from the sound (unlike a wall-clock or a frame-count timer, which accumulate
error — and unlike the frames-handed counter, which leads the ear by the
device's output buffering).

The span is resampled (offline) to a fixed device-friendly output rate, so
**any** playback speed and **any** source sample rate can play: the device's
maximum rate no longer limits playback. This is what makes fast pitch-shifted
playback and high-rate (e.g. ultrasonic) recordings audible — the latter via
slow "time-expansion" speeds that shift content into the audible band.

Resampling is *chunked*: only the first chunk is prepared synchronously (so
Play starts instantly), and a daemon thread keeps a bounded look-ahead of
resampled audio queued while the device plays. Pressing Play early in a long
trial therefore no longer resamples the whole remaining trial on the GUI
thread.

The stream callback runs on PortAudio's thread, so it only touches preloaded
NumPy chunks and integer counters — no disk reads, no Qt, no Python-heavy work.
"""

from __future__ import annotations

import logging
import threading
import time
from fractions import Fraction

import numpy as np

logger = logging.getLogger(__name__)

# Fixed output sample rate. Every sound card supports 48 kHz and it spans the
# full audible band, so we always resample the span to it rather than driving
# the device at ``media_rate * speed`` (which the device's max rate would cap).
OUTPUT_RATE = 48000.0

# resample_poly designs a FIR of 2*10*max(up, down)+1 taps. The exact ratio
# OUTPUT_RATE / (fs * speed) can reduce terribly (48000/24414 → 8000/4069: a
# ~160k-tap filter, designed and run over the whole span at every Play press).
# A bounded rational approximation keeps the filter a few thousand taps; the
# rate error (< ~1/MAX_RESAMPLE_DENOM² relative) mis-times the clock by
# microseconds per minute — far below a video frame.
MAX_RESAMPLE_DENOM = 1000

# Master output gain (0.0–1.0), applied in the stream callback so a volume
# change is audible immediately on every live clock. Module-level because a
# fresh AudioClock is built per Play press, in two places (VideoSync and
# AudioPlayer) — a per-instance setting would need re-wiring at both sites.
# Display-only: elapsed_s() and the DAC anchor are untouched by gain.
_master_volume = 1.0


def set_master_volume(volume: float) -> None:
    """Set the playback output gain (clamped to 0.0–1.0)."""
    global _master_volume
    _master_volume = min(1.0, max(0.0, float(volume)))


def master_volume() -> float:
    return _master_volume


def volume_pct_to_gain(pct: float) -> float:
    """Volume slider % (0–100) → linear gain, with a perceptual (cubic) taper.

    Loudness perception is logarithmic: a linear amplitude fader packs nearly
    all audible change into its bottom fifth. The cubic curve (VLC / Web Audio
    convention) approximates a dB fader — 50% ≈ −18 dB, equal slider steps ≈
    equal loudness steps — while keeping true unity at 100 and mute at 0.
    """
    return (min(100.0, max(0.0, float(pct))) / 100.0) ** 3


# Real-playback seconds of media resampled per producer step.
CHUNK_REAL_S = 2.0

# How much resampled audio the producer keeps ahead of the playhead. Playing
# one second of a 30-minute trial must not resample the remaining 29 minutes.
LOOKAHEAD_REAL_S = 30.0


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
        self._out_rate = self._fs * self._speed  # overwritten by _init_producer()
        self._idx = 0  # output frames of real audio served to the device
        self._finished = False
        self._stream = None
        self._latency_s = 0.0

        # Chunked-resample producer state (see _init_producer).
        self._resample = None  # scipy.signal.resample_poly when available
        self._up = 1
        self._down = 1
        self._block_in = 0
        self._margin_in = 0
        self._next_in = 0
        self._chunks: list[np.ndarray] = []
        self._chunk_i = 0  # callback-thread read cursor
        self._chunk_off = 0
        self._produced_out = 0
        self._produce_done = False
        self._stop_producing = False

        # Audible-position tracking. The callback publishes a DAC anchor —
        # (output frames handed before a buffer, the stream-clock time its
        # first sample becomes audible) — read lock-free by elapsed_s().
        # The wall-clock fields back a conservative fallback for host APIs
        # whose time_info is garbage (Windows MME); _dac_bad counts rejected
        # timestamps for the debug launcher.
        self._now = time.perf_counter  # injectable for tests
        self._wall_start: float | None = None
        self._dac_anchor: tuple[int, float] | None = None
        self._dac_bad = 0
        self._last_media_s = 0.0
        self._final_media_s: float | None = None

    @property
    def duration_s(self) -> float:
        """Length of the span in media-time seconds."""
        return len(self._media) / self._fs if self._fs else 0.0

    # ------------------------------------------------------------------
    # Chunked resampling
    # ------------------------------------------------------------------

    def _init_producer(self) -> None:
        """Plan the chunked resample and produce the first chunk synchronously.

        Falls back to streaming the raw media at ``fs * speed`` when SciPy is
        unavailable (works up to the device's max sample rate).
        """
        if len(self._media) == 0 or self._fs <= 0:
            self._produce_done = True
            return
        try:
            from scipy.signal import resample_poly
        except ImportError:
            self._chunks = [self._media]
            self._produced_out = len(self._media)
            self._out_rate = self._fs * self._speed
            self._produce_done = True
            return
        self._resample = resample_poly
        ratio = Fraction(OUTPUT_RATE / (self._fs * self._speed)).limit_denominator(MAX_RESAMPLE_DENOM)
        if ratio <= 0:  # fs * speed beyond any real device rate — keep a sane floor
            ratio = Fraction(1, MAX_RESAMPLE_DENOM)
        self._up, self._down = ratio.numerator, ratio.denominator
        self._out_rate = OUTPUT_RATE
        down = self._down
        block = int(CHUNK_REAL_S * self._fs * self._speed)
        self._block_in = max(down, block // down * down)
        # resample_poly's FIR spans 10*max(up, down) samples of the up-rate
        # grid to each side; chunks overlap by that much input (+2 cushion),
        # rounded up to a multiple of ``down`` so output indices stay
        # integral. Interior samples then match a whole-span resample exactly.
        margin = 10 * max(self._up, self._down) // self._up + 2
        self._margin_in = -(-margin // down) * down
        if not self._produce_chunk():
            self._produce_done = True

    def _produce_chunk(self) -> bool:
        """Resample the next input block; ``False`` once the span is exhausted."""
        a = self._next_in
        n = len(self._media)
        if a >= n:
            return False
        b = min(a + self._block_in, n)
        lo = max(0, a - self._margin_in)
        hi = min(n, b + self._margin_in)
        out = self._resample(self._media[lo:hi], self._up, self._down)
        # Input index i (a multiple of ``down``) lands at output index
        # i*up/down; the tail block keeps resample_poly's own ceil length.
        o_lo = lo * self._up // self._down
        o_a = a * self._up // self._down
        if b == n:
            o_b = -(-(b * self._up) // self._down)
        else:
            o_b = b * self._up // self._down
        chunk = np.ascontiguousarray(out[o_a - o_lo : o_b - o_lo], dtype="float32")
        self._chunks.append(chunk)
        self._produced_out += len(chunk)
        self._next_in = b
        return b < n

    def _produce_rest(self) -> None:
        """Daemon-thread refill: keep ``LOOKAHEAD_REAL_S`` of resampled audio
        ahead of the playhead until the span is exhausted or the clock stops."""
        try:
            while not self._stop_producing:
                ahead = (self._produced_out - self._idx) / self._out_rate
                if ahead > LOOKAHEAD_REAL_S:
                    time.sleep(0.1)
                    continue
                if not self._produce_chunk():
                    return
        except Exception:
            logger.warning("Audio resample producer failed; span truncated.", exc_info=True)
        finally:
            self._produce_done = True

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

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
        self._init_producer()
        if not self._chunks or self._out_rate <= 0:
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
            self._wall_start = self._now()
        except Exception:
            logger.warning("AudioClock could not open an output stream; falling back.", exc_info=True)
            self._stream = None
            self._stop_producing = True
            return False
        if not self._produce_done:
            threading.Thread(target=self._produce_rest, name="AudioClockResample", daemon=True).start()
        return True

    def _callback(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        idx0 = self._idx
        filled = 0
        while filled < frames and self._chunk_i < len(self._chunks):
            chunk = self._chunks[self._chunk_i]
            take = chunk[self._chunk_off : self._chunk_off + (frames - filled)]
            out[filled : filled + len(take)] = take
            filled += len(take)
            self._chunk_off += len(take)
            if self._chunk_off >= len(chunk):
                self._chunk_i += 1
                self._chunk_off = 0
        gain = _master_volume
        if gain != 1.0:
            out[:filled] *= gain
        # Only real audio advances the position: a producer underrun pads
        # silence without counting, so the marker waits with the sound.
        self._idx = idx0 + filled
        # Anchor: output frame ``idx0`` becomes audible at stream time
        # ``dac``. Sanity-gated — some host APIs (Windows MME) hand back
        # zero/garbage time_info, and a bad anchor is worse than none.
        dac = float(getattr(time_info, "outputBufferDacTime", 0.0) or 0.0)
        now = float(getattr(time_info, "currentTime", 0.0) or 0.0)
        prev_dac = self._dac_anchor[1] if self._dac_anchor is not None else 0.0
        if filled and dac <= now + 2.0 and dac >= max(now, prev_dac) > 0.0:
            self._dac_anchor = (idx0, dac)
        elif filled:
            self._dac_bad += 1
        if filled < frames:
            out[filled:] = 0.0
            if self._produce_done:
                import sounddevice as sd

                raise sd.CallbackStop

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

        Extrapolated from the callback's DAC anchor against the stream clock:
        before the first sample is physically audible the position holds at 0
        — the marker waits exactly as long as the sound does, instead of
        leading it by the device's output latency — and from then on it tracks
        the ear sample-accurately. Hosts with unusable timestamps fall back to
        a conservative wall-clock estimate that assumes sound starts one
        device latency after Play; it may lag slightly but never leads. A
        monotonic floor guarantees the position never steps backwards when
        the anchor updates or the fallback hands over.
        """
        if self._final_media_s is not None:
            return self._final_media_s
        if self._out_rate <= 0:
            return 0.0
        media_s = self._audible_media_s(self._stream)
        if media_s is None:
            media_s = self._fallback_estimate()
        media_s = min(media_s, self.duration_s)
        media_s = max(media_s, self._last_media_s)
        self._last_media_s = media_s
        return media_s

    def _audible_media_s(self, stream) -> float | None:
        """Anchor-extrapolated audible media-seconds; ``None`` without one."""
        anchor = self._dac_anchor
        if anchor is None or stream is None:
            return None
        try:
            t = float(stream.time)
        except Exception:
            return None
        if t <= 0.0:
            return None
        idx0, dac0 = anchor
        audible = idx0 + (t - dac0) * self._out_rate
        audible = max(0.0, min(audible, self._idx))
        return audible / self._out_rate * self._speed

    def _fallback_estimate(self) -> float:
        """Audible media-seconds when no DAC anchor exists.

        Assumes audibility starts one device latency after the wall-clock
        start and never exceeds the frames actually handed to the device.
        Conservative: may lag the true position, never leads it.
        """
        est = self._idx / self._out_rate - self._latency_s
        if self._wall_start is not None:
            est = min(self._now() - self._wall_start - self._latency_s, est)
        return max(0.0, est) * self._speed

    def stop(self):
        self._stop_producing = True
        stream, self._stream = self._stream, None
        if stream is not None:
            if self._final_media_s is None:
                # Freeze the position where the listener last heard the sound
                # BEFORE aborting (the anchor needs the live stream clock);
                # VideoSync commits it to current_frame on teardown.
                self._final_media_s = self._commit_position(stream)
            try:
                # abort(), not stop(): stop() drains pending buffers, adding
                # up to a full device latency of blocking to every Stop press.
                stream.abort()
                stream.close()
            except Exception:
                pass
        self._finished = True

    def _commit_position(self, stream) -> float:
        """The position a Stop press leaves the playhead at.

        With a DAC anchor this is the exact audible position. Without one
        (unusable host timestamps) the *conservative* display fallback would
        re-create the burst bug — a Play → Stop shorter than the device
        latency would commit 0 — so commit the optimistic wall-clock span
        capped by the frames handed instead: it overshoots by at most one
        device latency, but a short burst always advances the playhead.
        """
        media_s = self._audible_media_s(stream)
        if media_s is None:
            handed_s = self._idx / self._out_rate if self._out_rate > 0 else 0.0
            wall = self._now() - self._wall_start if self._wall_start is not None else handed_s
            media_s = max(0.0, min(wall, handed_s)) * self._speed
        media_s = min(media_s, self.duration_s)
        return max(media_s, self._last_media_s)
