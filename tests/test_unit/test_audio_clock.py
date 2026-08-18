"""AudioClock arithmetic without an audio device.

Covers the timing bugs from the Windows tester reports:

- ``elapsed_s()`` used to clamp ``idx/out_rate - latency`` at 0 (short
  Play → Stop bursts never advanced the playhead), and the wall-clock bridge
  that replaced it made the marker LEAD the sound by up to the device latency.
  The position now extrapolates from the callback's DAC anchor
  (``time_info.outputBufferDacTime``, the Audacity/mpv approach): it must hold
  at 0 until the first sample is audible, then track the stream clock exactly,
  and a Stop mid-burst must commit a non-zero position — on the fallback path
  (garbage host timestamps) too.
- ``_prepare_output`` used to resample the whole remaining trial with an
  unbounded polyphase ratio (48000/24414 → a ~160k-tap FIR) synchronously on
  the GUI thread. Resampling is now chunked with a bounded ratio; the chunks
  concatenated must match a whole-span ``resample_poly``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui.audio_clock import (
    MAX_RESAMPLE_DENOM,
    OUTPUT_RATE,
    AudioClock,
)

FS = 24414.0  # the tester's awkward TDT rate


def _clock(duration_s: float = 2.0, fs: float = FS, speed: float = 1.0) -> AudioClock:
    data = np.zeros(int(fs * duration_s), dtype="float32")
    return AudioClock(data, fs, speed=speed)


class _FakeTime:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _FakeStream:
    """Just enough of ``sd.OutputStream``: the stream clock + teardown."""

    def __init__(self):
        self.time = 0.0

    def abort(self):
        pass

    def close(self):
        pass


class _FakeTimeInfo:
    def __init__(self, current: float, dac: float):
        self.currentTime = current
        self.outputBufferDacTime = dac


# ----------------------------------------------------------------------
# elapsed_s: the DAC anchor
# ----------------------------------------------------------------------


def _start_stream(clock: AudioClock, latency_s: float = 0.3) -> tuple[_FakeStream, _FakeTime]:
    """Put the clock in the 'stream running' state without a device."""
    clock._init_producer()
    clock._latency_s = latency_s
    clock._stream = _FakeStream()
    now = _FakeTime()
    clock._now = now
    clock._wall_start = 0.0
    return clock._stream, now


def _run_callback(clock: AudioClock, n_frames: int, current: float, dac: float) -> None:
    out = np.zeros((n_frames, 1), dtype="float32")
    clock._callback(out, n_frames, _FakeTimeInfo(current, dac), None)


def test_marker_waits_until_the_sound_is_audible():
    clock = _clock()
    stream, _ = _start_stream(clock)
    # 0.1 s of audio handed over; its first sample hits the DAC at t=1.2.
    _run_callback(clock, int(0.1 * clock._out_rate), current=1.0, dac=1.2)
    stream.time = 1.1
    assert clock.elapsed_s() == 0.0  # handed, but not audible yet
    stream.time = 1.25
    assert clock.elapsed_s() == pytest.approx(0.05, abs=1e-6)
    stream.time = 1.28
    assert clock.elapsed_s() == pytest.approx(0.08, abs=1e-6)


def test_elapsed_never_leads_the_frames_handed_to_the_device():
    clock = _clock()
    stream, _ = _start_stream(clock)
    _run_callback(clock, int(0.1 * clock._out_rate), current=1.0, dac=1.2)
    # Stream clock far past the one buffer handed: clamp at what was handed.
    stream.time = 6.0
    assert clock.elapsed_s() == pytest.approx(0.1, abs=1e-6)


def test_elapsed_is_monotonic_across_anchor_updates():
    clock = _clock()
    stream, _ = _start_stream(clock)
    block = int(0.05 * clock._out_rate)
    values = []
    for i in range(20):
        # Slightly jittered DAC timestamps, as a real host delivers them.
        dac = 0.3 + i * 0.05 + (0.004 if i % 3 else -0.003)
        _run_callback(clock, block, current=dac - 0.2, dac=dac)
        for tick in range(5):
            stream.time = 0.15 + i * 0.05 + tick * 0.01
            values.append(clock.elapsed_s())
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_stop_commits_the_audible_burst_position():
    clock = _clock()
    stream, _ = _start_stream(clock)
    _run_callback(clock, int(0.5 * clock._out_rate), current=0.4, dac=0.5)
    stream.time = 0.65  # 0.15 s audible — a short Play → Stop burst
    clock.stop()
    frozen = clock.elapsed_s()
    assert frozen == pytest.approx(0.15, abs=1e-6)
    stream.time = 5.0
    assert clock.elapsed_s() == frozen


# ----------------------------------------------------------------------
# elapsed_s: the fallback when the host's time_info is garbage
# ----------------------------------------------------------------------


def test_garbage_timestamps_never_publish_an_anchor():
    clock = _clock()
    _start_stream(clock)
    _run_callback(clock, int(0.1 * clock._out_rate), current=0.0, dac=0.0)
    assert clock._dac_anchor is None
    assert clock._dac_bad == 1


def test_fallback_is_monotonic_and_never_leads():
    clock = _clock()
    stream, now = _start_stream(clock, latency_s=0.3)
    stream.time = 0.0  # invalid stream clock → fallback path
    values = []
    for step in range(150):
        now.t = step * 0.01
        clock._idx = int(min(now.t + 0.1, 1.4) * clock._out_rate)
        values.append(clock.elapsed_s())
    assert values[0] == 0.0  # no wall-clock lead at Play
    assert all(b >= a for a, b in zip(values, values[1:]))
    # Warm: wall - latency, still capped by frames handed minus latency.
    assert values[-1] == pytest.approx(min(now.t, 1.4) - 0.3, abs=0.02)


def test_stop_commits_a_short_burst_even_without_an_anchor():
    # The conservative display fallback reads 0 inside the latency window;
    # committing that would re-create the tester's burst bug. Stop must
    # commit the optimistic wall span instead.
    clock = _clock()
    stream, now = _start_stream(clock, latency_s=0.3)
    stream.time = 0.0  # no usable stream clock
    now.t = 0.15
    clock._idx = int(0.25 * clock._out_rate)
    assert clock.elapsed_s() == 0.0  # display holds inside the window
    clock.stop()
    assert clock.elapsed_s() == pytest.approx(0.15, abs=1e-6)


# ----------------------------------------------------------------------
# Chunked resampling
# ----------------------------------------------------------------------


def _produce_all(clock: AudioClock) -> np.ndarray:
    clock._init_producer()
    while clock._produce_chunk():
        pass
    return np.concatenate(clock._chunks)


def test_chunked_resample_matches_whole_span():
    resample_poly = pytest.importorskip("scipy.signal").resample_poly
    rng = np.random.default_rng(0)
    data = rng.standard_normal(int(FS * 7.3)).astype("float32")
    clock = AudioClock(data, FS, speed=1.0)
    got = _produce_all(clock)
    ref = resample_poly(data, clock._up, clock._down)
    assert len(got) == len(ref)
    assert np.allclose(got, ref, atol=1e-4)


def test_resample_ratio_is_bounded():
    pytest.importorskip("scipy.signal")
    clock = _clock()
    clock._init_producer()
    # The exact 48000/24414 reduces to 8000/4069 — a ~160k-tap FIR. The
    # bounded approximation must keep the polyphase factors small.
    assert clock._down <= MAX_RESAMPLE_DENOM
    assert max(clock._up, clock._down) < 5000
    # ... while staying accurate enough that the clock cannot visibly drift.
    exact = OUTPUT_RATE / FS
    assert clock._up / clock._down == pytest.approx(exact, rel=1e-5)


def test_output_length_scales_with_speed():
    pytest.importorskip("scipy.signal")
    duration = 2.0
    clock = _clock(duration_s=duration, speed=0.5)
    total = len(_produce_all(clock))
    # Half speed → the span lasts twice as long at the output rate.
    assert total == pytest.approx(OUTPUT_RATE * duration / 0.5, rel=1e-3)


def test_first_chunk_is_bounded_not_the_whole_span():
    pytest.importorskip("scipy.signal")
    clock = _clock(duration_s=120.0)
    clock._init_producer()
    # Play must not resample the remaining trial synchronously: the first
    # (synchronous) chunk covers seconds, not minutes.
    assert not clock._produce_done
    assert clock._produced_out < 10 * OUTPUT_RATE
