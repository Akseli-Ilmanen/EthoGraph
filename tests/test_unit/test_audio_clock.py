"""AudioClock arithmetic without an audio device.

Covers the two timing bugs from the Windows tester reports:

- ``elapsed_s()`` used to clamp ``idx/out_rate - latency`` at 0, so for the
  first device-latency window (0.1-0.3 s) the position was pinned to exactly
  0.0 and short Play → Stop bursts never advanced the playhead. It now bridges
  the window from the wall clock and must be monotonic and non-zero there.
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


class _DummyStream:
    def abort(self):
        pass

    def close(self):
        pass


# ----------------------------------------------------------------------
# elapsed_s: the latency-window bridge
# ----------------------------------------------------------------------


def _start_counters(clock: AudioClock, latency_s: float) -> _FakeTime:
    """Put the clock in the 'stream running' state without a device."""
    clock._init_producer()
    clock._latency_s = latency_s
    now = _FakeTime()
    clock._now = now
    clock._wall_start = 0.0
    return now


def test_elapsed_moves_inside_the_latency_window():
    clock = _clock()
    now = _start_counters(clock, latency_s=0.3)
    # Device prebuffers 0.1 s ahead of the wall clock: the naive
    # idx/out_rate - latency stays negative until wall = 0.2 s.
    now.t = 0.1
    clock._idx = int((now.t + 0.1) * clock._out_rate)
    assert clock.elapsed_s() > 0.05  # was exactly 0.0 before the fix


def test_elapsed_is_monotonic_across_the_handover():
    clock = _clock()
    now = _start_counters(clock, latency_s=0.3)
    values = []
    for step in range(150):
        now.t = step * 0.01
        clock._idx = int((now.t + 0.1) * clock._out_rate)
        values.append(clock.elapsed_s())
    assert all(b >= a for a, b in zip(values, values[1:]))
    # Well past the window the device counter is back in charge.
    assert values[-1] == pytest.approx((now.t + 0.1) - 0.3, abs=0.02)


def test_elapsed_never_leads_the_frames_handed_to_the_device():
    clock = _clock()
    now = _start_counters(clock, latency_s=0.3)
    # Wall clock runs, but the device has consumed almost nothing yet.
    now.t = 0.5
    clock._idx = int(0.05 * clock._out_rate)
    assert clock.elapsed_s() <= 0.05 + 1e-9


def test_stop_freezes_the_position():
    clock = _clock()
    now = _start_counters(clock, latency_s=0.3)
    now.t = 0.15
    clock._idx = int((now.t + 0.1) * clock._out_rate)
    clock._stream = _DummyStream()
    clock.stop()
    frozen = clock.elapsed_s()
    assert frozen > 0.0
    now.t = 5.0
    clock._idx = int(1.0 * clock._out_rate)
    assert clock.elapsed_s() == frozen


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
