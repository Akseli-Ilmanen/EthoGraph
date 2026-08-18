"""VideoSync's audio-clock bookkeeping (no device, no decoder).

- Stop commits the clock's final position to ``current_frame``, so a short
  Play → Stop burst leaves the playhead where the listener last heard it (the
  per-tick frame may never have advanced inside the device-latency window).
- A mid-playback channel switch rebuilds the clock from the current elapsed
  position instead of only taking effect after Stop → Play.
- Teardown (``cleanup``) stops a live audio clock — a trial change during
  synced playback must not leave the output stream sounding.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication

from ethograph.gui.video_sync import VideoSync

FPS = 25.0


class _FakeSignal:
    def connect(self, *_args, **_kwargs):
        pass

    def disconnect(self, *_args, **_kwargs):
        pass


class _FakeView:
    def __init__(self):
        self.time_offset = 0.0
        self.n_frames = 100
        self.time_changed = _FakeSignal()
        self.seeks: list[tuple[int, bool]] = []

    def seek_trial_frame(self, frame: int, synchronous: bool = False):
        self.seeks.append((frame, synchronous))

    def decoder_ready(self) -> bool:
        return True


class _FakeAppState:
    video_fps = FPS
    playback_speed_pct = 100.0
    current_frame = 0
    audio_path = None

    def get_audio_source(self, _sel):
        return None, 0

    def playback_mic_selection(self):
        return None


class _FakeClock:
    def __init__(self, elapsed: float = 0.0, startable: bool = True):
        self._elapsed = elapsed
        self._startable = startable
        self.stopped = False

    def start(self) -> bool:
        return self._startable

    def stop(self):
        self.stopped = True

    def elapsed_s(self) -> float:
        return self._elapsed

    @property
    def finished(self) -> bool:
        return False


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _make_sync(qapp) -> VideoSync:
    return VideoSync(_FakeAppState(), _FakeView(), video_source="fake.mp4")


def test_stop_commits_the_burst_position(qapp):
    sync = _make_sync(qapp)
    sync._audio_clock = _FakeClock(elapsed=0.24)
    sync._clock_start_t = 10 / FPS
    sync.stop()
    committed = 10 + round(0.24 * FPS)
    assert sync.current_frame == committed
    assert (committed, False) in sync.view.seeks


def test_stop_without_progress_stays_put(qapp):
    sync = _make_sync(qapp)
    sync._audio_clock = _FakeClock(elapsed=0.0)
    sync._clock_start_t = 0.0
    sync.stop()
    assert sync.current_frame == 0
    assert sync.view.seeks == []


def test_channel_switch_rebuilds_the_clock_in_place(qapp):
    sync = _make_sync(qapp)
    old = _FakeClock(elapsed=1.0)
    new = _FakeClock()
    sync._audio_clock = old
    sync._clock_start_t = 0.4
    sync._clock_start_marker = 0.4
    sync._play_timer.start(1000)
    try:
        asked: list[tuple[float, float]] = []

        def _build(t0, t1):
            asked.append((t0, t1))
            return new

        sync._build_audio_clock = _build
        sync._on_playback_channel_changed()
        assert old.stopped
        assert sync._audio_clock is new
        assert sync._clock_start_t == pytest.approx(1.4)
        assert sync._clock_start_marker == pytest.approx(1.4)
        assert asked and asked[0][0] == pytest.approx(1.4)
    finally:
        sync.stop()


def test_channel_switch_failure_falls_back_to_silent_frames(qapp):
    sync = _make_sync(qapp)
    sync._audio_clock = _FakeClock(elapsed=0.5)
    sync._play_timer.start(1000)
    try:
        sync._build_audio_clock = lambda t0, t1: None
        sync._on_playback_channel_changed()
        assert sync._audio_clock is None
        assert sync.is_playing  # the frame timer took over
    finally:
        sync.stop()


def test_channel_switch_ignored_while_stopped(qapp):
    sync = _make_sync(qapp)
    clock = _FakeClock()
    sync._audio_clock = clock  # e.g. mid-start; timer not running
    sync._on_playback_channel_changed()
    assert sync._audio_clock is clock
    assert not clock.stopped


def test_cleanup_stops_a_live_clock(qapp):
    sync = _make_sync(qapp)
    clock = _FakeClock()
    sync._audio_clock = clock
    sync.cleanup()
    assert clock.stopped
    assert sync._audio_clock is None
