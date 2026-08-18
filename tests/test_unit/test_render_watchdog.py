"""The video render chain survives exceptions and stalls.

pynaviz's ``PlotVideo.animate`` re-arms itself only as its own last line
(``self.canvas.request_draw(self.animate)``) and catches nothing but
``queue.Empty`` — any other raise, or one paint dropped by the rendercanvas
Qt scheduler, ends the chain permanently: audio and the playhead keep moving
while the image freezes (Windows tester capture: ``animate()`` not run for
~1000 s). ``install_animate_guard`` keeps the chain alive across raises and
stamps a heartbeat; ``nudge_stalled_render`` (driven by VideoSync's playback
watchdog) re-arms a chain whose heartbeat went stale.
"""

from __future__ import annotations

from ethograph.gui.pygfx_video import (
    ANIMATE_STALL_S,
    install_animate_guard,
    nudge_stalled_render,
)


class _FakeCanvas:
    def __init__(self):
        self.draws: list = []

    def request_draw(self, fn):
        self.draws.append(fn)


class _FakePlot:
    """PlotVideo stand-in: animate() re-arms itself like the real one."""

    def __init__(self):
        self.canvas = _FakeCanvas()
        self.calls = 0
        self.raise_once: Exception | None = None

    def animate(self):
        self.calls += 1
        if self.raise_once is not None:
            exc, self.raise_once = self.raise_once, None
            raise exc
        self.canvas.request_draw(self.animate)


class _FakeTime:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_normal_animate_stays_wrapped():
    plot = _FakePlot()
    install_animate_guard(plot)
    wrapper = plot.animate
    plot.animate()
    # The original's own re-arm (self.animate) resolves to the wrapper, so
    # the whole chain stays guarded.
    assert plot.canvas.draws == [wrapper]
    assert plot.calls == 1


def test_raise_does_not_end_the_chain():
    plot = _FakePlot()
    install_animate_guard(plot)
    plot.raise_once = RuntimeError("texture update failed")
    plot.animate()
    # The raise skipped the original's re-arm; the guard re-armed instead.
    assert plot.canvas.draws == [plot.animate]


def test_heartbeat_is_stamped_per_call():
    plot = _FakePlot()
    clock = _FakeTime()
    install_animate_guard(plot, clock=clock)
    clock.t = 5.0
    plot.animate()
    assert plot._eto_last_animate == 5.0


def test_nudge_ignores_a_fresh_heartbeat():
    plot = _FakePlot()
    clock = _FakeTime()
    install_animate_guard(plot, clock=clock)
    clock.t = ANIMATE_STALL_S / 2
    assert not nudge_stalled_render(plot, clock=clock)
    assert plot.canvas.draws == []


def test_nudge_rearms_a_stale_chain():
    plot = _FakePlot()
    clock = _FakeTime()
    install_animate_guard(plot, clock=clock)
    clock.t = ANIMATE_STALL_S * 3
    assert nudge_stalled_render(plot, clock=clock)
    assert plot.canvas.draws == [plot.animate]


def test_nudge_never_rearms_a_disarmed_canvas():
    plot = _FakePlot()
    clock = _FakeTime()
    install_animate_guard(plot, clock=clock)
    # _disarm_present() installs the no-op as an instance attribute; a
    # disarmed canvas is being torn down and must be left alone.
    plot.canvas._rc_request_paint = lambda: None
    clock.t = ANIMATE_STALL_S * 3
    assert not nudge_stalled_render(plot, clock=clock)
    assert plot.canvas.draws == []


def test_guard_respects_disarm_too():
    plot = _FakePlot()
    install_animate_guard(plot)
    plot.canvas._rc_request_paint = lambda: None
    plot.raise_once = RuntimeError("during teardown")
    plot.animate()
    assert plot.canvas.draws == []
