"""Clicking a labelled segment must play back *that* segment.

The existing click tests (``test_plot_click_sync``) only prove a click reaches
the labels widget. These prove the thing the user actually cares about: the
audio handed to the device, and the video frames shown, correspond to the
label's own onset/offset — and to *each other*.

A constant offset between what you see and what you hear is exactly what these
catch: the video side converts through ``time_to_frame`` (which applies the
camera's ``time_offset``) while the audio side slices the file at raw seconds,
so a non-zero video offset silently desynchronises the two.

Runs against both dataset fixtures, since a corrupt or differently-aligned
session shows up as one passing and the other failing.
"""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from ethograph.labels.intervals import empty_intervals

ONSET_S = 1.0
OFFSET_S = 1.5


def _arm_one_label(meta, onset_s: float = ONSET_S, offset_s: float = OFFSET_S):
    """Put a single known interval in place and select it for playback."""
    import pandas as pd

    df = empty_intervals()
    row = {"onset_s": onset_s, "offset_s": offset_s, "labels": 1, "individual": 0, "event_type": "state"}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    meta.app_state.label_intervals = df
    lw = meta.labels_widget
    lw.current_labels_pos = df.index[-1]
    return lw


def _capture_playback(meta, monkeypatch) -> dict:
    """Intercept the audio/video calls a segment playback makes."""
    captured: dict = {}
    video = meta.app_state.video

    if video is not None:
        monkeypatch.setattr(
            type(video),
            "_start_audio",
            lambda _self, t0, t1: captured.update(audio=(t0, t1)),
        )
        real_play = type(video).play_segment

        def spy(self, start_frame, end_frame, exact_t0=None, exact_t1=None):
            captured["frames"] = (start_frame, end_frame)
            return real_play(self, start_frame, end_frame, exact_t0=exact_t0, exact_t1=exact_t1)

        monkeypatch.setattr(type(video), "play_segment", spy)
    else:
        player = meta.plot_container.audio_player
        monkeypatch.setattr(
            type(player),
            "_build_clock",
            lambda _self, t0, t1: captured.update(audio=(t0, t1)) or None,
        )
    return captured


@pytest.fixture(params=["birdpark", "moll2025"])
def loaded_gui(request, birdpark_gui, moll2025_gui):
    """Both datasets, so a session-specific alignment problem is visible."""
    return request.param, (birdpark_gui if request.param == "birdpark" else moll2025_gui)


def test_segment_playback_uses_the_label_bounds(loaded_gui, monkeypatch):
    """Audio must be sliced at the label's own onset/offset, not near them."""
    name, (_shell, meta) = loaded_gui
    lw = _arm_one_label(meta)
    captured = _capture_playback(meta, monkeypatch)

    lw._play_segment()
    QApplication.processEvents()

    assert "audio" in captured, f"{name}: no audio was requested for the segment"
    t0, t1 = captured["audio"]
    assert t0 == pytest.approx(ONSET_S, abs=1e-6), f"{name}: audio starts at {t0}, label onset is {ONSET_S}"
    assert t1 == pytest.approx(OFFSET_S, abs=1e-6), f"{name}: audio ends at {t1}, label offset is {OFFSET_S}"


def test_video_and_audio_refer_to_the_same_instant(loaded_gui, monkeypatch):
    """The seen frame and the heard sample must be the same moment.

    ``time_to_frame`` subtracts the camera's ``time_offset``; the audio slice
    does not. With a non-zero offset the two drift apart by exactly that
    constant — heard as "roughly right but shifted".
    """
    name, (_shell, meta) = loaded_gui
    if meta.app_state.video is None:
        pytest.skip(f"{name}: audio-only session, no video/audio pairing to check")

    lw = _arm_one_label(meta)
    captured = _capture_playback(meta, monkeypatch)

    lw._play_segment()
    QApplication.processEvents()

    assert "frames" in captured and "audio" in captured, f"{name}: playback did not start"
    start_frame, _end_frame = captured["frames"]
    audio_t0, _audio_t1 = captured["audio"]

    video_t0 = meta.app_state.video.frame_to_time(start_frame)
    frame_period = 1.0 / meta.app_state.video.fps
    drift = abs(video_t0 - audio_t0)
    assert drift <= frame_period, (
        f"{name}: video starts at {video_t0:.4f}s but audio at {audio_t0:.4f}s "
        f"— a constant {drift:.4f}s offset (video time_offset="
        f"{getattr(meta.app_state.video, '_time_offset', None)})"
    )


def test_the_marker_starts_on_the_label_onset(loaded_gui, monkeypatch):
    """Whatever the media does, the red marker must land on the label."""
    name, (_shell, meta) = loaded_gui
    lw = _arm_one_label(meta)
    _capture_playback(meta, monkeypatch)

    captured_marker = []
    meta.plot_container.time_marker_updated.connect(captured_marker.append)
    lw._play_segment()
    QApplication.processEvents()
    assert captured_marker, f"{name}: the time marker never moved"
    marker = captured_marker[-1]
    assert marker == pytest.approx(ONSET_S, abs=0.05), f"{name}: marker at {marker}, label onset is {ONSET_S}"


def test_a_point_event_is_refused_not_played(loaded_gui, monkeypatch):
    """offset_s is NaN for point events — playing one must be a no-op."""
    name, (_shell, meta) = loaded_gui
    lw = _arm_one_label(meta, onset_s=1.0, offset_s=np.nan)
    captured = _capture_playback(meta, monkeypatch)

    lw._play_segment()
    QApplication.processEvents()

    assert "audio" not in captured, f"{name}: a point event started playback"
