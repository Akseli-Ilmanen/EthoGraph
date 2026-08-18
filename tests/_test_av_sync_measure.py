"""Tier-2 sync measurement: is AudioClock.elapsed_s() honest on THIS machine?

Plays a click train through the real ``AudioClock`` on the real output device
while (a) logging ``elapsed_s()`` against ``time.perf_counter()`` and
(b) loopback-recording the speaker output via the Realtek "Stereomix" input.
For every click the difference between *when the clock claimed the click's
media time was reached* and *when the click was physically audible* is the
sync error, in milliseconds — no ears involved.

Positive lead = the marker runs AHEAD of the sound (the reported desync).
Accuracy is limited by the recorder-side timestamping (~±20-30 ms); that error
is systematic, so comparing runs (MME vs WASAPI output) cancels it.

Run from the repo root:

    conda run -n ethograph python tests/_test_av_sync_measure.py [--wasapi]

Requires: speakers unmuted (Stereo Mix taps the post-mix signal; on some
drivers a muted output records silence). Prefixed ``_test_`` so pytest skips
it — this needs a physical audio device and makes noise.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import sounddevice as sd

from ethograph.gui.audio_clock import AudioClock

MEDIA_FS = 24414.0  # the tester's TDT rate — the awkward real-world case
N_CLICKS = 8
CLICK_PERIOD_S = 1.0
FIRST_CLICK_S = 0.5
CLICK_LEN_S = 0.005
CLICK_FREQ_HZ = 2000.0
TAIL_S = 1.0


def build_click_train() -> tuple[np.ndarray, list[float]]:
    dur = FIRST_CLICK_S + (N_CLICKS - 1) * CLICK_PERIOD_S + TAIL_S
    data = np.zeros(int(dur * MEDIA_FS), dtype="float32")
    t = np.arange(int(CLICK_LEN_S * MEDIA_FS)) / MEDIA_FS
    burst = (0.7 * np.sin(2 * np.pi * CLICK_FREQ_HZ * t) * np.hanning(len(t))).astype("float32")
    times = [FIRST_CLICK_S + k * CLICK_PERIOD_S for k in range(N_CLICKS)]
    for ct in times:
        i = int(ct * MEDIA_FS)
        data[i : i + len(burst)] = burst
    return data, times


def find_stereomix() -> int | None:
    for i, d in enumerate(sd.query_devices()):
        name = d["name"].lower()
        if ("stereomix" in name or "stereo mix" in name or "loopback" in name) and d["max_input_channels"] > 0:
            return i
    return None


class Recorder:
    """Loopback capture with a perf_counter wall-time stamp per block."""

    def __init__(self, device: int):
        info = sd.query_devices(device)
        self.fs = float(info["default_samplerate"])
        self.blocks: list[tuple[float, int, np.ndarray]] = []  # (t_cb, frames_before, mono)
        self._frames = 0
        self.stream = sd.InputStream(
            device=device, channels=min(2, info["max_input_channels"]),
            samplerate=self.fs, dtype="float32", callback=self._cb,
        )

    def _cb(self, indata, frames, time_info, status):
        self.blocks.append((time.perf_counter(), self._frames, indata.mean(axis=1).copy()))
        self._frames += frames

    def wall_time_of_sample(self, n: int) -> float:
        """Map recorded sample index -> perf_counter wall time.

        A callback fires just after its block finished being captured, so the
        block's first sample was heard ~frames/fs before t_cb, minus the
        device's input latency. Systematic error here shifts every click
        equally.
        """
        lat = float(self.stream.latency or 0.0)
        for t_cb, start, mono in self.blocks:
            if start <= n < start + len(mono):
                return t_cb - (start + len(mono) - n) / self.fs - lat
        raise ValueError(f"sample {n} not recorded")

    def audio(self) -> np.ndarray:
        return np.concatenate([m for _, _, m in self.blocks]) if self.blocks else np.zeros(0, "float32")


def detect_clicks(rec: np.ndarray, fs: float, n_expected: int) -> list[int]:
    """Onset sample of each click: envelope threshold with a refractory gap."""
    env = np.abs(rec)
    k = max(1, int(0.002 * fs))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    floor = np.median(env)
    thresh = max(floor * 8, env.max() * 0.25)
    above = env > thresh
    onsets: list[int] = []
    i, refractory = 0, int(0.5 * CLICK_PERIOD_S * fs)
    while i < len(above):
        if above[i]:
            onsets.append(i)
            i += refractory
        else:
            i += 1
    if len(onsets) != n_expected:
        print(f"  ! detected {len(onsets)} clicks, expected {n_expected} "
              f"(floor={floor:.4f}, peak={env.max():.4f}) — check Stereo Mix / volume")
    return onsets


def run_measurement(use_wasapi: bool) -> None:
    data, click_times = build_click_train()
    mix = find_stereomix()
    if mix is None:
        sys.exit("No Stereo Mix / loopback input device found — enable it in the Windows sound settings.")

    if use_wasapi:
        wasapi = next(i for i, ha in enumerate(sd.query_hostapis()) if "WASAPI" in ha["name"])
        sd.default.device = (None, sd.query_hostapis(wasapi)["default_output_device"])
    out_dev = sd.query_devices(sd.default.device[1]) if sd.default.device[1] is not None else sd.query_devices(kind="output")
    host = sd.query_hostapis(out_dev["hostapi"])["name"]
    print(f"Output: {out_dev['name']!r} via {host}")

    rec = Recorder(mix)
    rec.stream.start()
    time.sleep(0.3)  # recorder warm before Play

    clock = AudioClock(data, MEDIA_FS)
    log: list[tuple[float, float]] = []
    if not clock.start():
        sys.exit("AudioClock.start() failed — no output stream.")
    t_deadline = time.perf_counter() + clock.duration_s + 3.0
    while not clock.finished and time.perf_counter() < t_deadline:
        log.append((time.perf_counter(), clock.elapsed_s()))
        time.sleep(0.002)
    anchor, bad = clock._dac_anchor, clock._dac_bad
    clock.stop()
    time.sleep(0.3)
    rec.stream.stop()
    rec.stream.close()

    print(f"DAC anchor published: {anchor is not None}   rejected time_infos: {bad}")
    print(f"output stream latency (reported): {clock._latency_s:.3f}s" if clock._latency_s else "")

    audio = rec.audio()
    onsets = detect_clicks(audio, rec.fs, N_CLICKS)
    walls, elapsed = np.array([t for t, _ in log]), np.array([e for _, e in log])

    print(f"\n{'click':>5} {'media_t':>8} {'audible(wall)':>14} {'clock(wall)':>12} {'lead_ms':>8}")
    leads = []
    for k, onset in enumerate(onsets[: len(click_times)]):
        m = click_times[k]
        w_audible = rec.wall_time_of_sample(onset)
        idx = np.searchsorted(elapsed, m)
        if idx == 0 or idx >= len(elapsed):
            continue
        # interpolate the wall time at which elapsed_s crossed the click's media time
        e0, e1 = elapsed[idx - 1], elapsed[idx]
        w0, w1 = walls[idx - 1], walls[idx]
        w_clock = w0 + (w1 - w0) * ((m - e0) / (e1 - e0) if e1 > e0 else 0.0)
        lead_ms = (w_audible - w_clock) * 1000.0  # >0: clock reached m BEFORE it was audible → marker leads
        leads.append(lead_ms)
        print(f"{k:>5} {m:>8.3f} {w_audible:>14.3f} {w_clock:>12.3f} {lead_ms:>8.1f}")

    if leads:
        a = np.array(leads)
        print(f"\nmarker leads audio by {a.mean():+.1f} ms  (std {a.std():.1f}, n={len(a)})")
        print("positive = marker/video ahead of the sound; |value| < ~45 ms is imperceptible")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wasapi", action="store_true", help="route output via WASAPI instead of the default (MME)")
    run_measurement(p.parse_args().wasapi)
