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
CLICK_LEN_S = 0.02  # long enough to carry acoustic energy to a room mic
CLICK_FREQ_HZ = 2000.0
TAIL_S = 1.0


def build_click_train() -> tuple[np.ndarray, list[float]]:
    dur = FIRST_CLICK_S + (N_CLICKS - 1) * CLICK_PERIOD_S + TAIL_S
    data = np.zeros(int(dur * MEDIA_FS), dtype="float32")
    t = np.arange(int(CLICK_LEN_S * MEDIA_FS)) / MEDIA_FS
    burst = (0.9 * np.sin(2 * np.pi * CLICK_FREQ_HZ * t) * np.hanning(len(t))).astype("float32")
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

    def __init__(self, device: int, latency: str = "high"):
        info = sd.query_devices(device)
        self.fs = float(info["default_samplerate"])
        self.blocks: list[tuple[float, int, np.ndarray]] = []  # (t_cb, frames_before, mono)
        self._frames = 0
        self.stream = sd.InputStream(
            device=device, channels=min(2, info["max_input_channels"]),
            samplerate=self.fs, dtype="float32", callback=self._cb, latency=latency,
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
    """Onset sample of each click via a matched (narrowband 2 kHz) filter.

    Robust to faint acoustic pickup: room noise carries little energy in a
    narrow band around CLICK_FREQ_HZ, so the n_expected strongest narrowband
    peaks are the clicks. Peak centre → onset by subtracting half the click
    length; spacing is validated against the known 1 s period.
    """
    t = np.arange(len(rec)) / fs
    analytic = rec * np.exp(-2j * np.pi * CLICK_FREQ_HZ * t)
    k = max(1, int(CLICK_LEN_S * fs))
    env = np.abs(np.convolve(analytic, np.ones(k) / k, mode="same"))
    work = env.copy()
    refractory = int(0.5 * CLICK_PERIOD_S * fs)
    peaks: list[int] = []
    for _ in range(n_expected):
        i = int(np.argmax(work))
        if work[i] <= 0:
            break
        peaks.append(i)
        work[max(0, i - refractory) : i + refractory] = 0
    peaks.sort()
    gaps = np.diff(peaks) / fs
    good = np.abs(gaps - CLICK_PERIOD_S) < 0.05
    if not good.all():
        print(f"  ! click spacing off (gaps: {np.round(gaps, 3)}) — noise contaminated the detection")
    print(f"  narrowband SNR: peak {env.max():.4f} vs floor {np.median(env):.5f}")
    return [p - int(CLICK_LEN_S / 2 * fs) for p in peaks]


def run_measurement(use_wasapi: bool, input_latency: str = "high", input_device: int | None = None) -> None:
    data, click_times = build_click_train()
    mix = input_device if input_device is not None else find_stereomix()
    if mix is None:
        sys.exit("No Stereo Mix / loopback input device found — enable it in the Windows sound settings.")
    print(f"Recorder device: {sd.query_devices(mix)['name']!r}")

    if use_wasapi:
        wasapi = next(i for i, ha in enumerate(sd.query_hostapis()) if "WASAPI" in ha["name"])
        sd.default.device = (None, sd.query_hostapis(wasapi)["default_output_device"])
    out_dev = sd.query_devices(sd.default.device[1]) if sd.default.device[1] is not None else sd.query_devices(kind="output")
    host = sd.query_hostapis(out_dev["hostapi"])["name"]
    print(f"Output: {out_dev['name']!r} via {host}")

    rec = Recorder(mix, latency=input_latency)
    rec.stream.start()
    print(f"Recorder: input latency requested {input_latency!r}, reported {float(rec.stream.latency or 0.0):.3f}s")
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
    p.add_argument("--input-latency", choices=["low", "high"], default="high",
                   help="recorder-side buffering; if the measured lead moves with this, the bias is in the recorder")
    p.add_argument("--input-device", type=int, default=None,
                   help="capture device index (default: auto-detect Stereo Mix); use a real microphone "
                        "to cross-check the loopback path — acoustic delay is ~0")
    args = p.parse_args()
    run_measurement(args.wasapi, args.input_latency, args.input_device)
