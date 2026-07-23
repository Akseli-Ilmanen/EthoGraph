"""Generate the synthetic 2-trial audio/video dataset from the "Suggested
Synthetic Test Dataset" section of ehtograph_feedback_report.md, so Luca's
audio-sync/choppiness/zoom-stall/embedded-AAC issues can be reproduced
without the private biological videos.

Produces, under --out-dir (default ``data/synthetic_repro/``):

    trial{1,2}_video.mp4                    no-audio proxy-res video (for
                                             external-WAV playback tests)
    trial{1,2}_video_embeddedaudio.mp4       same video muxed with AAC audio
                                             (repro step: "embedded MP4/AAC")
    trial{1,2}_audio_mic-1.wav                mono PCM16 16 kHz (primary,
                                             linked in alignment.nwb)
    trial{1,2}_audio_mic-1_8k.wav             same content, 8 kHz PCM16
    trial{1,2}_audio_mic-1_24414hz_f32.wav    same content, 24414 Hz float32
    session.nc                              2-trial TrialTree (video_motion
                                             feature per trial)
    session_labels.tsv                      a few state + point labels,
                                             Male/Female individuals
    .ethograph/alignment.nwb                video_cam-1 + audio_mic-1 linked
                                             per trial
    .ethograph/mapping.txt                   label id -> name/branch/type

Video content: a moving frame-number/timestamp burn-in (so you can eyeball
drift) plus a full-white flash every ~5 s. The WAV files carry a short tone
click at those same flash times, so audio/video sync error is visible on
sight: if the flash and the click move apart under playback, that's drift.

Requires ffmpeg on PATH. Run from the ethograph conda env (needs numpy,
soundfile, pandas, xarray, ethograph itself for the alignment/label writers).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 47.683716
WIDTH, HEIGHT = 1280, 936
FLASH_INTERVAL_S = 5.0
FLASH_PERIOD_FRAMES = round(FPS * FLASH_INTERVAL_S)
AUDIO_RATE_PRIMARY = 16000


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr[-4000:]}")


def flash_times(duration_s: float) -> list[float]:
    """Times (s) of the video's white-flash frames, source for audio clicks too."""
    n_frames = int(duration_s * FPS)
    return [k * FLASH_PERIOD_FRAMES / FPS for k in range(n_frames // FLASH_PERIOD_FRAMES + 1)]


def make_video(out_path: Path, label: str, duration_s: float) -> None:
    """No-audio H.264 video: burnt-in frame/time counter + periodic white flash."""
    vf_parts = [
        f"drawtext=text='{label} frame %{{n}} t=%{{pts}}s':"
        "fontcolor=white:fontsize=32:x=20:y=60:box=1:boxcolor=black@0.6"
    ]
    vf_parts.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='eq(mod(n,{FLASH_PERIOD_FRAMES}),0)'")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration={duration_s}",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out_path),
    ]
    _run_ffmpeg(cmd)


def make_click_audio(
    out_path: Path,
    duration_s: float,
    samplerate: int,
    subtype: str,
    click_hz: float = 1000.0,
    click_ms: float = 15.0,
) -> None:
    """Mono WAV: near-silence with a short tone click at every video flash time."""
    import soundfile as sf

    n_samples = int(round(duration_s * samplerate))
    audio = np.zeros(n_samples, dtype=np.float64)

    click_len = int(round(click_ms / 1000 * samplerate))
    t_click = np.arange(click_len) / samplerate
    envelope = np.sin(np.pi * np.arange(click_len) / max(click_len - 1, 1)) ** 2
    tone = np.sin(2 * np.pi * click_hz * t_click) * envelope

    for t in flash_times(duration_s):
        i0 = int(round(t * samplerate))
        i1 = min(i0 + click_len, n_samples)
        if i0 >= n_samples:
            continue
        audio[i0:i1] += tone[: i1 - i0]

    audio = np.clip(audio, -1.0, 1.0)
    dtype = np.float32 if subtype == "FLOAT" else np.float64
    sf.write(str(out_path), audio.astype(dtype), samplerate, subtype=subtype)


def mux_embedded_audio(video_path: Path, wav_path: Path, out_path: Path) -> None:
    """Copy of video_path with wav_path's audio muxed in as AAC (repro: embedded MP4/AAC)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(wav_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(out_path),
    ]
    _run_ffmpeg(cmd)


def build_session(out_dir: Path, video_paths: list[Path], audio_paths: list[Path], trial_gap_s: float) -> None:
    """Write session.nc, alignment.nwb, session_labels.tsv, mapping.txt."""
    import xarray as xr

    import ethograph as eto
    from ethograph.features.movement import extract_video_motion
    from ethograph.io.nwb_alignment import align_media_per_trial
    from ethograph.labels.tsv_store import labels_tsv_path, save_labels_tsv

    datasets = []
    for i, vpath in enumerate(video_paths, start=1):
        motion = extract_video_motion(str(vpath), fps=FPS, time_coord_name="time_video")
        ds = xr.Dataset({"video_motion": motion}, coords={"individuals": ["Male", "Female"]})
        ds.attrs["trial"] = i
        ds.attrs["fps"] = FPS
        datasets.append(ds)

    tree = eto.from_datasets(datasets)
    nc_path = out_dir / "session.nc"
    tree.to_netcdf(str(nc_path))

    durations = [float(motion.sizes["time_video"]) / FPS for motion in (ds["video_motion"] for ds in datasets)]
    starts = []
    cursor = 0.0
    for d in durations:
        starts.append(cursor)
        cursor += d + trial_gap_s

    ethograph_dir = out_dir / ".ethograph"
    trial_table = pd.DataFrame(
        {
            "trial": [1, 2],
            "start_time": starts,
            "stop_time": [s + d for s, d in zip(starts, durations)],
            "video_cam-1": [p.name for p in video_paths],
            "audio_mic-1": [p.name for p in audio_paths],
        }
    )
    align_media_per_trial(
        trial_table,
        stream_rates={"video": FPS, "audio": AUDIO_RATE_PRIMARY},
        output_path=ethograph_dir / "alignment.nwb",
        media_root=out_dir,
    )

    mapping_lines = [
        "1 approach 0 state",
        "2 avoid 0 state",
        "3 vocalize 0 point",
    ]
    (ethograph_dir / "mapping.txt").write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")

    rows = []

    def add(trial, individual, label_id, onset, offset):
        rows.append(
            {
                "trial": trial,
                "individual": individual,
                "labels": label_id,
                "onset_s": onset,
                "offset_s": offset,
                "human_verified": 1,
                "changepoint_corrected": 0,
                "prediction_source": "",
                "n_samples": 0,
            }
        )

    for trial_idx, duration in enumerate(durations, start=1):
        flashes = flash_times(duration)
        if len(flashes) < 2:
            continue

        def pick(i: int) -> float:
            return flashes[min(i, len(flashes) - 1)]

        add(trial_idx, "Male", 1, pick(2), min(pick(2) + 2.0, duration))
        add(trial_idx, "Female", 2, pick(4), min(pick(4) + 1.5, duration))
        add(trial_idx, "Male", 3, pick(6), pick(6))
        add(trial_idx, "Female", 3, pick(8), pick(8))

    labels_df = pd.DataFrame(rows)
    save_labels_tsv(labels_tsv_path(nc_path), labels_df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic_repro"))
    parser.add_argument("--duration-s", type=float, default=120.0, help="Duration of each of the 2 trials.")
    parser.add_argument(
        "--trial-gap-s",
        type=float,
        default=20.0,
        help="Session-time gap between trial 1's end and trial 2's start "
        "(mirrors Luca's non-contiguous trial windows; also what triggers "
        "the dense-timestamp alignment.nwb bloat, issue #7).",
    )
    parser.add_argument(
        "--skip-extra-rates",
        action="store_true",
        help="Skip the optional 8kHz / 24414Hz-float32 WAV variants (only PCM16 16kHz).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".ethograph").mkdir(exist_ok=True)

    video_paths = []
    audio_paths = []
    for i in (1, 2):
        label = f"TRIAL {i}"
        vpath = out_dir / f"trial{i}_video.mp4"
        apath = out_dir / f"trial{i}_audio_mic-1.wav"
        print(f"[{i}/2] video -> {vpath.name}")
        make_video(vpath, label, args.duration_s)
        print(f"[{i}/2] audio -> {apath.name}")
        make_click_audio(apath, args.duration_s, AUDIO_RATE_PRIMARY, subtype="PCM_16")

        if not args.skip_extra_rates:
            a8k = out_dir / f"trial{i}_audio_mic-1_8k.wav"
            a24f32 = out_dir / f"trial{i}_audio_mic-1_24414hz_f32.wav"
            print(f"[{i}/2] audio -> {a8k.name}")
            make_click_audio(a8k, args.duration_s, 8000, subtype="PCM_16")
            print(f"[{i}/2] audio -> {a24f32.name}")
            make_click_audio(a24f32, args.duration_s, 24414, subtype="FLOAT")

        embedded = out_dir / f"trial{i}_video_embeddedaudio.mp4"
        print(f"[{i}/2] mux embedded AAC -> {embedded.name}")
        mux_embedded_audio(vpath, apath, embedded)

        video_paths.append(vpath)
        audio_paths.append(apath)

    print("Building session.nc + alignment.nwb + labels...")
    build_session(out_dir, video_paths, audio_paths, args.trial_gap_s)

    print(f"\nDone. Load {out_dir / 'session.nc'} in the GUI (video/audio folder = {out_dir}).")


if __name__ == "__main__":
    main()
