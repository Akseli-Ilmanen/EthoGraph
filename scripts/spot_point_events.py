"""Proof of principle: can a pixel model spot the stick/pellet point events?

The onset model (``ethograph/labels/onset_model.py``) reads point events off
kinematic time series. This script asks whether the *pixels* carry the same
information at a useful precision, using E2E-Spot (Hong et al., ECCV 2022,
BSD-3, cloned into ``spot/``) — the reference model for **precise event
spotting**: one anchor timestamp per event, scored at a frame-level tolerance
rather than in seconds.

The question is deliberately narrow, because the answer decides whether any
of this belongs in the GUI: **train on N-1 sessions, predict the held-out
one, and report the error in frames and in milliseconds.** If a held-out
session cannot be predicted to a few frames there is nothing to integrate.

Two properties of this data make it an unusually clean test:

* every trial is its own short video (4-9 s at 200 fps), so a clip needs no
  windowing around the event and the model cannot learn "the event is always
  N seconds in" — a leak that trimming to a window would introduce;
* labels 31 and 32 come as a pair, one of each per trial, which is the same
  shape as the figure-skating ``*_takeoff``/``*_landing`` classes E2E-Spot
  was evaluated on.

200 fps is the hard part. The literature's delta = 1-2 frame tolerances were
set at 25 fps, where one frame is 40 ms; here one frame is 5 ms, so a "3
frame" target is 15 ms and is far stricter than anything the benchmarks
report. :func:`score` therefore reports a *sweep* over tolerances in both
units, so the trade between precision and hit rate is visible rather than
hidden in a single number.

Usage (from the repo root, with the ``ethograph`` env active)::

    python scripts/spot_point_events.py plan
    python scripts/spot_point_events.py export --max-trials 60
    python scripts/spot_point_events.py train
    python scripts/spot_point_events.py score spot/runs/crow_pellet/pred-test.recall.json.gz

``train`` shells out to ``spot/train_e2e.py`` rather than reimplementing it:
a negative result then belongs to the task, not to a re-implementation. That
needs ``timm``, which must be installed **without** touching the CUDA torch
build already in the environment::

    pip install --no-deps timm

What this is a step toward
--------------------------

If a held-out session comes back within a useful tolerance, the proposal is
**not** a new subsystem but a second *backend* behind the onset model's
existing workflow. E2E-Spot's head emits a per-frame probability per class,
which is the same object :func:`~ethograph.labels.onset_model.target_curves`
produces — so ``tallest_peak``, the confidence written to a label, the
``~/.ethograph/models/{name}`` store, ``onset_curves.npz`` and its
frame-by-frame review overlay, ``automated`` -> curation -> ``curated``, and
``predict_onsets`` as a workflow step all carry over untouched. What changes
is the input side: the ``features`` config (feature -> dim -> values) becomes
a camera, a crop and a clip length, and a model's training data becomes
cached clips around each event rather than ``(T, D)`` arrays. The model files
themselves would be vendored following ADR 0001 — ``model/shift.py`` and its
two ``impl/`` modules only, never the toolbox layer or ``train_e2e.py`` —
and installed as ``ethograph[spot]``, which adds ``timm`` and leaves torch
out-of-band as the ``model`` extra already does.

The JPEG frame dump this script writes is **not** part of that proposal. It
exists because E2E-Spot's dataloader samples random clips from long untrimmed
sports video, where seeking into H.264 costs a decode from the previous
keyframe. These trials are 4-9 s, so decoding one whole trial (~0.1 s for the
H.264 alone) yields every clip in it. In the GUI, **inference never dumps
frames** — a short video is decoded whole and sliced, a long one is seeked
(or read through ``io/video_proxy.py``'s short-GOP proxy where seeking is
dear). A dump is a *training-time* cache, and there it is a reasonable
default — clips around each event, not whole trials. The full trade-off is
tabulated in ``spot_point_events.md`` under *Costs*.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ethograph.io.nwb_alignment import NWBAlignment
from ethograph.video_features.frames import iter_frames

logger = logging.getLogger("spot_poc")

#: The point events this experiment predicts, and the names they get in
#: ``class.txt``. Both are written for every trial in the sessions used.
TARGET_LABELS: dict[int, str] = {31: "label_31", 32: "label_32"}

#: Camera to read. The alignment resolves the file; this only picks the view.
CAMERA = "cam-1"

#: E2E-Spot's frames are stored at a fixed height with the aspect preserved.
FRAME_HEIGHT = 224
JPEG_QUALITY = 85

#: Where the exported dataset lands. ``train_e2e.py`` resolves a dataset by
#: name against ``data/`` relative to its own working directory.
SPOT_ROOT = Path("spot")
DATASET_NAME = "crow_pellet"

#: Fraction of the *training* trials held out for E2E-Spot's model selection.
#: Drawn per trial, deterministically — the test session stays untouched.
VAL_FRACTION = 0.15
_SEED = 0

#: Pause before a crashed training run is resumed — long enough for the
#: GPU to release memory and a spawned worker pool to be torn down.
RETRY_PAUSE_S = 60

#: Rough bytes per exported JPEG at :data:`FRAME_HEIGHT`, for the disk
#: estimate ``plan`` prints. Measured, not assumed — re-measure if the
#: quality or the height changes.
_BYTES_PER_JPEG = 9_000

_SESSION_RE = re.compile(r"ses-\d+_date-(\d{8})_(\d+)$")


@dataclass(frozen=True)
class SessionPaths:
    """One session's three inputs: labels, alignment, videos."""

    name: str
    labels_tsv: Path
    alignment_nwb: Path
    video_dir: Path


def discover_sessions(derivatives_root: Path, viddata_root: Path) -> list[SessionPaths]:
    """Every session under *derivatives_root* that labels the target events.

    The video folder is ``{date}_{index}_{subject}`` under *viddata_root*,
    with the subject read off the ``sub-NN_id-Name`` folder — the alignment
    holds the file names, this only says which folder to look in.
    """
    subject = derivatives_root.name.split("id-")[-1]
    sessions: list[SessionPaths] = []
    for session_dir in sorted(derivatives_root.glob("ses-*")):
        match = _SESSION_RE.match(session_dir.name)
        if match is None:
            continue
        behav = session_dir / "behav"
        labels_tsv = behav / "Trial_data_labels.tsv"
        alignment = behav / ".ethograph" / "alignment.nwb"
        if not labels_tsv.is_file() or not alignment.is_file():
            continue
        if not read_point_events(labels_tsv):
            continue
        date, index = match.groups()
        sessions.append(
            SessionPaths(
                name=f"{date}_{index}",
                labels_tsv=labels_tsv,
                alignment_nwb=alignment,
                video_dir=viddata_root / f"{date}_{index}_{subject}",
            )
        )
    return sessions


def read_point_events(labels_tsv: Path) -> dict[int, dict[int, float]]:
    """``{trial: {label: onset_s}}`` for the target point events.

    Onsets are trial-relative, as every labels TSV is; converting them to the
    video's clock is :func:`event_frames`' job.
    """
    df = pd.read_csv(labels_tsv, sep="\t", comment="#")
    points = df[(df["event_type"] == "point") & (df["labels"].isin(TARGET_LABELS))]
    out: dict[int, dict[int, float]] = {}
    for row in points.itertuples():
        out.setdefault(int(row.trial), {})[int(row.labels)] = float(row.onset_s)
    return out


def event_frames(onsets: dict[int, float], offset: float, fps: float) -> dict[int, int]:
    """Trial-relative onsets to video frame indices.

    ``VideoSync``'s convention is ``trial = video + offset``, so a trial time
    samples the video at ``t - offset``. Getting this backwards shifts every
    label by the offset and is invisible in the result, which is why the
    conversion is written once, here.
    """
    return {label: int(round((t - offset) * fps)) for label, t in onsets.items()}


@dataclass
class TrialRecord:
    """One exported trial, in E2E-Spot's per-video schema."""

    video_id: str
    video_path: Path
    num_frames: int
    fps: float
    width: int
    height: int
    events: dict[int, int]

    def to_json(self) -> dict:
        return {
            "video": self.video_id,
            "num_frames": self.num_frames,
            "num_events": len(self.events),
            "events": [
                {"frame": frame, "label": TARGET_LABELS[label], "comment": ""}
                for label, frame in sorted(self.events.items())
            ],
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }


def _probe(video: Path) -> tuple[float, int, int, int]:
    """``(fps, n_frames, width, height)`` of one video."""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None:
            raise ValueError(f"Cannot determine frame rate of {video}")
        return float(rate), int(stream.frames), stream.codec_context.width, stream.codec_context.height


def plan_session(session: SessionPaths, max_trials: int | None) -> list[TrialRecord]:
    """What :func:`export_frames` would write, without decoding anything."""
    alignment = NWBAlignment(session.alignment_nwb)
    events = read_point_events(session.labels_tsv)
    records: list[TrialRecord] = []
    for trial in sorted(events)[:max_trials]:
        found = alignment.resolve_media_path(trial, "video", device=CAMERA, fallback_folder=str(session.video_dir))
        if not found:
            logger.warning("%s trial %s: no %s video, skipped", session.name, trial, CAMERA)
            continue
        video = Path(found)
        fps, n_frames, width, height = _probe(video)
        offset = float(alignment.stream_offset_for_trial(trial, "video", device=CAMERA))
        frames = event_frames(events[trial], offset, fps)
        outside = {label: f for label, f in frames.items() if not 0 <= f < n_frames}
        if outside:
            logger.warning("%s trial %s: events %s fall outside the video, skipped", session.name, trial, outside)
            continue
        scale = FRAME_HEIGHT / height
        records.append(
            TrialRecord(
                video_id=f"{session.name}_trial{trial}",
                video_path=video,
                num_frames=n_frames,
                fps=fps,
                width=int(round(width * scale)),
                height=FRAME_HEIGHT,
                events=frames,
            )
        )
    return records


def export_frames(record: TrialRecord, frame_root: Path) -> int:
    """Write one trial's frames as ``{frame_root}/{video_id}/{i:06d}.jpg``.

    Returns the number written, which the caller trusts over the container's
    claimed frame count. A trial whose folder already holds the expected
    count is left alone, so an interrupted export resumes rather than
    starting over.

    The cost here is neither the H.264 decode (~9600 fps) nor the JPEG write
    (~0.4 s per trial) but the YUV to RGB conversion in ``to_ndarray``, which
    is why :func:`export_all` spends cores on it rather than optimising the
    encode.
    """
    out_dir = frame_root / record.video_id
    if out_dir.is_dir() and len(list(out_dir.glob("*.jpg"))) == record.num_frames:
        return record.num_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (record.width, record.height)
    written = 0
    for i, frame in enumerate(iter_frames(record.video_path)):
        Image.fromarray(frame).resize(size, Image.BILINEAR).save(out_dir / f"{i:06d}.jpg", quality=JPEG_QUALITY)
        written += 1
    return written


def _export_one(job: tuple[TrialRecord, Path]) -> tuple[str, int]:
    """Pool entry point: ``(video_id, frames written)``."""
    record, frame_root = job
    return record.video_id, export_frames(record, frame_root)


def export_all(records: list[TrialRecord], frame_root: Path, workers: int) -> None:
    """Export every trial, in parallel, correcting frame counts in place.

    Trials are independent and the work is CPU-bound colour conversion, so
    this scales with cores. The frame count a worker reports replaces the
    container's, since the JSON must describe what is actually on disk.
    """
    by_id = {r.video_id: r for r in records}
    jobs = [(r, frame_root) for r in records]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for video_id, written in pool.map(_export_one, jobs, chunksize=1):
            record = by_id[video_id]
            if written != record.num_frames:
                logger.warning("%s: decoded %d frames, container claimed %d", video_id, written, record.num_frames)
                record.num_frames = written
            done += 1
            if done % 25 == 0 or done == len(jobs):
                logger.info("  %d/%d trials", done, len(jobs))


def exported_records(records: list[TrialRecord], frame_root: Path) -> list[TrialRecord]:
    """The records whose frames are fully on disk.

    A folder an interrupted worker left short is skipped and said so, with
    both counts, so a container that miscounted its own frames is told apart
    from a half-written trial.
    """
    kept: list[TrialRecord] = []
    for record in records:
        on_disk = len(list((frame_root / record.video_id).glob("*.jpg")))
        if on_disk == 0:
            continue
        if on_disk != record.num_frames:
            logger.warning("%s: %d frames on disk, video has %d — skipped", record.video_id, on_disk, record.num_frames)
            continue
        kept.append(record)
    return kept


def assign_splits(records: dict[str, list[TrialRecord]], test_session: str) -> dict[str, list[TrialRecord]]:
    """Leave-one-session-out, with a validation slice cut from the rest.

    The held-out session is never touched: model selection happens on trials
    of the *training* sessions, so the reported number is what the model does
    on a session it has never seen — which is the question being asked.
    """
    if test_session not in records:
        raise ValueError(f"Unknown test session {test_session!r}; have {sorted(records)}")
    rng = np.random.default_rng(_SEED)
    train: list[TrialRecord] = []
    val: list[TrialRecord] = []
    for name, session_records in sorted(records.items()):
        if name == test_session:
            continue
        held = rng.random(len(session_records)) < VAL_FRACTION
        val.extend(r for r, h in zip(session_records, held) if h)
        train.extend(r for r, h in zip(session_records, held) if not h)
    return {"train": train, "val": val, "test": records[test_session]}


def write_dataset(splits: dict[str, list[TrialRecord]], dataset_dir: Path) -> None:
    """Write ``class.txt`` and one JSON per split, in E2E-Spot's layout."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    names = "\n".join(TARGET_LABELS[label] for label in sorted(TARGET_LABELS))
    (dataset_dir / "class.txt").write_text(names + "\n", encoding="utf-8")
    for split, records in splits.items():
        payload = [r.to_json() for r in records]
        (dataset_dir / f"{split}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        logger.info("%s: %d videos, %d events", split, len(payload), sum(v["num_events"] for v in payload))


# ---------------------------------------------------------------------------
# Scoring — the verdict, in frames and in milliseconds
# ---------------------------------------------------------------------------

#: Tolerances the sweep reports. One frame is 5 ms at 200 fps, so the
#: literature's delta = 1-2 sits at the very left of this range.
TOLERANCES = (1, 2, 3, 4, 6, 10, 20, 40)


def best_per_class(entry: dict) -> dict[str, int]:
    """The highest-scoring predicted frame per class in one video.

    One event per class per trial is this task's constraint, and taking the
    top-scoring prediction is the same rule ``tallest_peak`` applies to the
    onset model's curve.
    """
    best: dict[str, tuple[float, int]] = {}
    for event in entry.get("events", []):
        label = event["label"]
        confidence = float(event.get("score", 1.0))
        if label not in best or confidence > best[label][0]:
            best[label] = (confidence, int(event["frame"]))
    return {label: frame for label, (_, frame) in best.items()}


def read_predictions(path: Path) -> list[dict]:
    """E2E-Spot's per-video predictions, from either file it writes.

    ``pred-{split}.{epoch}.json`` holds only the argmax — a class appears
    where its score beats background — while ``.recall.json.gz`` holds every
    frame scoring above a low threshold. For one event per class per trial
    the tallest peak of the class's own curve is the answer, whatever
    background says, so prefer the recall file; the argmax one still reads.
    """
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def score(predictions_path: Path, truth_path: Path) -> None:
    """Per-class hit rate over a sweep of tolerances, plus the error spread."""
    truth = {v["video"]: v for v in json.loads(truth_path.read_text(encoding="utf-8"))}
    predicted = read_predictions(predictions_path)

    errors: dict[str, list[int]] = {name: [] for name in TARGET_LABELS.values()}
    missing: dict[str, int] = {name: 0 for name in TARGET_LABELS.values()}
    rates: list[float] = []
    for entry in predicted:
        gt = truth.get(entry["video"])
        if gt is None:
            continue
        rates.append(float(gt["fps"]))
        # A strided run predicts on a downsampled clock and says so through
        # its fps; its frame indices come back to the truth's clock here.
        stride = float(gt["fps"]) / float(entry["fps"])
        if abs(stride - round(stride)) > 1e-6:
            raise ValueError(f"{entry['video']}: predicted at {entry['fps']} fps, truth at {gt['fps']} — not a stride")
        best = {label: frame * int(round(stride)) for label, frame in best_per_class(entry).items()}
        for gt_event in gt["events"]:
            label = gt_event["label"]
            if label in best:
                errors[label].append(abs(best[label] - int(gt_event["frame"])))
            else:
                missing[label] += 1

    fps = float(np.median(rates)) if rates else float("nan")
    header = "  ".join(f"<={d:<3}" for d in TOLERANCES)
    print(f"\n{len(truth)} held-out trials at {fps:g} fps — 1 frame = {1000 / fps:.1f} ms")
    print(f"{'class':<10} {'n':>4} {'miss':>5} {'mean':>6} {'med':>5}   {header}")
    for name, deltas in errors.items():
        if not deltas:
            print(f"{name:<10} {0:>4} {missing[name]:>5}      -     -   (no predictions)")
            continue
        arr = np.asarray(deltas, dtype=float)
        hit = "  ".join(f"{float((arr <= d).mean()):>5.0%}" for d in TOLERANCES)
        print(f"{name:<10} {len(arr):>4} {missing[name]:>5} {arr.mean():>6.1f} {np.median(arr):>5.1f}   {hit}")
    print("\nmean/med are absolute frame errors; <=k is the fraction landing within k frames.")
    print(f"The 3-4 frame target is {3000 / fps:.0f}-{4000 / fps:.0f} ms here.")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _sessions(args: argparse.Namespace) -> list[SessionPaths]:
    found = discover_sessions(Path(args.derivatives), Path(args.viddata))
    if args.sessions:
        wanted = set(args.sessions)
        found = [s for s in found if s.name in wanted]
        if missing := wanted - {s.name for s in found}:
            raise SystemExit(f"No such session(s) with target labels: {sorted(missing)}")
    if not found:
        raise SystemExit(f"No sessions with labels {sorted(TARGET_LABELS)} under {args.derivatives}")
    return found


def cmd_plan(args: argparse.Namespace) -> None:
    total = 0
    for session in _sessions(args):
        records = plan_session(session, args.max_trials)
        frames = sum(r.num_frames for r in records)
        minutes = frames / records[0].fps / 60 if records else 0.0
        total += frames
        print(f"{session.name:<14} {len(records):>4} trials {frames:>8} frames {minutes:>6.1f} min of video")
    print(f"\ntotal {total} frames — roughly {total * _BYTES_PER_JPEG / 1e9:.1f} GB of JPEG at {FRAME_HEIGHT}px high")


def cmd_export(args: argparse.Namespace) -> None:
    frame_root = Path(args.frame_dir)
    workers = args.workers or max(1, min(12, (os.cpu_count() or 4) - 2))
    records = {session.name: plan_session(session, args.max_trials) for session in _sessions(args)}
    # One pool for every session: each worker imports ethograph once per run
    # rather than once per session, which otherwise dominates a short export.
    flat = [record for session_records in records.values() for record in session_records]
    logger.info("Exporting %d trials from %d sessions on %d workers", len(flat), len(records), workers)
    export_all(flat, frame_root, workers)
    test_session = args.test_session or sorted(records)[-1]
    write_dataset(assign_splits(records, test_session), SPOT_ROOT / "data" / DATASET_NAME)
    print(f"\nHeld out: {test_session}")
    print(f"Frames:   {frame_root.resolve()}")
    print(f"Dataset:  {(SPOT_ROOT / 'data' / DATASET_NAME).resolve()}")


def cmd_index(args: argparse.Namespace) -> None:
    """Rewrite the split JSONs from whatever frames are on disk, decoding nothing.

    ``export`` writes the JSONs only once every trial is done, so an
    interrupted run leaves frames the JSONs do not know about. This is how
    they catch up.
    """
    frame_root = Path(args.frame_dir)
    records = {
        session.name: exported_records(plan_session(session, args.max_trials), frame_root)
        for session in _sessions(args)
    }
    records = {name: found for name, found in records.items() if found}
    if not records:
        raise SystemExit(f"No exported trials under {frame_root.resolve()}")
    test_session = args.test_session or sorted(records)[-1]
    write_dataset(assign_splits(records, test_session), SPOT_ROOT / "data" / DATASET_NAME)
    print(f"\nHeld out: {test_session}")
    print(f"Dataset:  {(SPOT_ROOT / 'data' / DATASET_NAME).resolve()}")


def _has_checkpoint(save_dir: Path) -> bool:
    """Whether ``train_e2e.py --resume`` has an epoch to pick up from."""
    return save_dir.is_dir() and any(save_dir.glob("optim_*.pt"))


def _train_command(args: argparse.Namespace, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "train_e2e.py",
        DATASET_NAME,
        str(Path(args.frame_dir).resolve()),
        "-s",
        args.save_dir,
        "-m",
        args.arch,
        "--num_epochs",
        str(args.epochs),
        "--clip_len",
        str(args.clip_len),
        "--stride",
        str(args.stride),
        "--batch_size",
        str(args.batch_size),
        "-ag",
        str(args.acc_grad),
    ]
    if args.workers is not None:
        command += ["-j", str(args.workers)]
    if resume:
        command.append("--resume")
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return command + extra


def _run_logged(command: list[str], log_path: Path) -> int:
    """Run *command* in ``spot/``, mirroring its output to the console and *log_path*.

    Chunked rather than line-based so tqdm's ``\\r`` updates reach the log as
    they happen; ``PYTHONUNBUFFERED`` does the same on the child's side.
    ``expandable_segments`` lets the CUDA allocator grow in place instead of
    failing on fragmentation — the failure a run near the card's limit
    otherwise hits hours in, during validation.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log, subprocess.Popen(
        command, cwd=SPOT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as proc:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
    return proc.returncode


def cmd_train(args: argparse.Namespace) -> None:
    """Train, and on a crash resume from the last epoch's checkpoint.

    ``train_e2e.py`` writes ``checkpoint_NNN.pt`` + ``optim_NNN.pt`` after
    every epoch and ``--resume`` continues from the newest, so a failure
    costs at most one epoch. ``--retries`` is how many such restarts to
    allow before giving up; the output of every attempt lands in
    ``spot/{save_dir}/train.log``.
    """
    save_dir = SPOT_ROOT / args.save_dir
    log_path = save_dir / "train.log"
    resume = args.resume
    for attempt in range(args.retries + 1):
        command = _train_command(args, resume=resume and _has_checkpoint(save_dir))
        logger.info("Attempt %d/%d: %s (cwd=%s)", attempt + 1, args.retries + 1, " ".join(command), SPOT_ROOT)
        if args.dry_run:
            return
        code = _run_logged(command, log_path)
        if code == 0:
            return
        logger.error("train_e2e.py exited with %d (see %s)", code, log_path)
        if attempt == args.retries:
            raise SystemExit(code)
        resume = True
        logger.info("Restarting in %d s, resuming from the last checkpoint", RETRY_PAUSE_S)
        time.sleep(RETRY_PAUSE_S)


def cmd_score(args: argparse.Namespace) -> None:
    truth = Path(args.truth) if args.truth else SPOT_ROOT / "data" / DATASET_NAME / "test.json"
    score(Path(args.predictions), truth)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--derivatives", default=r"C:\Users\aksel\Documents\AI_data\derivatives\sub-02_id-Poppy")
    common.add_argument("--viddata", default=r"C:\Users\aksel\Documents\VidData")
    common.add_argument("--frame-dir", default="spot_frames")
    common.add_argument("--sessions", nargs="*", help="Session names to use, e.g. 20260304_01")
    common.add_argument("--max-trials", type=int, default=None, help="Cap trials per session (smoke runs)")

    parser = argparse.ArgumentParser(
        description=__doc__, parents=[common], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", parents=[common], help="What export would write, without decoding").set_defaults(
        func=cmd_plan
    )

    export = sub.add_parser("export", parents=[common], help="Extract frames and write the spot dataset")
    export.add_argument("--test-session", default=None, help="Session to hold out (default: the last)")
    export.add_argument(
        "--workers", type=int, default=None, help="Parallel decode workers (default: cores - 2, max 12)"
    )
    export.set_defaults(func=cmd_export)

    index = sub.add_parser("index", parents=[common], help="Rewrite the split JSONs from the frames on disk")
    index.add_argument("--test-session", default=None, help="Session to hold out (default: the last)")
    index.set_defaults(func=cmd_index)

    train = sub.add_parser("train", parents=[common], help="Run spot/train_e2e.py on the exported dataset")
    train.add_argument("--save-dir", default="runs/crow_pellet")
    train.add_argument("--arch", default="rny008_gsm")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--clip-len", type=int, default=200, dest="clip_len")
    train.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Read every k-th frame: k times the context per clip for the same compute, "
        "at the price of k-frame resolution. A speed knob for high frame rates; the sweep reports "
        "errors in full-rate frames either way",
    )
    train.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    train.add_argument(
        "--acc-grad",
        type=int,
        default=4,
        dest="acc_grad",
        help="Gradient-accumulation steps: loader batch = batch-size / this; same effective batch, less GPU memory. "
        "Keep batch-size / this * clip-len at or under 200 frames on a 10 GB card (see the .md)",
    )
    train.add_argument("--workers", type=int, default=None, help="DataLoader workers (upstream default: 8 / 4 val)")
    train.add_argument("--retries", type=int, default=0, help="Restarts allowed, each resuming the last checkpoint")
    train.add_argument("--resume", action="store_true", help="Continue from the checkpoints already in --save-dir")
    train.add_argument("--dry-run", action="store_true", dest="dry_run", help="Print the command and exit")
    train.add_argument("extra", nargs=argparse.REMAINDER, help="Anything after `--` goes to train_e2e.py verbatim")
    train.set_defaults(func=cmd_train)

    scoring = sub.add_parser("score", parents=[common], help="Frame/ms error of a predictions JSON")
    scoring.add_argument("predictions")
    scoring.add_argument("--truth", default=None)
    scoring.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
