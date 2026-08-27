"""Stage 1 of the distillation recipe: train the pose-only teacher.

Reads the features ``materialise`` wrote (``features/``) and the same
``dataset/{split}.json`` the pixel model reads, so the two modalities share
one split and one set of trials. Trains :class:`~ethograph.spot.pose_model.PoseSpotter`
with E2E-Spot's own per-frame objective — a ``K + 1`` softmax, foreground
classes up-weighted — on clips cut the way the pixel model's are: the same
:class:`~ethograph.spot.config.ClipConfig`, resolved against the *features'*
rate, so context, resolution and the positive window mean the same seconds on
both sides.

Every epoch writes ``pred-val.{epoch}.recall.json.gz`` in E2E-Spot's schema,
so :func:`~ethograph.spot.predict.spot_entry`, the epoch choice and
``evaluate()`` read a teacher run exactly as they read a pixel run. When
training ends, the epoch the sweep ranks first writes the teacher's
**embeddings** — one ``(T', d)`` array per clip under ``features/embeddings/``
— which is what the student distils from.

The teacher exists to be learned from, not deployed: a user whose pose exists
only for a few labelled sessions can train it there and run the student on
video alone. Whether that helps is measured, never assumed —
``evaluate()`` scores the teacher on the same test split as the baseline,
and a student should only be distilled from a teacher that beats it.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ethograph.spot.config import ResolvedClip, SpotConfig, features_fingerprint, save_config
from ethograph.spot.features import Stats, load_split, read_names, read_trial_features, strided
from ethograph.spot.pose_model import PoseSpotter
from ethograph.utils.device import resolve_device

logger = logging.getLogger(__name__)

#: Candidates below this score are not written — E2E-Spot's own cut-off for
#: its high-recall file.
RECALL_THRESHOLD = 0.01

#: How far consecutive inference windows overlap, as a fraction of the clip.
INFERENCE_OVERLAP = 0.5

STATS_FILE = "stats.npz"
WEIGHTS_FILE = "teacher.pt"
#: What a teacher run's ``config.json`` names as its architecture.
FEATURE_ARCH = "pose_shift_gru"


@dataclass
class TeacherTrial:
    """One trial on the strided clock, as the model reads it."""

    video_id: str
    #: ``(T', F)`` — the listed columns.
    x: np.ndarray
    #: 0 = background, ``i + 1`` = the i-th class of ``config.labels.classes``.
    target: np.ndarray
    fps: float


def _class_index(config: SpotConfig) -> dict[int, int]:
    return {label: i + 1 for i, label in enumerate(config.labels.classes)}


def load_trials(config: SpotConfig, video_ids: list[str], clip: ResolvedClip) -> list[TeacherTrial]:
    """Trials on the strided clock, targets dilated by the resolved window.

    Dilation is ``dilate_len`` strided frames either side, exactly as the
    pixel dataset does it, so teacher and student see one target.
    """
    index = _class_index(config)
    trials: list[TeacherTrial] = []
    for video_id in video_ids:
        path = config.features_dir / f"{video_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing — was materialise() run with features: set?")
        raw = read_trial_features(path)
        x = strided(raw, clip.stride, clip.fps)
        target = np.zeros(len(x), dtype=np.int64)
        for frame, label in zip(raw["events"], raw["labels"]):
            centre = int(frame) // clip.stride
            lo, hi = max(0, centre - clip.dilate_len), min(len(x), centre + clip.dilate_len + 1)
            target[lo:hi] = index[int(label)]
        trials.append(TeacherTrial(video_id=video_id, x=x, target=target, fps=clip.fps / clip.stride))
    return trials


def sample_clip(trial: TeacherTrial, clip_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A random window of *clip_len* strided frames, zero-padded past the end."""
    n = len(trial.x)
    start = int(rng.integers(0, max(1, n - clip_len + 1)))
    x = trial.x[start : start + clip_len]
    y = trial.target[start : start + clip_len]
    if len(x) < clip_len:
        pad = clip_len - len(x)
        x = np.concatenate([x, np.zeros((pad,) + x.shape[1:], dtype=x.dtype)])
        y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
    return x, y


def build_model(config: SpotConfig, n_features: int, fs_strided: float) -> PoseSpotter:
    teacher = config.teacher
    return PoseSpotter(
        n_features=n_features,
        n_classes=len(config.labels.classes),
        scales=teacher.shift_samples(fs_strided),
        hidden=teacher.hidden,
        depth=teacher.depth,
        shift_fraction=teacher.shift_fraction,
        head_hidden=teacher.head_hidden,
    )


@torch.no_grad()
def _windows(model: PoseSpotter, trial: TeacherTrial, clip_len: int, device: torch.device, fn) -> np.ndarray:
    """Run *fn(model, x)* over overlapping windows and average where they overlap."""
    model.eval()
    n = len(trial.x)
    step = max(1, int(clip_len * (1.0 - INFERENCE_OVERLAP)))
    starts = list(range(0, max(1, n - clip_len + 1), step))
    if starts[-1] + clip_len < n:
        starts.append(max(0, n - clip_len))
    out = None
    counts = np.zeros(n, dtype=np.float32)
    for start in starts:
        x = trial.x[start : start + clip_len]
        pad = clip_len - len(x)
        if pad:
            x = np.concatenate([x, np.zeros((pad,) + x.shape[1:], dtype=x.dtype)])
        values = fn(model, torch.from_numpy(x)[None].to(device))[0]
        values = values.float().cpu().numpy()[: clip_len - pad]
        if out is None:
            out = np.zeros((n,) + values.shape[1:], dtype=np.float32)
        out[start : start + len(values)] += values
        counts[start : start + len(values)] += 1
    assert out is not None
    return out / np.maximum(counts, 1)[:, None]


def predict_scores(model: PoseSpotter, trial: TeacherTrial, clip_len: int, device: torch.device) -> np.ndarray:
    """Per-frame class probabilities ``(T', K + 1)`` over a whole trial."""
    return _windows(model, trial, clip_len, device, lambda m, x: torch.softmax(m(x), dim=-1))


def embed(model: PoseSpotter, trial: TeacherTrial, clip_len: int, device: torch.device) -> np.ndarray:
    """Per-frame embeddings ``(T', d)`` over a whole trial — the distillation target."""
    return _windows(model, trial, clip_len, device, lambda m, x: m.features(x))


def recall_entry(config: SpotConfig, trial: TeacherTrial, probs: np.ndarray) -> dict:
    """One video in E2E-Spot's high-recall schema, on the strided clock."""
    events = []
    for i, label in enumerate(config.labels.classes):
        name = config.class_name(label)
        column = probs[:, i + 1]
        for frame in np.flatnonzero(column >= RECALL_THRESHOLD):
            events.append({"label": name, "frame": int(frame), "score": float(column[frame])})
    return {"video": trial.video_id, "fps": trial.fps, "num_frames": len(probs), "events": events}


def write_recall(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(entries, fh)
    return path


def _class_weights(config: SpotConfig, device: torch.device) -> torch.Tensor:
    weights = [1.0] + [config.teacher.fg_weight] * len(config.labels.classes)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _upstream_style_config(config: SpotConfig, clip: ResolvedClip) -> dict:
    """The ``config.json`` the epoch choice and ``evaluate()`` read off any run."""
    return {
        "dataset": str(config.dataset_dir),
        "stride": clip.stride,
        "clip_len": clip.clip_len,
        "dilate_len": clip.dilate_len,
        "feature_arch": FEATURE_ARCH,
        "temporal_arch": "gru",
    }


def teacher_run_dir(config: SpotConfig, clip: ResolvedClip) -> Path:
    """``teacher/{clip}_{fingerprint}``: a changed feature list or teacher setting is a new folder."""
    name = config.train.run_name or f"ctx{clip.context_s:g}s_res{clip.resolution_ms:g}ms"
    return config.teacher_dir / f"{name}_{features_fingerprint(config)}"


def _fit_stats(trials: list[TeacherTrial]) -> Stats:
    return Stats([t.x for t in trials])


def _apply_stats(stats: Stats, trials: list[TeacherTrial]) -> None:
    for t in trials:
        t.x = stats.apply(t.x)


def train_teacher(config: SpotConfig, clip: ResolvedClip, run_dir: Path | None = None) -> Path:
    """Train the teacher; every epoch's val predictions land beside the weights.

    Returns the run directory. Trials come from ``dataset/train.json`` and
    ``dataset/val.json``, columns from ``features/``. Ends by writing the
    embeddings of the epoch the sweep ranks first (:func:`write_embeddings`).
    """
    if not config.features:
        raise ValueError("features: is empty — the teacher has nothing to read")
    teacher = config.teacher
    run_dir = run_dir or teacher_run_dir(config, clip)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    (run_dir / "config.json").write_text(json.dumps(_upstream_style_config(config, clip), indent=2), encoding="utf-8")

    names = read_names(config.features_dir)
    train = load_trials(config, load_split(config, "train"), clip)
    val = load_trials(config, load_split(config, "val"), clip)
    if not train:
        raise ValueError("train.json lists no trials")
    stats = _fit_stats(train)
    stats.save(run_dir / STATS_FILE)
    _apply_stats(stats, train + val)

    device = torch.device(config.train.device or resolve_device())
    torch.manual_seed(teacher.seed)
    rng = np.random.default_rng(teacher.seed)
    fs_strided = train[0].fps
    model = build_model(config, len(names), fs_strided).to(device)
    logger.info(
        "Teacher: %d features %s, shifts %s at %g Hz, %d params",
        len(names),
        names,
        teacher.shift_samples(fs_strided),
        fs_strided,
        sum(p.numel() for p in model.parameters()),
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=teacher.learning_rate, weight_decay=teacher.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(config, device))
    clips_per_epoch = max(teacher.batch_size, 2 * sum(len(t.x) for t in train) // clip.clip_len)
    history = []
    for epoch in range(teacher.epochs):
        model.train()
        total = 0.0
        n_batches = 0
        for _ in range(clips_per_epoch // teacher.batch_size):
            batch = [
                sample_clip(train[int(rng.integers(len(train)))], clip.clip_len, rng) for _ in range(teacher.batch_size)
            ]
            x = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
            y = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.detach().item()
            n_batches += 1
        entries = [recall_entry(config, t, predict_scores(model, t, clip.clip_len, device)) for t in val]
        write_recall(run_dir / f"pred-val.{epoch}.recall.json.gz", entries)
        torch.save(model.state_dict(), run_dir / f"checkpoint_{epoch:03d}.pt")
        history.append({"epoch": epoch, "train_loss": total / max(1, n_batches)})
        logger.info("epoch %d: train loss %.4f", epoch, history[-1]["train_loss"])
        (run_dir / "loss.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / WEIGHTS_FILE)

    from ethograph.spot.inference import best_epoch

    epoch = best_epoch(run_dir, config)
    write_embeddings(config, run_dir, epoch, clip)
    return run_dir


def load_teacher(
    config: SpotConfig, run_dir: Path, epoch: int, clip: ResolvedClip, device: torch.device
) -> tuple[PoseSpotter, Stats]:
    stats = Stats.load(run_dir / STATS_FILE)
    model = build_model(config, len(stats.mean), clip.fps / clip.stride).to(device)
    model.load_state_dict(torch.load(run_dir / f"checkpoint_{epoch:03d}.pt", map_location=device))
    return model, stats


def predict_split(config: SpotConfig, run_dir: Path, epoch: int, split: str) -> Path:
    """``pred-{split}.{epoch}.recall.json.gz`` of a teacher run, in E2E-Spot's schema, for ``evaluate()``."""
    device = torch.device(config.train.device or resolve_device())
    ids = load_split(config, split)
    if not ids:
        raise ValueError(f"{split}.json lists no trials")
    fps = float(read_trial_features(config.features_dir / f"{ids[0]}.npz")["fps"])
    clip = config.clip.resolve(fps)
    model, stats = load_teacher(config, run_dir, epoch, clip, device)
    model.eval()
    trials = load_trials(config, ids, clip)
    _apply_stats(stats, trials)
    entries = [recall_entry(config, t, predict_scores(model, t, clip.clip_len, device)) for t in trials]
    return write_recall(run_dir / f"pred-{split}.{epoch}.recall.json.gz", entries)


def is_teacher_run(run_dir: Path) -> bool:
    """Whether *run_dir* is a pose teacher (its ``config.json`` names a ``pose_*`` architecture)."""
    path = run_dir / "config.json"
    if not path.is_file():
        return False
    arch = str(json.loads(path.read_text(encoding="utf-8")).get("feature_arch", ""))
    return arch.startswith("pose_")


def write_embeddings(config: SpotConfig, run_dir: Path, epoch: int, clip: ResolvedClip) -> Path:
    """The teacher's per-clip embeddings for every train and val clip.

    ``features/embeddings/{video_id}.npz`` — ``embedding (T', d)`` on the
    strided clock, plus ``stride``/``fps`` so the student can refuse a clock
    it does not share. The test split is left out on purpose: a held-out trial
    must not be touched by any stage.
    """
    device = torch.device(config.train.device or resolve_device())
    model, stats = load_teacher(config, run_dir, epoch, clip, device)
    out_dir = config.embeddings_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = load_split(config, "train") + load_split(config, "val")
    trials = load_trials(config, ids, clip)
    _apply_stats(stats, trials)
    for trial in trials:
        emb = embed(model, trial, clip.clip_len, device)
        np.savez_compressed(
            out_dir / f"{trial.video_id}.npz",
            embedding=emb.astype(np.float32),
            stride=np.int64(clip.stride),
            fps=np.float64(clip.fps),
            teacher=str(run_dir),
            epoch=np.int64(epoch),
        )
    (out_dir / "teacher.json").write_text(
        json.dumps({"run": str(run_dir), "epoch": epoch, "dim": model.embed_dim, "n_clips": len(ids)}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "embeddings: %d clips x %d dims from %s epoch %d -> %s", len(ids), model.embed_dim, run_dir.name, epoch, out_dir
    )
    return out_dir
