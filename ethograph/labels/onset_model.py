"""Supervised point-event onset detection ("GradBoost").

A deliberately constrained CPU model: the user labels a point event in a few
trials, the features they pick are windowed around every frame, and a
scikit-learn ``HistGradientBoostingClassifier`` learns frame-vs-no-event.
The constraints that keep the task tractable for a small model:

* only **point** events — state events have two boundaries and are out of scope;
* at most **one** event per trial, so inference is an argmax over the trial's
  smoothed probability curve, not an open-ended peak-picking problem;
* every chosen feature column must share **one sampling rate** — windows are
  index-based, so mixed rates would silently misalign and are refused.

Target encoding: the classifier sees a binary target (frames within
``tolerance_s`` of the labelled event are positive) with a Gaussian sample
weight peaking at the event, so a near-miss frame counts less than the exact
frame. At inference the per-frame probability is Gaussian-smoothed with the
same tolerance and the trial's argmax becomes the predicted onset.

Model layout (``~/.ethograph/models/{name}/``)::

    config.yaml                 # frozen at creation: target, features, params
    model.joblib                # trained classifier bundle
    train_data/{session}/       # one folder per contributing session
        meta.yaml               # source path, columns, fs, trials
        trial_{id}.npz          # time (T,), data (T, D), y_time, fs

Feature selection reuses the catalog's dim logic: the config stores, per
feature, the explicit values chosen for each of its dims; every combination is
pinned and selected through ``DataLoader.select`` (the same ``sel_valid``
path the plots use), yielding one column per combination. Freezing explicit
values (instead of "all") keeps the column set — and thus the model's input
layout — identical across sessions.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from ethograph.utils.paths import ethograph_home

#: Cap on window taps per input column: beyond this, lags are spread evenly
#: across the window so high-rate features don't explode the design matrix.
MAX_LAGS = 25

#: Relative tolerance for "same sampling rate" comparisons.
_FS_RTOL = 1e-3

#: Deterministic seed for negative-frame subsampling.
_RNG_SEED = 0

_MODEL_FILE = "model.joblib"
_CONFIG_FILE = "config.yaml"
_TRAIN_DIR = "train_data"


# ---------------------------------------------------------------------------
# Config + storage layout
# ---------------------------------------------------------------------------


@dataclass
class OnsetModelConfig:
    """Frozen description of one onset model (written once, at creation)."""

    name: str
    target_label: int
    target_name: str
    #: feature -> dim -> explicit values to include (every combination becomes
    #: one input column). A feature with no dims maps to ``{}``.
    features: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    window_s: float = 0.5
    tolerance_s: float = 0.05
    max_iter: int = 200
    learning_rate: float = 0.1
    #: Negative frames kept per positive frame (hard negatives near the event
    #: are always kept; the rest are subsampled deterministically).
    neg_per_pos: int = 20


def models_root() -> Path:
    """The global model store, ``~/.ethograph/models``."""
    return ethograph_home() / "models"


def model_dir(name: str) -> Path:
    return models_root() / name


def list_models() -> list[str]:
    """Names of all models that have a config on disk."""
    root = models_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / _CONFIG_FILE).is_file())


def save_config(config: OnsetModelConfig) -> Path:
    d = model_dir(config.name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / _CONFIG_FILE
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    return path


def load_config(name: str) -> OnsetModelConfig:
    raw = yaml.safe_load((model_dir(name) / _CONFIG_FILE).read_text(encoding="utf-8"))
    return OnsetModelConfig(**raw)


def session_id(source_path: str | Path) -> str:
    """Stable, filesystem-safe identifier for one loaded session.

    ``{stem}-{hash}``: the stem keeps the folder recognisable, the hash of the
    resolved path keeps two same-named sessions apart.
    """
    p = Path(source_path)
    digest = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem) or "session"
    return f"{stem}-{digest}"


def session_dir(name: str, session: str) -> Path:
    return model_dir(name) / _TRAIN_DIR / session


def list_sessions(name: str) -> dict[str, dict]:
    """Contributed training sessions: ``{session_id: meta dict}``."""
    train_root = model_dir(name) / _TRAIN_DIR
    if not train_root.is_dir():
        return {}
    out: dict[str, dict] = {}
    for p in sorted(train_root.iterdir()):
        meta_path = p / "meta.yaml"
        if meta_path.is_file():
            out[p.name] = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    return out


# ---------------------------------------------------------------------------
# Feature columns — the catalog-selection view of the config
# ---------------------------------------------------------------------------


class FeatureColumn(NamedTuple):
    """One input column: a feature with every dim pinned to one value."""

    feature: str
    selections: dict[str, str]
    name: str


def enumerate_columns(features: dict[str, dict[str, list[str]]]) -> list[FeatureColumn]:
    """Expand the config's per-dim value lists into pinned columns.

    Order is deterministic (config order, then the cartesian product in the
    values' stored order) — it defines the model's input layout.
    """
    columns: list[FeatureColumn] = []
    for feature, dims in features.items():
        if not dims:
            columns.append(FeatureColumn(feature, {}, feature))
            continue
        dim_names = list(dims)
        for combo in itertools.product(*(dims[d] for d in dim_names)):
            selections = dict(zip(dim_names, (str(v) for v in combo)))
            label = ",".join(f"{d}={v}" for d, v in selections.items())
            columns.append(FeatureColumn(feature, selections, f"{feature}|{label}"))
    return columns


def sampling_rate(time: np.ndarray) -> float:
    """Sampling rate implied by a time vector (median spacing)."""
    time = np.asarray(time, dtype=np.float64)
    if time.size < 2:
        raise ValueError("Need at least 2 samples to determine a sampling rate.")
    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError("Time vector is not increasing.")
    return 1.0 / dt


def _check_same_fs(fs_ref: float, fs: float, what: str) -> None:
    if not np.isclose(fs_ref, fs, rtol=_FS_RTOL):
        raise ValueError(
            f"Sampling-rate mismatch: {what} runs at {fs:.6g} Hz but the model's "
            f"features run at {fs_ref:.6g} Hz. All selected features must share "
            "one sampling rate."
        )


def extract_features(
    loader: Any,
    features: dict[str, dict[str, list[str]]],
    t0: float | None = None,
    t1: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select every configured column over ``[t0, t1]`` and stack to ``(T, D)``.

    *loader* is any :class:`~ethograph.io.catalog.DataLoader`; times are in the
    loader's native clock. Raises ``ValueError`` when a feature is missing,
    a selection does not pin down to one column, or sampling rates differ.
    """
    columns = enumerate_columns(features)
    if not columns:
        raise ValueError("The model config selects no features.")

    time_ref: np.ndarray | None = None
    fs_ref = 0.0
    arrays: list[np.ndarray] = []
    for col in columns:
        plot_data = loader.select(col.feature, col.selections, t0, t1)
        if plot_data is None:
            raise ValueError(f"Feature {col.feature!r} is not available in this session.")
        data = np.asarray(plot_data.data, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(
                f"Column {col.name!r} did not pin down to a single series "
                f"(got shape {data.shape}) — the dataset has a dim the model "
                "config does not cover. Recreate the model on this dataset."
            )
        time = np.asarray(plot_data.time, dtype=np.float64)
        fs = sampling_rate(time)
        if time_ref is None:
            time_ref, fs_ref = time, fs
        else:
            _check_same_fs(fs_ref, fs, f"feature {col.feature!r}")
            if abs(float(time[0]) - float(time_ref[0])) > 0.5 / fs_ref:
                raise ValueError(
                    f"Feature {col.feature!r} starts {abs(time[0] - time_ref[0]):.4g} s "
                    "away from the other features — their samples cannot be aligned."
                )
        arrays.append(data)

    assert time_ref is not None
    n = min(len(time_ref), *(len(a) for a in arrays))
    stacked = np.column_stack([a[:n] for a in arrays])
    return time_ref[:n], stacked


# ---------------------------------------------------------------------------
# Training-data storage
# ---------------------------------------------------------------------------


def write_trial_training_data(
    name: str,
    session: str,
    trial_id: int | str,
    time: np.ndarray,
    data: np.ndarray,
    y_time: float,
) -> Path:
    """Persist one trial's assembled features + event time under the model."""
    d = session_dir(name, session)
    d.mkdir(parents=True, exist_ok=True)
    safe_trial = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(trial_id))
    path = d / f"trial_{safe_trial}.npz"
    np.savez_compressed(
        path,
        time=np.asarray(time, dtype=np.float64),
        data=np.asarray(data, dtype=np.float64),
        y_time=np.float64(y_time),
        fs=np.float64(sampling_rate(time)),
    )
    return path


def write_session_meta(name: str, session: str, meta: dict) -> Path:
    d = session_dir(name, session)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "meta.yaml"
    path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Windowing + targets
# ---------------------------------------------------------------------------


def lag_offsets(fs: float, window_s: float, max_lags: int = MAX_LAGS) -> np.ndarray:
    """Sample offsets (in frames) of the centred window seen at each frame.

    At most *max_lags* taps, spread evenly across ``window_s`` — identical for
    training and inference because both derive it from (fs, window_s).
    """
    half = max(1, int(round(window_s * fs / 2)))
    n = min(2 * half + 1, max_lags)
    return np.unique(np.round(np.linspace(-half, half, n)).astype(int))


def build_windows(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-frame design matrix ``(T, D * len(offsets))``.

    Frames whose window reaches past the trial edge get NaN taps —
    ``HistGradientBoostingClassifier`` handles NaN natively.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    n_time, n_dims = data.shape
    x = np.full((n_time, n_dims * len(offsets)), np.nan)
    for j, off in enumerate(offsets):
        src0, src1 = max(0, off), min(n_time, n_time + off)
        dst0, dst1 = max(0, -off), min(n_time, n_time - off)
        x[dst0:dst1, j * n_dims : (j + 1) * n_dims] = data[src0:src1]
    return x


def make_targets(
    time: np.ndarray,
    y_time: float,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary per-frame target + Gaussian sample weight around the event.

    Frames within ``tolerance_s`` of the event are positive; their weight is a
    Gaussian bump (sigma = tolerance/2) peaking at the event, so the exact
    frame counts most. Negative frames get weight 1.
    """
    time = np.asarray(time, dtype=np.float64)
    delta = time - float(y_time)
    y = (np.abs(delta) <= tolerance_s).astype(np.int8)
    if not y.any():
        raise ValueError(
            f"The event at {y_time:.3f} s falls outside the feature time base "
            f"({time[0]:.3f}–{time[-1]:.3f} s)."
        )
    sigma = tolerance_s / 2.0
    weights = np.ones_like(time)
    pos = y.astype(bool)
    weights[pos] = np.exp(-0.5 * (delta[pos] / sigma) ** 2)
    return y, weights


def _subsample_mask(
    y: np.ndarray,
    delta: np.ndarray,
    tolerance_s: float,
    neg_per_pos: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep all positives, all hard negatives near the event, and a random
    subset of the remaining negatives (``neg_per_pos`` per positive)."""
    keep = y.astype(bool) | (np.abs(delta) <= 5.0 * tolerance_s)
    n_pos = int(y.sum())
    far_idx = np.flatnonzero(~keep)
    n_far = max(0, neg_per_pos * n_pos - int((keep & ~y.astype(bool)).sum()))
    if len(far_idx) > n_far:
        keep[rng.choice(far_idx, size=n_far, replace=False)] = True
    else:
        keep[far_idx] = True
    return keep


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------


def _iter_trial_files(name: str) -> list[Path]:
    train_root = model_dir(name) / _TRAIN_DIR
    if not train_root.is_dir():
        return []
    return sorted(train_root.glob("*/trial_*.npz"))


def train_model(name: str) -> dict:
    """Fit the classifier from every stored training trial.

    Returns a summary dict (``n_sessions``, ``n_trials``, ``n_frames``,
    ``n_positive``). Raises ``ValueError`` when there is no training data or
    the stored trials disagree on sampling rate.
    """
    config = load_config(name)
    files = _iter_trial_files(name)
    if not files:
        raise ValueError("No training data yet — add at least one session first.")

    rng = np.random.default_rng(_RNG_SEED)
    fs_ref: float | None = None
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    for path in files:
        with np.load(path) as npz:
            time, data = npz["time"], npz["data"]
            y_time, fs = float(npz["y_time"]), float(npz["fs"])
        if fs_ref is None:
            fs_ref = fs
        else:
            _check_same_fs(fs_ref, fs, f"training trial {path.name!r}")
        offsets = lag_offsets(fs_ref, config.window_s)
        x = build_windows(data, offsets)
        y, w = make_targets(time, y_time, config.tolerance_s)
        keep = _subsample_mask(y, time - y_time, config.tolerance_s, config.neg_per_pos, rng)
        xs.append(x[keep])
        ys.append(y[keep])
        ws.append(w[keep])

    assert fs_ref is not None
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    w_train = np.concatenate(ws)
    # Balance the classes: scale positive weights so both classes carry the
    # same total mass (HistGradientBoostingClassifier has no class_weight).
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0:
        raise ValueError("Training data contains no positive frames.")
    w_train = w_train.copy()
    w_train[y_train == 1] *= n_neg / max(1, n_pos)

    clf = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        random_state=_RNG_SEED,
    )
    clf.fit(x_train, y_train, sample_weight=w_train)

    bundle = {
        "model": clf,
        "fs": fs_ref,
        "columns": [c.name for c in enumerate_columns(config.features)],
        "config": asdict(config),
    }
    joblib.dump(bundle, model_dir(name) / _MODEL_FILE)
    return {
        "n_sessions": len(list_sessions(name)),
        "n_trials": len(files),
        "n_frames": len(y_train),
        "n_positive": n_pos,
    }


def is_trained(name: str) -> bool:
    return (model_dir(name) / _MODEL_FILE).is_file()


def load_bundle(name: str) -> dict:
    """The trained bundle: ``{"model", "fs", "columns", "config"}``."""
    path = model_dir(name) / _MODEL_FILE
    if not path.is_file():
        raise ValueError(f"Model {name!r} has not been trained yet.")
    return joblib.load(path)


def predict_onset(bundle: dict, time: np.ndarray, data: np.ndarray) -> tuple[float, float]:
    """Predict the single event time in one trial's assembled features.

    Returns ``(onset_time, confidence)`` where *onset_time* is on *time*'s
    clock and *confidence* is the smoothed event probability at the peak.
    """
    config = OnsetModelConfig(**bundle["config"])
    fs = sampling_rate(time)
    _check_same_fs(float(bundle["fs"]), fs, "this session's features")

    offsets = lag_offsets(float(bundle["fs"]), config.window_s)
    x = build_windows(data, offsets)
    proba = bundle["model"].predict_proba(x)[:, 1]

    # Smooth with the training tolerance so a plateau of near-hits beats a
    # single spurious spike; the argmax of the smoothed curve is the onset.
    sigma = max(1.0, config.tolerance_s * fs / 2.0)
    half = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(proba, kernel, mode="same")

    idx = int(np.argmax(smoothed))
    return float(time[idx]), float(smoothed[idx])
