"""Supervised point-event onset detection ("LightGBM").

A deliberately constrained CPU model: the user labels one or more point-event
classes in a few trials, the features they pick are windowed around every
frame, and one scikit-learn ``HistGradientBoostingClassifier`` per class
learns frame-vs-no-event — scikit-learn's histogram-based gradient boosting,
modelled on Microsoft's LightGBM. The constraints that keep the task tractable
for a small model:

* only **point** events — state events have two boundaries and are out of scope;
* at most **one** event per class per trial, so inference is an argmax over
  that class's smoothed probability curve, not an open-ended peak-picking
  problem;
* every chosen feature column must share **one sampling rate** — windows are
  index-based, so mixed rates would silently misalign and are refused.

A model can predict **several point-event classes at once**: the features,
window and tolerance are shared (extracted once per trial), and each class
gets its own binary classifier trained on the trials that carry it.

Target encoding: each classifier sees a binary target (frames within
``tolerance_s`` of the labelled event are positive) with a Gaussian sample
weight peaking at the event, so a near-miss frame counts less than the exact
frame. At inference the per-frame probability is Gaussian-smoothed with the
same tolerance and the trial's argmax becomes the predicted onset.

Confidence: the smoothed curve answers two questions — *how strongly* does the
model believe (its peak height) and *how localised* is that belief (one minus
the normalised entropy of the curve read as a distribution over time). A model
that is sure of the moment scores high on both; a flat curve with one mild
bump scores low. :func:`predict_events` reports both and their geometric mean
as ``confidence``, which is what lands in the labels TSV's ``confidence``
column (a human label is 1.0 by definition).

Sequence model (optional): when the classes follow a stereotypic order in a
trial (A then B then C), ``use_crf`` adds a linear-chain CRF
(``sklearn-crfsuite``) on top. Every frame is tagged with the class of the most
recent event, so the CRF's transitions *are* the sequence dependencies, and
Viterbi decoding returns one ordered, mutually consistent set of events per
trial instead of independent per-class argmaxes. See :func:`phase_tags`.

Model layout (``~/.ethograph/models/{name}/``)::

    config.yaml                 # frozen at creation: targets, features, params
    model.joblib                # trained bundle: one clf per target (+ CRF)
    train_data/{session}/       # one folder per contributing session
        meta.yaml               # source path, columns, fs, trials
        trial_{id}.npz          # time (T,), data (T, D), y_labels, y_times, fs

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
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import sklearn_crfsuite
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from ethograph.utils.paths import ethograph_home

logger = logging.getLogger(__name__)

#: Cap on window taps per input column: beyond this, lags are spread evenly
#: across the window so high-rate features don't explode the design matrix.
MAX_LAGS = 25

#: Relative tolerance for "same sampling rate" comparisons.
_FS_RTOL = 1e-3

#: Deterministic seed for negative-frame subsampling.
_RNG_SEED = 0

#: Placeholder label for training trials stored before multi-target support
#: (their npz holds one unlabelled ``y_time``).
_LEGACY_TARGET = -1

#: Tag for the frames of a trial before its first event.
CRF_NONE_TAG = "none"

#: A CRF sequence is one token per frame, so a trial's frame count is the cost
#: driver. Refuse absurd lengths with a message instead of hanging in CRFsuite.
CRF_MAX_FRAMES = 200_000

#: CRFsuite L1/L2 regularisation and iteration cap. Not exposed: the emissions
#: are already a handful of probabilities, so there is little to overfit and
#: nothing here rewards tuning.
_CRF_C1 = 0.1
_CRF_C2 = 0.1
_CRF_MAX_ITER = 120

#: Folds used to cross-fit the probability curves the CRF trains on. Each fold
#: refits every target, so this multiplies training time — 3 keeps the
#: emissions honest without making Train a coffee break.
_CRF_FOLDS = 3

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
    #: label id -> label name, one entry per point-event class the model
    #: predicts. Each gets its own binary classifier over the shared features.
    targets: dict[int, str] = field(default_factory=dict)
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
    #: Model the order the classes occur in with a linear-chain CRF on top of
    #: the per-class classifiers (see :func:`train_crf`). Only worth it when
    #: the classes really do follow a stereotypic sequence.
    use_crf: bool = False

    def __post_init__(self) -> None:
        # YAML round-trips keys as ints already, but a config built from GUI
        # widgets may carry numpy ints or strings.
        self.targets = {int(label): str(name) for label, name in self.targets.items()}

    @property
    def target_labels(self) -> list[int]:
        return list(self.targets)

    def target_name(self, label: int) -> str:
        return self.targets.get(int(label), str(label))

    def describe_targets(self) -> str:
        return ", ".join(f"{name} ({label})" for label, name in self.targets.items())


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


def _upgrade_config_dict(raw: dict) -> dict:
    """A stored config as the current dataclass expects it.

    Single-target models written before multi-target support carry
    ``target_label``/``target_name`` instead of ``targets``.
    """
    raw = dict(raw)
    if "targets" not in raw and "target_label" in raw:
        raw["targets"] = {int(raw.pop("target_label")): str(raw.pop("target_name", ""))}
    raw.pop("target_label", None)
    raw.pop("target_name", None)
    return raw


def load_config(name: str) -> OnsetModelConfig:
    raw = yaml.safe_load((model_dir(name) / _CONFIG_FILE).read_text(encoding="utf-8"))
    return OnsetModelConfig(**_upgrade_config_dict(raw))


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
    y_times: dict[int, float],
) -> Path:
    """Persist one trial's assembled features + its event times under the model.

    *y_times* maps target label id to that event's time in this trial. A
    target the trial does not carry is simply absent: the trial then
    contributes nothing to that target's classifier, because an unlabelled
    trial is not evidence that the event never happened.
    """
    d = session_dir(name, session)
    d.mkdir(parents=True, exist_ok=True)
    safe_trial = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(trial_id))
    path = d / f"trial_{safe_trial}.npz"
    labels = sorted(y_times)
    np.savez_compressed(
        path,
        time=np.asarray(time, dtype=np.float64),
        data=np.asarray(data, dtype=np.float64),
        y_labels=np.asarray(labels, dtype=np.int64),
        y_times=np.asarray([y_times[label] for label in labels], dtype=np.float64),
        fs=np.float64(sampling_rate(time)),
    )
    return path


def read_trial_training_data(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, float], float]:
    """Read one stored training trial: ``(time, data, {label: y_time}, fs)``.

    A trial written before multi-target support stores a bare ``y_time`` with
    no label; it reads back under :data:`_LEGACY_TARGET`, which
    :func:`_resolve_y_time` maps onto a single-target model's only target.
    """
    with np.load(path) as npz:
        time = np.asarray(npz["time"], dtype=np.float64)
        data = np.asarray(npz["data"], dtype=np.float64)
        fs = float(npz["fs"])
        if "y_labels" in npz:
            y_times = {int(label): float(t) for label, t in zip(npz["y_labels"], npz["y_times"])}
        else:
            y_times = {_LEGACY_TARGET: float(npz["y_time"])}
    return time, data, y_times, fs


def _resolve_y_time(y_times: dict[int, float], label: int, config: OnsetModelConfig) -> float | None:
    """This trial's event time for *label*, or ``None`` if it carries none."""
    if label in y_times:
        return y_times[label]
    if _LEGACY_TARGET in y_times and len(config.targets) == 1:
        return y_times[_LEGACY_TARGET]
    return None


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


def _fit_target(
    label: int,
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    ws: list[np.ndarray],
    config: OnsetModelConfig,
) -> tuple[HistGradientBoostingClassifier, dict]:
    """Fit one target's binary classifier from its per-trial slices."""
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    w_train = np.concatenate(ws)
    # Balance the classes: scale positive weights so both classes carry the
    # same total mass (HistGradientBoostingClassifier has no class_weight).
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0:
        raise ValueError(f"Target {config.target_name(label)!r} has no positive frames in its training data.")
    w_train = w_train.copy()
    w_train[y_train == 1] *= n_neg / max(1, n_pos)

    clf = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        random_state=_RNG_SEED,
    )
    clf.fit(x_train, y_train, sample_weight=w_train)
    return clf, {"n_trials": len(xs), "n_frames": len(y_train), "n_positive": n_pos}


class TrainingTrial(NamedTuple):
    """One stored training trial, as training and cross-fitting see it."""

    time: np.ndarray
    data: np.ndarray
    y_times: dict[int, float]
    fs: float


def load_training_trials(name: str, config: OnsetModelConfig) -> list[TrainingTrial]:
    """Every stored trial of a model, checked to share one sampling rate."""
    trials: list[TrainingTrial] = []
    fs_ref: float | None = None
    for path in _iter_trial_files(name):
        time, data, y_times, fs = read_trial_training_data(path)
        if fs_ref is None:
            fs_ref = fs
        else:
            _check_same_fs(fs_ref, fs, f"training trial {path.name!r}")
        trials.append(TrainingTrial(time, data, y_times, fs))
    return trials


def _fit_targets(
    trials: list[TrainingTrial],
    offsets: np.ndarray,
    config: OnsetModelConfig,
    targets: list[int] | None = None,
) -> tuple[dict[int, HistGradientBoostingClassifier], dict[int, dict]]:
    """Fit one binary classifier per target over *trials*.

    The design matrix is built once per trial and reused for every target —
    only the binary labels and the negative subsampling differ.
    """
    rng = np.random.default_rng(_RNG_SEED)
    buckets: dict[int, tuple[list, list, list]] = {
        label: ([], [], []) for label in (targets if targets is not None else config.targets)
    }
    for trial in trials:
        x = build_windows(trial.data, offsets)
        for label, (xs, ys, ws) in buckets.items():
            y_time = _resolve_y_time(trial.y_times, label, config)
            if y_time is None:
                continue
            y, w = make_targets(trial.time, y_time, config.tolerance_s)
            keep = _subsample_mask(y, trial.time - y_time, config.tolerance_s, config.neg_per_pos, rng)
            xs.append(x[keep])
            ys.append(y[keep])
            ws.append(w[keep])

    models: dict[int, HistGradientBoostingClassifier] = {}
    per_target: dict[int, dict] = {}
    for label, (xs, ys, ws) in buckets.items():
        if not xs:
            raise ValueError(
                f"No training trial carries {config.target_name(label)!r} — add a session "
                "that labels it, or create the model without that target."
            )
        models[label], per_target[label] = _fit_target(label, xs, ys, ws, config)
    return models, per_target


def train_model(name: str) -> dict:
    """Fit one classifier per target — and the sequence CRF when configured.

    Returns a summary dict (``n_sessions``, ``n_trials``, ``targets`` mapping
    each label to its own ``n_trials``/``n_frames``/``n_positive``, and ``crf``
    when a sequence model was fitted). Raises ``ValueError`` when there is no
    training data, a target has no labelled trial, or the stored trials
    disagree on sampling rate.
    """
    config = load_config(name)
    if not config.targets:
        raise ValueError(f"Model {name!r} has no target point events configured.")
    trials = load_training_trials(name, config)
    if not trials:
        raise ValueError("No training data yet — add at least one session first.")

    fs_ref = trials[0].fs
    offsets = lag_offsets(fs_ref, config.window_s)
    models, per_target = _fit_targets(trials, offsets, config)

    bundle = {
        "models": models,
        "fs": fs_ref,
        "columns": [c.name for c in enumerate_columns(config.features)],
        "config": asdict(config),
    }
    summary = {
        "n_sessions": len(list_sessions(name)),
        "n_trials": len(trials),
        "targets": per_target,
    }
    if config.use_crf:
        bundle["crf"], summary["crf"] = train_crf(trials, offsets, config)
    joblib.dump(bundle, model_dir(name) / _MODEL_FILE)
    return summary


def is_trained(name: str) -> bool:
    return (model_dir(name) / _MODEL_FILE).is_file()


def load_bundle(name: str) -> dict:
    """The trained bundle: ``{"models", "fs", "columns", "config"}``.

    A bundle written by a single-target model (one ``"model"``) is presented
    as a one-entry ``"models"`` mapping, so callers only see one shape.
    """
    path = model_dir(name) / _MODEL_FILE
    if not path.is_file():
        raise ValueError(f"Model {name!r} has not been trained yet.")
    bundle = joblib.load(path)
    bundle["config"] = _upgrade_config_dict(bundle["config"])
    if "models" not in bundle:
        bundle["models"] = {OnsetModelConfig(**bundle["config"]).target_labels[0]: bundle.pop("model")}
    bundle["models"] = {int(label): clf for label, clf in bundle["models"].items()}
    return bundle


def bundle_config(bundle: dict) -> OnsetModelConfig:
    """The config a trained bundle was fitted with."""
    return OnsetModelConfig(**_upgrade_config_dict(bundle["config"]))


def _smoothing_kernel(tolerance_s: float, fs: float) -> np.ndarray:
    """Gaussian kernel matching the training tolerance, normalised to sum 1."""
    sigma = max(1.0, tolerance_s * fs / 2.0)
    half = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    return kernel / kernel.sum()


def curve_sharpness(curve: np.ndarray) -> float:
    """How localised a probability curve is over time, in ``[0, 1]``.

    The curve is read as a distribution over the trial's frames and scored by
    one minus its normalised entropy: a single-frame spike scores 1, a flat
    curve scores 0. This is the point-event counterpart of the frame-wise
    ``1 - normalised entropy`` used for dense predictions — there the
    distribution is over classes, here it is over time.
    """
    curve = np.asarray(curve, dtype=np.float64)
    total = float(curve.sum())
    if curve.size < 2 or total <= 0:
        return 0.0
    q = curve / total
    q = q[q > 0]
    entropy = float(-np.sum(q * np.log(q)))
    return float(np.clip(1.0 - entropy / np.log(curve.size), 0.0, 1.0))


@dataclass
class OnsetPrediction:
    """One target's predicted event in one trial."""

    label: int
    name: str
    time: float
    #: Smoothed event probability at the peak — how strongly the model believes.
    peak: float
    #: How localised that belief is over the trial (see :func:`curve_sharpness`).
    sharpness: float

    @property
    def confidence(self) -> float:
        """Geometric mean of *peak* and *sharpness* — both have to hold up.

        This is the number written to the label's ``confidence`` column.
        """
        return float(np.sqrt(np.clip(self.peak, 0.0, 1.0) * self.sharpness))


def target_curves(
    models: dict[int, HistGradientBoostingClassifier],
    x: np.ndarray,
    kernel: np.ndarray,
) -> dict[int, np.ndarray]:
    """Each target's smoothed per-frame event probability over a trial.

    Smoothing with the training tolerance makes a plateau of near-hits beat a
    single spurious spike.
    """
    return {label: np.convolve(clf.predict_proba(x)[:, 1], kernel, mode="same") for label, clf in models.items()}


def predict_events(
    bundle: dict,
    time: np.ndarray,
    data: np.ndarray,
    use_crf: bool = True,
) -> dict[int, OnsetPrediction]:
    """Predict one event per target in one trial's assembled features.

    Times are on *time*'s clock. Without a sequence model each target's
    smoothed probability curve is read independently and its argmax is that
    target's event. With one (and *use_crf*), the curves are decoded jointly
    so the events come out in an order the training data actually showed —
    which can also mean a target gets no prediction in this trial.
    """
    config = bundle_config(bundle)
    fs = sampling_rate(time)
    _check_same_fs(float(bundle["fs"]), fs, "this session's features")

    offsets = lag_offsets(float(bundle["fs"]), config.window_s)
    x = build_windows(data, offsets)
    curves = target_curves(bundle["models"], x, _smoothing_kernel(config.tolerance_s, fs))

    crf = bundle.get("crf")
    if crf is not None and use_crf:
        return decode_crf(crf, time, curves, config)
    return {
        label: OnsetPrediction(
            label=label,
            name=config.target_name(label),
            time=float(time[int(np.argmax(curve))]),
            peak=float(curve[int(np.argmax(curve))]),
            sharpness=curve_sharpness(curve),
        )
        for label, curve in curves.items()
    }


# ---------------------------------------------------------------------------
# Sequence model — a linear-chain CRF over "which event happened last"
# ---------------------------------------------------------------------------


def phase_tags(time: np.ndarray, y_times: dict[int, float]) -> list[str]:
    """Tag every frame with the class of the most recent event.

    Before the first event a frame is :data:`CRF_NONE_TAG`; from the frame of
    event A onwards it is ``"A"``, from event B onwards ``"B"``, and so on.
    Written this way the CRF's *transitions* are exactly the sequence
    dependencies — ``none→A``, ``A→B``, ``B→C`` — and the frames where the tag
    changes are the events. Because a tag persists until the next event, a
    first-order chain is enough to carry the whole order.
    """
    tags = np.full(len(time), CRF_NONE_TAG, dtype=object)
    for label, t_event in sorted(y_times.items(), key=lambda item: item[1]):
        tags[int(np.searchsorted(time, t_event)) :] = str(label)
    return tags.tolist()


def crf_features(time: np.ndarray, curves: dict[int, np.ndarray]) -> list[dict[str, float]]:
    """One CRFsuite feature dict per frame.

    The emissions are the per-target probabilities the classifiers produced,
    not the raw signals: the boosted trees answer "is the event here?" frame
    by frame, and the CRF only has to glue those answers into one ordered
    sequence. Each target contributes its value at the frame plus its
    neighbours (CRFsuite has no window of its own), and every frame carries
    its relative position in the trial.
    """
    n = len(time)
    span = float(time[-1] - time[0]) if n > 1 else 0.0
    position = (np.asarray(time, dtype=np.float64) - float(time[0])) / span if span > 0 else np.zeros(n)
    sequence = [{"bias": 1.0, "t": float(position[i])} for i in range(n)]
    for label, curve in curves.items():
        previous = np.concatenate([curve[:1], curve[:-1]])
        following = np.concatenate([curve[1:], curve[-1:]])
        for i in range(n):
            frame = sequence[i]
            frame[f"p:{label}"] = float(curve[i])
            frame[f"p:{label}@-1"] = float(previous[i])
            frame[f"p:{label}@+1"] = float(following[i])
    return sequence


def observed_sequences(trials: list[TrainingTrial], config: OnsetModelConfig) -> dict[str, int]:
    """How often each event order appears in the training trials.

    ``{"3-4-5": 18, "3-5": 2}`` reads as "18 trials ran 3 then 4 then 5". This
    is what the CRF has to work with — an order that never appears here is an
    order it will never predict.
    """
    counts: dict[str, int] = {}
    for trial in trials:
        present = {
            label: t
            for label in config.targets
            if (t := _resolve_y_time(trial.y_times, label, config)) is not None
        }
        if not present:
            continue
        key = "-".join(str(label) for label, _ in sorted(present.items(), key=lambda item: item[1]))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _cross_fit_curves(
    trials: list[TrainingTrial],
    offsets: np.ndarray,
    config: OnsetModelConfig,
) -> list[dict[int, np.ndarray]]:
    """Out-of-fold probability curves, one dict per trial.

    A classifier scores its own training trials almost perfectly, so a CRF fed
    those curves would learn to trust the emissions far more than it should.
    Each trial is therefore scored by classifiers that never saw it: the
    trials are split into folds and each fold is scored by a model fitted on
    the rest.
    """
    n_splits = min(_CRF_FOLDS, len(trials))
    curves: list[dict[int, np.ndarray]] = [{} for _ in trials]
    for fold in range(n_splits):
        held = [i for i in range(len(trials)) if i % n_splits == fold]
        rest = [trials[i] for i in range(len(trials)) if i % n_splits != fold]
        covered = [
            label
            for label in config.targets
            if any(_resolve_y_time(trial.y_times, label, config) is not None for trial in rest)
        ]
        missing = [label for label in config.targets if label not in covered]
        if missing:
            # Too few labelled trials for this target to be held out fairly.
            # The sequence model still trains; its emissions for these trials
            # are simply the ones the final model produces.
            logger.warning(
                "Cross-fitting fold %d has no training trial for %s — those curves are not out-of-fold.",
                fold,
                ", ".join(config.target_name(label) for label in missing),
            )
        models, _ = _fit_targets(rest, offsets, config, targets=covered) if covered else ({}, {})
        fallback, _ = _fit_targets(trials, offsets, config, targets=missing) if missing else ({}, {})
        for i in held:
            trial = trials[i]
            x = build_windows(trial.data, offsets)
            kernel = _smoothing_kernel(config.tolerance_s, trial.fs)
            curves[i] = target_curves({**models, **fallback}, x, kernel)
    return curves


def train_crf(
    trials: list[TrainingTrial],
    offsets: np.ndarray,
    config: OnsetModelConfig,
) -> tuple[sklearn_crfsuite.CRF, dict]:
    """Fit the linear-chain CRF that models the order the classes occur in.

    ``all_possible_transitions=False`` is the structural choice that makes
    this worth doing: CRFsuite only weights transitions it saw in training, so
    an order the trials never showed is one Viterbi cannot produce.
    """
    if len(trials) < 2:
        raise ValueError("The sequence model needs at least 2 training trials — add more before training.")
    _check_crf_length(max(len(trial.time) for trial in trials))

    curves = _cross_fit_curves(trials, offsets, config)
    sequences = [crf_features(trial.time, curve) for trial, curve in zip(trials, curves)]
    tag_lists = [phase_tags(trial.time, trial.y_times) for trial in trials]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=_CRF_C1,
        c2=_CRF_C2,
        max_iterations=_CRF_MAX_ITER,
        all_possible_transitions=False,
    )
    crf.fit(sequences, tag_lists)
    return crf, {"n_trials": len(trials), "sequences": observed_sequences(trials, config)}


def _check_crf_length(n_frames: int) -> None:
    if n_frames > CRF_MAX_FRAMES:
        raise ValueError(
            f"A trial has {n_frames} frames; the sequence model takes one token per frame and "
            f"is capped at {CRF_MAX_FRAMES}. Pick features from a lower-rate stream, or train "
            "without the sequence model."
        )


def decode_crf(
    crf: sklearn_crfsuite.CRF,
    time: np.ndarray,
    curves: dict[int, np.ndarray],
    config: OnsetModelConfig,
) -> dict[int, OnsetPrediction]:
    """Viterbi-decode one trial into an ordered set of events.

    Each frame where the decoded tag changes is the event of the class it
    changes *to*; a class the path never enters gets no prediction in this
    trial, which is the sequence model saying it did not happen. Confidence
    reuses the same two-part reading as the argmax path, on the CRF's
    marginals: *peak* is how sure it is of the state at that frame, and
    *sharpness* how localised the switch into it was.
    """
    _check_crf_length(len(time))
    sequence = crf_features(time, curves)
    path = list(crf.predict_single(sequence))
    marginals = crf.predict_marginals_single(sequence)

    out: dict[int, OnsetPrediction] = {}
    previous = CRF_NONE_TAG
    for idx, tag in enumerate(path):
        if tag == previous:
            continue
        previous = tag
        if tag == CRF_NONE_TAG or int(tag) not in config.targets:
            continue
        state = np.array([frame.get(tag, 0.0) for frame in marginals], dtype=np.float64)
        switch = np.clip(np.diff(state, prepend=state[0]), 0.0, None)
        label = int(tag)
        out[label] = OnsetPrediction(
            label=label,
            name=config.target_name(label),
            time=float(time[idx]),
            peak=float(state[idx]),
            sharpness=curve_sharpness(switch),
        )
    return out
