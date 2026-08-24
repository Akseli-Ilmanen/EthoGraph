r"""Which S3D dimensions separate the behaviours — supervised selection.

S3D gives 1024 dimensions per frame; most of them say nothing about the
behaviours in a given dataset. Effect size answers "does this dimension move
when this behaviour happens?" one dimension at a time, with no model to fit:
for feature *f* and class *c*, Cohen's d between the frames labelled *c* and
every other frame,

.. math::

    d_{f,c} = \frac{|\bar{x}_{f,c} - \bar{x}_{f,\lnot c}|}
                   {\sqrt{(s^2_{f,c} + s^2_{f,\lnot c}) / 2}}

A feature's score is its best class (:func:`cohens_d` returns both), and a
dataset's score is that averaged over trials (:func:`rank_features`) — so a
dimension has to work in most trials, not one.

Two conventions make the result mean what it looks like:

* **Background is not a class.** A frame that is *not* labelled is the
  contrast, never a thing to detect; ``background=0`` leaves it out (pass
  ``None`` to score it like any other class).
* **A class too rare to measure is skipped by count, not by name.**
  ``min_frames`` drops a class from the trial where it barely occurs; the
  other trials still score it.

A pooled SD of zero — a constant or all-zero column — scores 0 rather than
dividing: there is no spread to express the difference in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Rows per block of the two streaming passes. Bounds the temporaries so a
#: 50k-frame trial costs tens of MB, not gigabytes.
_BLOCK = 4096

#: A pooled variance at or below this fraction of the column's own variance
#: is floating-point noise around zero, and is read as zero.
_REL_EPS = 1e-12


@dataclass
class FeatureRanking:
    """Cohen's d per feature, averaged over trials.

    ``scores`` ranks the features; ``per_class`` says which behaviour each
    one answers to, with ``class_ids`` naming its columns.
    """

    scores: np.ndarray
    per_class: np.ndarray
    class_ids: np.ndarray
    n_trials: int

    @property
    def n_features(self) -> int:
        return int(len(self.scores))

    def top(self, k: int) -> np.ndarray:
        """Indices of the *k* best-scoring features, best first."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        order = np.argsort(-np.asarray(self.scores), kind="stable")
        return order[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": np.asarray(self.scores).tolist(),
            "per_class": np.asarray(self.per_class).tolist(),
            "class_ids": np.asarray(self.class_ids).astype(int).tolist(),
            "n_trials": int(self.n_trials),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureRanking:
        scores = np.asarray(data["scores"], dtype=float)
        per_class = np.asarray(data["per_class"], dtype=float).reshape(len(scores), -1)
        return cls(
            scores=scores,
            per_class=per_class,
            class_ids=np.asarray(data["class_ids"], dtype=np.int64),
            n_trials=int(data["n_trials"]),
        )

    def save(self, path: str | Path) -> Path:
        """Write to *path* as ``.npz``; returns the file actually written."""
        out = Path(path)
        if out.suffix != ".npz":
            out = out.with_suffix(out.suffix + ".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            scores=np.asarray(self.scores, dtype=float),
            per_class=np.asarray(self.per_class, dtype=float),
            class_ids=np.asarray(self.class_ids, dtype=np.int64),
            n_trials=np.int64(self.n_trials),
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> FeatureRanking:
        with np.load(Path(path)) as data:
            return cls(
                scores=np.asarray(data["scores"], dtype=float),
                per_class=np.asarray(data["per_class"], dtype=float),
                class_ids=np.asarray(data["class_ids"], dtype=np.int64),
                n_trials=int(data["n_trials"]),
            )


def _scoreable_classes(
    labels: np.ndarray, n_frames: int, background: int | None, min_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """The classes worth scoring in this trial, and their frame counts.

    A class needs ``min_frames`` frames *and* at least one frame outside it:
    a trial that is entirely one behaviour offers no contrast to measure.
    """
    present = np.unique(labels)
    if background is not None:
        present = present[present != background]
    if len(present) == 0:
        return present.astype(np.int64), np.zeros(0, dtype=np.int64)
    counts = np.array([int(np.count_nonzero(labels == c)) for c in present], dtype=np.int64)
    keep = (counts >= max(min_frames, 1)) & (counts < n_frames)
    return present[keep].astype(np.int64), counts[keep]


def cohens_d(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    background: int | None = 0,
    min_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Effect size of every feature against every class, for one trial.

    Parameters
    ----------
    values
        ``(T, F)`` features, one row per frame.
    labels
        ``(T,)`` dense integer class id per frame.
    background
        Class id meaning "nothing happening", left out of the scoring;
        ``None`` scores every class present.
    min_frames
        Skip a class with fewer than this many frames in this trial.

    Returns
    -------
    max_d, per_class, class_ids
        ``(F,)`` best d per feature, ``(F, C)`` d per feature and class, and
        the ``(C,)`` class ids naming those columns, sorted ascending.
    """
    values = np.asarray(values)
    labels = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (T, F), got shape {values.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D (T,), got shape {labels.shape}")
    if values.shape[0] != labels.shape[0]:
        raise ValueError(f"values has {values.shape[0]} frames but labels has {labels.shape[0]}")
    if min_frames < 0:
        raise ValueError(f"min_frames must be non-negative, got {min_frames}")

    n_frames, n_features = values.shape
    class_ids, counts = _scoreable_classes(labels, n_frames, background, min_frames)
    n_classes = len(class_ids)
    if n_frames == 0 or n_classes == 0:
        return np.zeros(n_features), np.zeros((n_features, n_classes)), class_ids

    total = np.zeros(n_features)
    for start in range(0, n_frames, _BLOCK):
        total += np.asarray(values[start : start + _BLOCK], dtype=np.float64).sum(axis=0)
    centre = total / n_frames

    # Centring on the column mean before squaring keeps the sum-of-squares
    # route to the variance conditioned; it cancels out of the difference.
    class_sum = np.zeros((n_classes, n_features))
    class_sq = np.zeros((n_classes, n_features))
    all_sum = np.zeros(n_features)
    all_sq = np.zeros(n_features)
    for start in range(0, n_frames, _BLOCK):
        block = np.asarray(values[start : start + _BLOCK], dtype=np.float64) - centre
        squared = block * block
        indicator = (labels[start : start + _BLOCK, None] == class_ids[None, :]).astype(np.float64)
        all_sum += block.sum(axis=0)
        all_sq += squared.sum(axis=0)
        class_sum += indicator.T @ block
        class_sq += indicator.T @ squared

    n_in = counts.astype(np.float64)[:, None]
    n_out = float(n_frames) - n_in
    mean_in = class_sum / n_in
    mean_out = (all_sum - class_sum) / n_out
    var_in = np.where(n_in > 1, (class_sq - n_in * mean_in**2) / np.maximum(n_in - 1.0, 1.0), 0.0)
    var_out = np.where(n_out > 1, ((all_sq - class_sq) - n_out * mean_out**2) / np.maximum(n_out - 1.0, 1.0), 0.0)
    pooled_var = (np.maximum(var_in, 0.0) + np.maximum(var_out, 0.0)) / 2.0

    column_var = all_sq / max(n_frames - 1, 1)
    degenerate = pooled_var <= _REL_EPS * column_var[None, :]
    diff = np.abs(mean_in - mean_out)
    d = np.where(degenerate, 0.0, diff / np.sqrt(np.where(degenerate, 1.0, pooled_var)))

    per_class = d.T
    return per_class.max(axis=1), per_class, class_ids


def rank_features(
    trials: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    background: int | None = 0,
    min_frames: int = 0,
) -> FeatureRanking:
    """Rank features by mean-over-trials Cohen's d.

    *trials* yields ``(values (T, F), labels (T,))`` pairs — one per trial,
    all with the same feature count. A trial with no scoreable class still
    counts in the average (contributing zero), so a feature that only works
    where a behaviour happens to be common is scored down, not up.
    """
    scored: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    n_features: int | None = None
    for index, (values, labels) in enumerate(trials):
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"Trial {index}: values must be 2-D (T, F), got shape {values.shape}")
        if n_features is None:
            n_features = int(values.shape[1])
        elif values.shape[1] != n_features:
            raise ValueError(f"Trial {index} has {values.shape[1]} features, but trial 0 has {n_features}")
        scored.append(cohens_d(values, labels, background=background, min_frames=min_frames))

    if n_features is None:
        raise ValueError("rank_features needs at least one trial, got none.")
    if all(len(ids) == 0 for _, _, ids in scored):
        raise ValueError(
            f"No trial has two classes to contrast (background={background}, min_frames={min_frames}); "
            "there is nothing to rank features against."
        )

    class_ids = np.unique(np.concatenate([ids for _, _, ids in scored if len(ids)])).astype(np.int64)
    scores = np.zeros(n_features)
    per_class = np.zeros((n_features, len(class_ids)))
    for max_d, trial_per_class, ids in scored:
        scores += max_d
        if len(ids):
            per_class[:, np.searchsorted(class_ids, ids)] += trial_per_class

    n_trials = len(scored)
    return FeatureRanking(
        scores=scores / n_trials,
        per_class=per_class / n_trials,
        class_ids=class_ids,
        n_trials=n_trials,
    )
