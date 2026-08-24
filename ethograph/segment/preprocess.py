"""The fixed preprocessing chain and the run-level normalisation statistics.

Materialised features are stored after the *session-level* steps
(likelihood threshold → interpolate → clip). Z-scoring is a *run-level*
step: its mean/std come from the run's training samples only, are saved in
the run directory, and are applied identically at training, validation,
test and inference.

``normalise=0`` (unit vectors, angles, binary flags, segment ids) is one
statement — *this column's values already mean what they say* — so it gates
both run-level z-scoring **and** session-level percentile clipping. Clipping
a sparse binary mask to its 2nd/98th percentile collapses it to a constant,
and clipping a proximity feature truncates exactly the peaks that carry the
signal; those are values in a known range, not outliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ethograph.features.preprocessing import clip_by_percentiles, interpolate_nans
from ethograph.segment.config import PreprocessConfig


def preprocess_session_level(x: np.ndarray, cfg: PreprocessConfig, normalise: np.ndarray | None = None) -> np.ndarray:
    """Interpolate NaNs and clip outliers on one sample's ``(T, F)`` matrix.

    *normalise* is the layout's per-column flag; a column declaring
    ``normalise=0`` keeps its own scale and is not clipped. ``None`` clips
    every column — for a probe with no layout to consult.
    """
    x = np.asarray(x, dtype=np.float64)
    if cfg.interpolate:
        x = interpolate_nans(x, axis=0)
    if cfg.clip_percentiles is not None:
        clipped = clip_by_percentiles(x, percentile_range=cfg.clip_percentiles)
        if normalise is None:
            x = clipped
        else:
            x = np.where(np.asarray(normalise, dtype=bool)[None, :], clipped, x)
    return x


@dataclass
class NormStats:
    """Per-column mean/std of the training samples; ``normalise=0`` columns pass through."""

    mean: np.ndarray
    std: np.ndarray
    normalise: np.ndarray

    @classmethod
    def compute(cls, matrices: list[np.ndarray], normalise: np.ndarray) -> NormStats:
        """*matrices* are ``(F, T)`` arrays of the training samples."""
        if not matrices:
            raise ValueError("No training samples to compute normalisation statistics from.")
        stacked = np.concatenate([np.asarray(m, dtype=np.float64) for m in matrices], axis=1)
        mean = np.nanmean(stacked, axis=1)
        std = np.nanstd(stacked, axis=1)
        std[~np.isfinite(std) | (std == 0)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        normalise = np.asarray(normalise, dtype=bool)
        mean[~normalise] = 0.0
        std[~normalise] = 1.0
        return cls(mean=mean, std=std, normalise=normalise)

    @classmethod
    def identity(cls, n_features: int) -> NormStats:
        return cls(np.zeros(n_features), np.ones(n_features), np.zeros(n_features, dtype=bool))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Normalise an ``(F, T)`` matrix."""
        return (np.asarray(x, dtype=np.float32) - self.mean[:, None].astype(np.float32)) / self.std[:, None].astype(
            np.float32
        )

    def save(self, path: Path) -> Path:
        np.savez(path, mean=self.mean, std=self.std, normalise=self.normalise)
        return path

    @classmethod
    def load(cls, path: Path) -> NormStats:
        with np.load(path) as npz:
            return cls(mean=npz["mean"], std=npz["std"], normalise=npz["normalise"].astype(bool))
