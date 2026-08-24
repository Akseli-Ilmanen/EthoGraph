"""Training-time augmentation on one sample, before normalisation.

Noise and temporal stretch apply to every sample; mirror and rotation act
only on *vector groups* — columns spanning the space dim of one vector
(position, velocity, …), which the column layout records — so a dataset
without coordinates silently gets none of the geometry.
"""

from __future__ import annotations

import numpy as np

from ethograph.segment.config import AugmentConfig


def augment(
    x: np.ndarray,
    y: np.ndarray,
    cfg: AugmentConfig,
    vector_groups: list[list[int]],
    normalise: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``x`` is ``(F, T)``, ``y`` ``(T,)``; returns new arrays."""
    x = np.array(x, dtype=np.float32, copy=True)
    y = np.asarray(y)
    if cfg.stretch is not None:
        x, y = _stretch(x, y, rng.uniform(cfg.stretch[0], cfg.stretch[1]))
    if cfg.mirror and vector_groups and rng.random() < 0.5:
        for group in vector_groups:
            x[group[0]] *= -1.0
    if cfg.rotate_deg > 0 and vector_groups:
        theta = np.deg2rad(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
        c, s = np.cos(theta), np.sin(theta)
        for group in vector_groups:
            gx, gy = x[group[0]].copy(), x[group[1]].copy()
            x[group[0]] = c * gx - s * gy
            x[group[1]] = s * gx + c * gy
    if cfg.noise_std > 0:
        noise = rng.normal(0.0, cfg.noise_std, size=x.shape).astype(np.float32)
        noise[~np.asarray(normalise, dtype=bool)] = 0.0
        x = x + noise * np.nanstd(x, axis=1, keepdims=True)
    return x, y


def _stretch(x: np.ndarray, y: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[1]
    m = max(2, int(round(n * factor)))
    src = np.linspace(0, n - 1, m)
    base = np.arange(n)
    xs = np.stack([np.interp(src, base, row) for row in x]).astype(np.float32)
    ys = y[np.clip(np.rint(src).astype(int), 0, n - 1)]
    return xs, ys
