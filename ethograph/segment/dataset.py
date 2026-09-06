"""Torch access to a materialised dataset."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ethograph.segment.augment import augment
from ethograph.segment.config import AugmentConfig
from ethograph.segment.materialise import load_sample, read_classes, read_index, read_layout
from ethograph.segment.preprocess import NormStats
from ethograph.segment.samples import ColumnLayout, TargetTable

PAD_TARGET = -100


@dataclass
class MaterialisedStore:
    """A materialised dataset, optionally read at a fraction of its own rate.

    *subsample* takes every *k*-th frame of every sample, and the layout it
    hands out says ``fs / k`` — so a run at a lower temporal resolution is a
    different store over the same files, and everything downstream (the
    metrics, the post-processing durations) follows the rate it reports
    rather than the rate on disk.

    Striding, with no anti-alias filter — so a model that does worse at
    ``k > 1`` bounds from above what temporal resolution alone is worth,
    since it also pays for whatever aliasing the missing filter lets in.
    """

    data_dir: Path
    layout: ColumnLayout
    classes: TargetTable
    index: pd.DataFrame
    #: Frame stride; ``1`` is the dataset's own rate (``train.subsample``).
    subsample: int = 1

    @classmethod
    def open(cls, data_dir: Path, subsample: int = 1) -> MaterialisedStore:
        subsample = int(subsample)
        if subsample < 1:
            raise ValueError(f"train.subsample={subsample} — a frame stride is 1 (the dataset's rate) or more.")
        layout = read_layout(data_dir)
        if subsample > 1:
            layout = replace(layout, fs=layout.fs / subsample)
        return cls(data_dir, layout, read_classes(data_dir), read_index(data_dir), subsample)

    @property
    def keys(self) -> list[str]:
        return list(self.index["key"])

    def load(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        x, y = load_sample(self.data_dir, key, self.classes)
        if self.subsample > 1:
            x, y = np.ascontiguousarray(x[:, :: self.subsample]), np.ascontiguousarray(y[..., :: self.subsample])
        return x, y


class SampleDataset(Dataset):
    """Samples by key; augments (training only), then ablates, then normalises.

    Every item also carries its **candidate frames** — where any raw
    changepoint mask fires (:meth:`ColumnLayout.candidate_columns`), read off
    the full sample *before* an ablation drops columns, so ``train.drop_kinds``
    changes what the model sees and never what the loss is told. All-``False``
    when the layout has no such column.
    """

    def __init__(
        self,
        store: MaterialisedStore,
        keys: list[str],
        stats: NormStats,
        augment_cfg: AugmentConfig | None = None,
        seed: int = 0,
        keep: np.ndarray | None = None,
        layout: ColumnLayout | None = None,
    ) -> None:
        self.store = store
        self.keys = list(keys)
        self.stats = stats
        self.augment_cfg = augment_cfg
        self.rng = np.random.default_rng(seed)
        #: Column mask of an ablation (``None`` = every column).
        self.keep = keep
        #: The layout the model sees — the store's, or its ablated subset.
        self.layout = layout if layout is not None else store.layout

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        key = self.keys[i]
        x, y = self.store.load(key)
        full = self.store.layout
        if self.augment_cfg is not None:
            # On the full sample, so the candidate frames are stretched with it.
            normalise = np.asarray(full.normalise, dtype=bool)
            x, y = augment(x, y, self.augment_cfg, full.vector_groups, normalise, self.rng)
        candidates = self._candidates(x)
        if self.keep is not None:
            x = x[self.keep]
        x = self.stats.apply(x)
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
            torch.from_numpy(candidates),
            key,
        )

    def _candidates(self, x: np.ndarray) -> np.ndarray:
        """``(T,)`` bool: frames where any raw changepoint mask fires (``> 0.5`` survives a stretch's interpolation)."""
        cols = self.store.layout.candidate_columns()
        if cols.size == 0:
            return np.zeros(x.shape[1], dtype=bool)
        return np.asarray(x[cols] > 0.5).any(axis=0)


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pad to the longest sample: ``x (B, F, T)``, ``y (B, T)`` (pad −100), ``mask (B, 1, T)``, ``candidates (B, T)``.

    A multi-label target pads to ``y (B, C, T)`` — time is the last axis of
    either shape. *candidates* is bool, ``False`` on padding.
    """
    n_features = batch[0][0].shape[0]
    t_max = max(x.shape[1] for x, _, _, _ in batch)
    x_out = torch.zeros(len(batch), n_features, t_max, dtype=torch.float32)
    y0 = batch[0][1]
    y_shape = (len(batch), *y0.shape[:-1], t_max)
    y_out = torch.full(y_shape, PAD_TARGET, dtype=torch.long)
    mask = torch.zeros(len(batch), 1, t_max, dtype=torch.float32)
    candidates = torch.zeros(len(batch), t_max, dtype=torch.bool)
    keys = []
    for i, (x, y, cand, key) in enumerate(batch):
        t = x.shape[1]
        x_out[i, :, :t] = x
        y_out[i, ..., :t] = y
        mask[i, :, :t] = 1.0
        candidates[i, :t] = cand
        keys.append(key)
    return x_out, y_out, mask, candidates, keys
