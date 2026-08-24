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
from ethograph.segment.samples import ClassTable, ColumnLayout

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
    classes: ClassTable
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
            x, y = np.ascontiguousarray(x[:, :: self.subsample]), np.ascontiguousarray(y[:: self.subsample])
        return x, y


class SampleDataset(Dataset):
    """Samples by key; augments (training only) then normalises."""

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

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        key = self.keys[i]
        x, y = self.store.load(key)
        if self.keep is not None:
            x = x[self.keep]
        if self.augment_cfg is not None:
            x, y = augment(x, y, self.augment_cfg, self.layout.vector_groups, self.stats.normalise, self.rng)
        x = self.stats.apply(x)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)), key


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pad to the longest sample: ``x (B, F, T)``, ``y (B, T)`` (pad −100), ``mask (B, 1, T)``."""
    n_features = batch[0][0].shape[0]
    t_max = max(x.shape[1] for x, _, _ in batch)
    x_out = torch.zeros(len(batch), n_features, t_max, dtype=torch.float32)
    y_out = torch.full((len(batch), t_max), PAD_TARGET, dtype=torch.long)
    mask = torch.zeros(len(batch), 1, t_max, dtype=torch.float32)
    keys = []
    for i, (x, y, key) in enumerate(batch):
        t = x.shape[1]
        x_out[i, :, :t] = x
        y_out[i, :t] = y
        mask[i, :, :t] = 1.0
        keys.append(key)
    return x_out, y_out, mask, keys
