"""The pose-only spotter: features → multi-scale temporal shifts → bi-GRU → per-frame softmax.

The teacher of the distillation recipe (:mod:`~ethograph.spot.teacher`) and
nothing else. Its input is the feature columns a config lists — a few
positions, velocities and the distances the user wrote down — so the model
has no geometry to discover, only timing: *when* does the closest stick part
reach the pellet, *when* does it leave. Two things give it that:

* a **parameter-free multi-scale temporal shift** (UMEG-Net's, Liu et al.
  2026, eq. 3): a fraction of the channels is copied from ``t - k`` and
  ``t + k`` for several ``k``, so a block sees several ranges of context
  without extra parameters — the right regime for a few hundred events;
* a **bidirectional GRU**: "the last frame of contact" is defined by what
  happens after it, so the judgement is non-causal — the same reason the
  pixel model's head is a bi-GRU.

Shapes: input ``(B, T, F)`` — a clip of *T* frames of *F* features — and
output per-frame logits ``(B, T, K + 1)`` with background at index 0, the
contract E2E-Spot's head emits, so everything downstream of the softmax is
shared with the pixel model. :meth:`PoseSpotter.features` is the per-frame
embedding a student distils from.

Every temporal offset is in **samples of the input's own clock**; the config
resolves them from milliseconds (:meth:`TeacherConfig.shift_samples`).
"""

from __future__ import annotations

import torch
from torch import nn


def shift_channels(h: torch.Tensor, offset: int, n_shift: int) -> torch.Tensor:
    """The last ``2 * n_shift`` channels shifted by ``offset`` frames.

    The first ``n_shift`` of them read the past (frame ``t - offset``), the
    next ``n_shift`` the future (frame ``t + offset``); the rest are static.
    Zero-padded at the clip's ends. *h* is ``(B, T, d)``.
    """
    if offset < 1:
        raise ValueError(f"offset must be >= 1 sample, got {offset}")
    d = h.shape[-1]
    static, fwd, bwd = torch.split(h, [d - 2 * n_shift, n_shift, n_shift], dim=-1)
    fwd_shifted = torch.zeros_like(fwd)
    bwd_shifted = torch.zeros_like(bwd)
    t = h.shape[1]
    if offset < t:
        fwd_shifted[:, offset:] = fwd[:, :-offset]
        bwd_shifted[:, :-offset] = bwd[:, offset:]
    return torch.cat([static, fwd_shifted, bwd_shifted], dim=-1)


class ShiftBlock(nn.Module):
    """Multi-scale shift → one linear per scale → fuse → residual, with a LayerNorm."""

    def __init__(self, width: int, scales: list[int], shift_fraction: float) -> None:
        super().__init__()
        if not scales:
            raise ValueError("a block needs at least one shift scale")
        self.scales = list(scales)
        self.n_shift = max(1, int(round(width * shift_fraction)))
        if 2 * self.n_shift >= width:
            raise ValueError(f"shift_fraction {shift_fraction} leaves no static channels at width={width}")
        branch = max(1, width // len(self.scales))
        self.narrow = nn.ModuleList([nn.Linear(width, branch) for _ in self.scales])
        self.fuse = nn.Linear(branch * len(self.scales), width)
        self.norm = nn.LayerNorm(width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        parts = [
            narrow(torch.relu(shift_channels(h, scale, self.n_shift)))
            for scale, narrow in zip(self.scales, self.narrow)
        ]
        return self.norm(self.fuse(torch.relu(torch.cat(parts, dim=-1))) + h)


class PoseSpotter(nn.Module):
    """Shift blocks over the listed features, read by a bi-GRU."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        scales: list[int],
        hidden: int = 64,
        depth: int = 4,
        shift_fraction: float = 0.125,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("the pose model needs at least one feature column — features: is empty")
        if n_classes < 1:
            raise ValueError("n_classes counts the foreground classes and must be >= 1")
        self.n_features = int(n_features)
        self.embed = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList([ShiftBlock(hidden, scales, shift_fraction) for _ in range(depth)])
        self.head = nn.GRU(hidden, head_hidden, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(2 * head_hidden, n_classes + 1)

    @property
    def embed_dim(self) -> int:
        """Width of :meth:`features` — what a student's projection must match."""
        return self.head.hidden_size * 2

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame embeddings ``(B, T, embed_dim)`` — the distillation target."""
        if x.ndim != 3 or x.shape[-1] != self.n_features:
            raise ValueError(f"expected (B, T, {self.n_features}), got {tuple(x.shape)}")
        h = torch.relu(self.embed(torch.nan_to_num(x, nan=0.0)))
        for block in self.blocks:
            h = block(h)
        out, _ = self.head(h)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.features(x))
