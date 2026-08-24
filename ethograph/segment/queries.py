"""The segment-level objective: BaFormer's set prediction, in our layout.

A frame-wise loss optimises a different thing from the segmental metric being
reported. A query-based head predicts *segments as instances* — each query
emits a class and a soft mask over the timeline — and the loss matches them
one-to-one against the true segments, so the objective is IoU-shaped from the
start (`arXiv:2405.15995 <https://arxiv.org/abs/2405.15995>`_).

This module is the training half of that: the targets, the Hungarian matcher
and the set criterion. The model half lives in
:mod:`ethograph.segment.models.baformer`; the class-agnostic boundary term
they share lives in :mod:`ethograph.segment.boundary`.

Three deliberate departures from upstream:

* **Ours are dense per-frame targets, not a file of segments.** Everything
  the pipeline stores is a dense class-index array, so
  :func:`segment_targets` derives the instances from it — one target per
  maximal run of a class, background included. That keeps one source of
  truth and makes the head swappable against any other architecture.
* **Too many segments is an error, not a silent truncation.** A trial with
  more runs than the head has queries cannot be matched, and upstream would
  quietly drop the overflow. :class:`SetCriterion` raises and names
  ``model.params.num_queries``.
* **No dependency on detectron2 or einops.** The matcher is
  ``scipy.optimize.linear_sum_assignment`` (scipy is already a core
  dependency) and the attention is plain ``torch``.

Covered by ``tests/test_unit/test_segment_queries.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from ethograph.segment.boundary import BoundaryLoss
from ethograph.segment.dataset import PAD_TARGET


@dataclass
class SegmentTargets:
    """The true segments of one sample, as instances.

    ``labels`` ``(N,)`` class indices, ``masks`` ``(N, T)`` in ``{0, 1}``, one
    row per maximal run of a class. Padded frames belong to no segment.
    """

    labels: torch.Tensor
    masks: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def segment_targets(y: torch.Tensor, mask: torch.Tensor) -> list[SegmentTargets]:
    """``(B, T)`` dense class indices → one :class:`SegmentTargets` per sample.

    Background runs are targets like any other class: a query that claims a
    stretch of background is making a real, checkable statement, and leaving
    those spans unmatched would let the head cover them with anything.
    """
    out: list[SegmentTargets] = []
    for row, valid in zip(y, mask[:, 0, :] > 0):
        labels = row[valid]
        if labels.numel() == 0:
            out.append(SegmentTargets(row.new_zeros(0, dtype=torch.long), row.new_zeros((0, row.shape[0]))))
            continue
        labels = torch.where(labels == PAD_TARGET, torch.zeros_like(labels), labels)
        index = torch.nonzero(valid, as_tuple=False)[:, 0]
        starts = torch.cat([labels.new_ones(1, dtype=torch.bool), labels[1:] != labels[:-1]])
        run_start = torch.nonzero(starts, as_tuple=False)[:, 0]
        run_stop = torch.cat([run_start[1:], run_start.new_tensor([labels.numel()])])
        masks = torch.zeros((run_start.numel(), row.shape[0]), dtype=torch.float32, device=row.device)
        for i, (a, b) in enumerate(zip(run_start.tolist(), run_stop.tolist())):
            masks[i, index[a:b]] = 1.0
        out.append(SegmentTargets(labels[run_start].long(), masks))
    return out


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft-IoU-shaped loss between ``(N, T)`` mask logits and their targets."""
    prob = logits.sigmoid()
    numerator = 2 * (prob * targets).sum(dim=-1)
    denominator = prob.sum(dim=-1) + targets.sum(dim=-1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


def sigmoid_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Focal BCE over ``(N, T)`` mask logits — a segment is a thin slice of the timeline."""
    prob = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = bce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean(dim=-1).mean()


class HungarianMatcher(nn.Module):
    """One-to-one assignment of queries to true segments, by class + mask cost.

    The cost of pairing query *q* with target *n* is
    ``-p(class_n | q) * w_class + focal(mask_q, mask_n) * w_mask +
    dice(mask_q, mask_n) * w_dice`` — the same three terms the loss uses, so
    the assignment and the gradient agree about what a good match is.
    """

    def __init__(self, class_weight: float = 1.0, mask_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        if class_weight == mask_weight == dice_weight == 0:
            raise ValueError("A matcher with all three costs at zero has nothing to match on.")
        self.class_weight = float(class_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)

    @torch.no_grad()
    def forward(
        self, query_logits: torch.Tensor, query_masks: torch.Tensor, targets: list[SegmentTargets]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """``(B, Q, C+1)`` + ``(B, Q, T)`` + targets → per sample ``(query index, target index)``."""
        out = []
        for logits, masks, target in zip(query_logits, query_masks, targets):
            if len(target) == 0:
                empty = logits.new_zeros(0, dtype=torch.long)
                out.append((empty, empty))
                continue
            prob = logits.softmax(dim=-1)
            cost_class = -prob[:, target.labels]
            tgt = target.masks.to(masks.dtype)
            cost_mask = _pairwise_focal(masks, tgt)
            cost_dice = _pairwise_dice(masks, tgt)
            cost = self.class_weight * cost_class + self.mask_weight * cost_mask + self.dice_weight * cost_dice
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            out.append(
                (
                    torch.as_tensor(rows, dtype=torch.long, device=logits.device),
                    torch.as_tensor(cols, dtype=torch.long, device=logits.device),
                )
            )
        return out


def _pairwise_focal(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
    """``(Q, T)`` × ``(N, T)`` → ``(Q, N)`` focal cost."""
    prob = logits.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits), reduction="none"
    )
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
    pos = alpha * focal_pos
    neg = (1 - alpha) * focal_neg
    return (pos @ targets.T + neg @ (1 - targets).T) / logits.shape[-1]


def _pairwise_dice(logits: torch.Tensor, targets: torch.Tensor):
    """``(Q, T)`` × ``(N, T)`` → ``(Q, N)`` dice cost."""
    prob = logits.sigmoid()
    numerator = 2 * (prob @ targets.T)
    denominator = prob.sum(dim=-1)[:, None] + targets.sum(dim=-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


class SetCriterion(nn.Module):
    """BaFormer's loss: matched class + mask + dice, plus the boundary term.

    Queries left unmatched are trained towards the "no segment" class, whose
    weight in the classification cross-entropy is *eos_coef* — without it the
    head learns that predicting nothing is always safe.

    With *deep_supervision* every decoder level is matched and scored, not
    just the last; upstream does the same, and it is what makes a ten-level
    decoder trainable.
    """

    def __init__(
        self,
        n_classes: int,
        matcher: HungarianMatcher,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        boundary_weight: float = 1.0,
        eos_coef: float = 0.1,
        label_smoothing: float = 0.0,
        deep_supervision: bool = True,
        boundary_loss: BoundaryLoss | None = None,
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.matcher = matcher
        self.class_weight = float(class_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)
        self.boundary_weight = float(boundary_weight)
        self.label_smoothing = float(label_smoothing)
        self.deep_supervision = bool(deep_supervision)
        self.boundary_loss = boundary_loss
        empty_weight = torch.ones(self.n_classes + 1)
        empty_weight[-1] = float(eos_coef)
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        query_logits: torch.Tensor,
        query_masks: torch.Tensor,
        boundary: torch.Tensor | None,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``(L, B, Q, C+1)`` + ``(L, B, Q, T)`` (+ boundary) → total loss and its parts."""
        targets = segment_targets(y, mask)
        self._check_capacity(query_logits.shape[2], targets)
        levels = range(query_logits.shape[0]) if self.deep_supervision else [query_logits.shape[0] - 1]
        total = query_logits.new_zeros(())
        parts: dict[str, float] = {}
        for level in levels:
            loss, level_parts = self._level_loss(query_logits[level], query_masks[level], targets, mask)
            total = total + loss
            if level == query_logits.shape[0] - 1:
                parts.update(level_parts)
        total = total / len(list(levels))
        if boundary is not None and self.boundary_loss is not None and self.boundary_weight:
            boundary_term = self.boundary_loss(boundary, y, mask)
            total = total + self.boundary_weight * boundary_term
            parts["query_boundary"] = float(boundary_term.detach())
        return total, parts

    def _check_capacity(self, n_queries: int, targets: list[SegmentTargets]) -> None:
        worst = max((len(t) for t in targets), default=0)
        if worst > n_queries:
            raise ValueError(
                f"A sample in this batch has {worst} segments but the head has only {n_queries} queries, "
                "so they cannot be matched one-to-one. Raise model.params.num_queries to at least "
                f"{2 * worst} (roughly twice the worst trial, as the paper recommends)."
            )

    def _level_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        targets: list[SegmentTargets],
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        indices = self.matcher(logits, masks, targets)
        target_classes = torch.full(logits.shape[:2], self.n_classes, dtype=torch.long, device=logits.device)
        for b, (src, tgt) in enumerate(indices):
            target_classes[b, src] = targets[b].labels[tgt]
        loss_class = F.cross_entropy(
            logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight.to(logits.dtype),
            label_smoothing=self.label_smoothing,
        )
        src_masks, tgt_masks = [], []
        for b, (src, tgt) in enumerate(indices):
            if src.numel() == 0:
                continue
            valid = frame_mask[b, 0] > 0
            src_masks.append(masks[b][src][:, valid])
            tgt_masks.append(targets[b].masks[tgt][:, valid].to(masks.dtype))
        if src_masks:
            src_cat = torch.cat(src_masks)
            tgt_cat = torch.cat(tgt_masks)
            loss_mask = sigmoid_focal_loss(src_cat, tgt_cat)
            loss_dice = dice_loss(src_cat, tgt_cat)
        else:
            loss_mask = logits.new_zeros(())
            loss_dice = logits.new_zeros(())
        total = self.class_weight * loss_class + self.mask_weight * loss_mask + self.dice_weight * loss_dice
        parts = {
            "query_class": float(loss_class.detach()),
            "query_mask": float(loss_mask.detach()),
            "query_dice": float(loss_dice.detach()),
        }
        return total, parts
