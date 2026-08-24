"""The segment-query objective: targets from dense labels, matching, and the set loss.

Two invariants are worth more than the rest. **Targets come from the dense
array, not from a segment file** — that is what keeps one source of truth
between the frame loss and the set loss. And **a trial with more segments
than the head has queries is an error**, because a one-to-many match that
silently drops the overflow trains a model whose worst trials are invisible.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ethograph.segment.boundary import BoundaryLoss  # noqa: E402
from ethograph.segment.dataset import PAD_TARGET  # noqa: E402
from ethograph.segment.queries import (  # noqa: E402
    HungarianMatcher,
    SetCriterion,
    dice_loss,
    segment_targets,
    sigmoid_focal_loss,
)

N_CLASSES = 4


def _dense(spans: list[tuple[int, int, int]], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int64)
    for start, stop, label in spans:
        out[start:stop] = label
    return out


def _batch(rows: list[np.ndarray], padded: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(r) for r in rows)
    y = torch.full((len(rows), n), PAD_TARGET, dtype=torch.long)
    mask = torch.zeros(len(rows), 1, n)
    for i, row in enumerate(rows):
        keep = len(row) - padded
        y[i, :keep] = torch.from_numpy(row[:keep])
        mask[i, :, :keep] = 1.0
    return y, mask


class TestTargets:
    def test_one_target_per_run_including_background(self) -> None:
        y, mask = _batch([_dense([(10, 20, 1), (20, 30, 2)], 40)])
        target = segment_targets(y, mask)[0]
        assert target.labels.tolist() == [0, 1, 2, 0]
        assert len(target) == 4

    def test_the_masks_tile_the_valid_timeline_exactly_once(self) -> None:
        y, mask = _batch([_dense([(10, 20, 1), (25, 30, 3)], 40)])
        target = segment_targets(y, mask)[0]
        assert torch.equal(target.masks.sum(dim=0), torch.ones(40))

    def test_a_repeated_class_is_two_targets_not_one(self) -> None:
        y, mask = _batch([_dense([(5, 10, 1), (15, 20, 1)], 30)])
        target = segment_targets(y, mask)[0]
        assert target.labels.tolist() == [0, 1, 0, 1, 0]

    def test_padding_belongs_to_no_segment(self) -> None:
        y, mask = _batch([_dense([(5, 15, 1)], 40)], padded=10)
        target = segment_targets(y, mask)[0]
        assert target.masks[:, 30:].sum() == 0
        assert target.masks.sum() == 30

    def test_an_entirely_padded_sample_has_no_targets(self) -> None:
        y = torch.full((1, 20), PAD_TARGET, dtype=torch.long)
        assert len(segment_targets(y, torch.zeros(1, 1, 20))[0]) == 0


class TestMatcher:
    def test_the_obvious_assignment_is_found(self) -> None:
        y, mask = _batch([_dense([(10, 20, 1)], 30)])
        targets = segment_targets(y, mask)
        n = len(targets[0])
        logits = torch.full((1, n, N_CLASSES + 1), -5.0)
        masks = torch.full((1, n, 30), -5.0)
        # Query i is built to be exactly target i, in reverse order.
        for i in range(n):
            logits[0, n - 1 - i, targets[0].labels[i]] = 5.0
            masks[0, n - 1 - i] = targets[0].masks[i] * 10 - 5
        src, tgt = HungarianMatcher()(logits, masks, targets)[0]
        assert [int(t) for _, t in sorted(zip(src.tolist(), tgt.tolist()))] == list(range(n))[::-1]

    def test_a_sample_with_no_targets_matches_nothing(self) -> None:
        y = torch.full((1, 20), PAD_TARGET, dtype=torch.long)
        targets = segment_targets(y, torch.zeros(1, 1, 20))
        src, tgt = HungarianMatcher()(torch.zeros(1, 5, N_CLASSES + 1), torch.zeros(1, 5, 20), targets)[0]
        assert src.numel() == 0 and tgt.numel() == 0

    def test_the_assignment_is_one_to_one(self) -> None:
        y, mask = _batch([_dense([(5, 10, 1), (12, 18, 2), (20, 25, 3)], 30)])
        targets = segment_targets(y, mask)
        torch.manual_seed(0)
        src, tgt = HungarianMatcher()(torch.randn(1, 12, N_CLASSES + 1), torch.randn(1, 12, 30), targets)[0]
        assert len(set(src.tolist())) == len(src) == len(targets[0])
        assert sorted(tgt.tolist()) == list(range(len(targets[0])))

    def test_a_matcher_with_no_cost_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to match"):
            HungarianMatcher(0, 0, 0)


class TestTerms:
    def test_dice_rewards_overlap(self) -> None:
        target = torch.zeros(1, 40)
        target[0, 10:20] = 1.0
        good = target * 20 - 10
        bad = -good
        assert float(dice_loss(good, target)) < float(dice_loss(bad, target))

    def test_focal_rewards_overlap(self) -> None:
        target = torch.zeros(1, 40)
        target[0, 10:20] = 1.0
        good = target * 20 - 10
        assert float(sigmoid_focal_loss(good, target)) < float(sigmoid_focal_loss(-good, target))


def _criterion(**kwargs) -> SetCriterion:
    defaults = dict(n_classes=N_CLASSES, matcher=HungarianMatcher(2.0, 5.0, 5.0), boundary_weight=0.0)
    return SetCriterion(**{**defaults, **kwargs})


class TestSetCriterion:
    def test_a_perfect_head_costs_less_than_a_random_one(self) -> None:
        y, mask = _batch([_dense([(10, 20, 1), (25, 35, 2)], 50)])
        targets = segment_targets(y, mask)
        n_queries = 8
        logits = torch.full((1, 1, n_queries, N_CLASSES + 1), -6.0)
        masks = torch.full((1, 1, n_queries, 50), -6.0)
        logits[..., N_CLASSES] = 6.0
        for i in range(len(targets[0])):
            logits[0, 0, i] = -6.0
            logits[0, 0, i, targets[0].labels[i]] = 6.0
            masks[0, 0, i] = targets[0].masks[i] * 12 - 6
        criterion = _criterion()
        good, parts = criterion(logits, masks, None, y, mask)
        torch.manual_seed(0)
        bad, _ = criterion(torch.randn_like(logits), torch.randn_like(masks), None, y, mask)
        assert float(good) < float(bad)
        assert set(parts) == {"query_class", "query_mask", "query_dice"}

    def test_too_few_queries_names_the_setting_to_raise(self) -> None:
        y, mask = _batch([_dense([(2, 4, 1), (6, 8, 2), (10, 12, 3)], 20)])
        criterion = _criterion()
        with pytest.raises(ValueError, match="model.params.num_queries"):
            criterion(torch.zeros(1, 1, 2, N_CLASSES + 1), torch.zeros(1, 1, 2, 20), None, y, mask)

    def test_deep_supervision_scores_every_level(self) -> None:
        y, mask = _batch([_dense([(10, 20, 1)], 40)])
        torch.manual_seed(0)
        logits = torch.randn(3, 1, 6, N_CLASSES + 1)
        masks = torch.randn(3, 1, 6, 40)
        # Make the last level identical to the first, so a last-level-only
        # criterion cannot tell the two stacks apart but a deep one can.
        logits[2] = logits[0]
        masks[2] = masks[0]
        shallow = float(_criterion(deep_supervision=False)(logits, masks, None, y, mask)[0])
        deep = float(_criterion(deep_supervision=True)(logits, masks, None, y, mask)[0])
        assert shallow != deep

    def test_the_boundary_term_is_added_when_it_has_a_weight(self) -> None:
        y, mask = _batch([_dense([(10, 20, 1)], 40)])
        torch.manual_seed(0)
        logits = torch.randn(1, 1, 6, N_CLASSES + 1)
        masks = torch.randn(1, 1, 6, 40)
        boundary = torch.randn(1, 1, 1, 40)
        without = _criterion()(logits, masks, boundary, y, mask)
        with_boundary = _criterion(boundary_weight=1.0, boundary_loss=BoundaryLoss())(logits, masks, boundary, y, mask)
        assert "query_boundary" not in without[1]
        assert "query_boundary" in with_boundary[1]
        assert float(with_boundary[0]) > float(without[0])

    def test_gradients_reach_both_heads(self) -> None:
        y, mask = _batch([_dense([(10, 20, 1)], 40)])
        logits = torch.randn(1, 1, 6, N_CLASSES + 1, requires_grad=True)
        masks = torch.randn(1, 1, 6, 40, requires_grad=True)
        loss, _ = _criterion()(logits, masks, None, y, mask)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        assert masks.grad is not None and torch.isfinite(masks.grad).all()
