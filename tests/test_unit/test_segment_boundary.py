"""The boundary channel: targets in seconds, a loss that learns, peaks, refinement.

The invariant this file exists to protect is that **nothing in the boundary
path is spelled in frames**. Every tolerance is a duration and is resolved
against the dataset's own sampling rate, so the same config means the same
thing at 15 fps and at 200 Hz.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ethograph.segment.boundary import (  # noqa: E402
    REFINEMENT_MODES,
    BoundaryLoss,
    boundary_peaks,
    boundary_probabilities,
    boundary_scores,
    boundary_targets,
    refine_with_boundary,
    snap_to_candidates,
    tolerance_frames,
)
from ethograph.segment.dataset import PAD_TARGET  # noqa: E402

FS = 200.0


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


class TestTolerance:
    def test_seconds_resolve_against_the_rate(self) -> None:
        assert tolerance_frames(0.01, 200.0) == 2
        assert tolerance_frames(0.01, 30.0) == 0
        assert tolerance_frames(0.05, 200.0) == 10

    def test_zero_seconds_is_a_single_frame(self) -> None:
        assert tolerance_frames(0.0, FS) == 0

    def test_an_unknown_rate_raises_rather_than_guessing(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            tolerance_frames(0.01, 0.0)


class TestTargets:
    def test_a_boundary_is_a_transition_not_the_first_frame(self) -> None:
        y, mask = _batch([_dense([(10, 20, 1)], 40)])
        target = boundary_targets(y, mask)
        assert target[0, 0, 0] == 0
        assert target[0, 0].nonzero().flatten().tolist() == [10, 20]

    def test_offsets_count_as_well_as_onsets(self) -> None:
        y, mask = _batch([_dense([(5, 10, 1), (10, 15, 2)], 30)])
        target = boundary_targets(y, mask)
        assert target[0, 0].nonzero().flatten().tolist() == [5, 10, 15]

    def test_tolerance_dilates_symmetrically(self) -> None:
        y, mask = _batch([_dense([(10, 20, 1)], 40)])
        target = boundary_targets(y, mask, tolerance=2)
        assert target[0, 0, 8:13].sum() == 5
        assert target[0, 0, 7] == 0

    def test_padding_is_never_a_boundary(self) -> None:
        y, mask = _batch([_dense([(5, 30, 1)], 40)], padded=10)
        target = boundary_targets(y, mask, tolerance=3)
        assert target[0, 0, 30:].sum() == 0
        assert target[0, 0, 5] == 1

    def test_a_wrong_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\(B, T\)"):
            boundary_targets(torch.zeros(4, 1, 8, dtype=torch.long), torch.ones(4, 1, 8))


class TestLoss:
    def test_perfect_logits_cost_less_than_wrong_ones(self) -> None:
        y, mask = _batch([_dense([(20, 60, 1), (60, 90, 2)], 200)])
        target = boundary_targets(y, mask, tolerance=2)
        good = ((target * 2 - 1) * 8).unsqueeze(0)
        bad = -good
        loss = BoundaryLoss(tolerance=2)
        assert float(loss(good, y, mask)) < float(loss(bad, y, mask))

    def test_it_trains_a_linear_probe_towards_the_target(self) -> None:
        y, mask = _batch([_dense([(20, 60, 1), (60, 90, 2)], 200)])
        logits = torch.zeros(1, 1, 1, 200, requires_grad=True)
        loss_fn = BoundaryLoss(tolerance=2)
        optimizer = torch.optim.Adam([logits], lr=0.5)
        first = float(loss_fn(logits, y, mask).detach())
        for _ in range(50):
            optimizer.zero_grad()
            loss = loss_fn(logits, y, mask)
            loss.backward()
            optimizer.step()
        assert float(loss_fn(logits, y, mask).detach()) < first
        peaks = boundary_peaks(torch.sigmoid(logits[-1, 0, 0]).detach().numpy(), 0.5)
        assert set(peaks.tolist()) <= {20, 60, 90}

    def test_padded_frames_do_not_reach_the_loss(self) -> None:
        row = _dense([(20, 60, 1)], 200)
        y, mask = _batch([row], padded=100)
        logits = torch.randn(1, 1, 1, 200)
        clean = logits.clone()
        clean[:, :, :, 100:] = torch.randn(1, 1, 1, 100) * 10
        loss_fn = BoundaryLoss()
        assert float(loss_fn(logits, y, mask)) == pytest.approx(float(loss_fn(clean, y, mask)), abs=1e-6)

    def test_the_focal_form_is_a_different_number_not_an_error(self) -> None:
        y, mask = _batch([_dense([(20, 60, 1)], 200)])
        logits = torch.randn(2, 1, 1, 200)
        plain = float(BoundaryLoss(tolerance=1)(logits, y, mask))
        focal = float(BoundaryLoss(tolerance=1, focal=True)(logits, y, mask))
        assert np.isfinite(plain) and np.isfinite(focal)
        assert plain != focal

    def test_a_pinned_positive_weight_is_used_verbatim(self) -> None:
        y, mask = _batch([_dense([(20, 60, 1)], 200)])
        logits = torch.zeros(1, 1, 1, 200)
        light = float(BoundaryLoss(pos_weight=1.0)(logits, y, mask))
        heavy = float(BoundaryLoss(pos_weight=50.0)(logits, y, mask))
        assert heavy > light

    def test_stages_are_averaged_not_summed(self) -> None:
        y, mask = _batch([_dense([(20, 60, 1)], 200)])
        one = torch.zeros(1, 1, 1, 200)
        three = one.repeat(3, 1, 1, 1)
        loss_fn = BoundaryLoss()
        assert float(loss_fn(three, y, mask)) == pytest.approx(float(loss_fn(one, y, mask)))

    def test_probabilities_read_the_last_stage(self) -> None:
        stages = torch.stack([torch.full((1, 1, 8), -10.0), torch.full((1, 1, 8), 10.0)])
        assert torch.all(boundary_probabilities(stages) > 0.99)


class TestPeaks:
    def test_only_local_maxima_above_the_threshold(self) -> None:
        prob = np.zeros(50)
        prob[10] = 0.9
        prob[30] = 0.3
        assert boundary_peaks(prob, 0.5).tolist() == [10]

    def test_frame_zero_is_never_claimed_as_a_prediction(self) -> None:
        prob = np.ones(20)
        assert 0 not in boundary_peaks(prob, 0.5).tolist()

    def test_a_short_signal_has_no_peaks(self) -> None:
        assert boundary_peaks(np.array([1.0, 1.0]), 0.5).size == 0


class TestSnapping:
    def test_a_peak_moves_onto_its_nearest_candidate(self) -> None:
        assert snap_to_candidates(np.array([12]), np.array([10, 40]), max_shift=5).tolist() == [10]

    def test_a_peak_with_nothing_nearby_is_dropped(self) -> None:
        assert snap_to_candidates(np.array([25]), np.array([10, 40]), max_shift=5).size == 0

    def test_two_peaks_on_one_candidate_collapse(self) -> None:
        assert snap_to_candidates(np.array([9, 11]), np.array([10]), max_shift=3).tolist() == [10]

    def test_no_candidates_means_no_boundaries(self) -> None:
        assert snap_to_candidates(np.array([5, 9]), np.array([]), max_shift=5).size == 0


class TestRefinement:
    def test_none_is_the_identity(self) -> None:
        pred = _dense([(10, 20, 1)], 40)
        assert np.array_equal(refine_with_boundary(pred, np.ones(40), 0.5, mode="none"), pred)

    def test_a_span_takes_its_majority_class(self) -> None:
        pred = _dense([(10, 19, 1), (19, 20, 2)], 40)
        prob = np.zeros(40)
        prob[10] = prob[20] = 1.0
        out = refine_with_boundary(pred, prob, 0.5)
        assert set(out[10:20].tolist()) == {1}
        assert set(out[20:].tolist()) == {0}

    def test_it_cleans_a_hole_the_frame_argmax_left(self) -> None:
        pred = _dense([(10, 20, 1), (14, 16, 0), (16, 20, 1)], 40)
        prob = np.zeros(40)
        prob[10] = prob[20] = 1.0
        assert set(refine_with_boundary(pred, prob, 0.5)[10:20].tolist()) == {1}

    def test_hybrid_only_cuts_where_the_physics_agrees(self) -> None:
        pred = _dense([(10, 30, 1)], 60)
        prob = np.zeros(60)
        prob[12] = prob[45] = 1.0
        # Only frame 10 is a detected changepoint, so the peak at 45 is dropped
        # and the whole tail becomes one span.
        out = refine_with_boundary(pred, prob, 0.5, mode="hybrid", candidates=np.array([10]), max_shift=3)
        assert out[:10].tolist() == [0] * 10
        assert len(np.unique(out[10:])) == 1

    def test_hybrid_without_candidates_says_what_to_turn_on(self) -> None:
        with pytest.raises(ValueError, match="changepoint_correction"):
            refine_with_boundary(np.zeros(10, dtype=int), np.zeros(10), 0.5, mode="hybrid")

    def test_an_unknown_mode_names_the_ones_that_exist(self) -> None:
        with pytest.raises(ValueError, match="predicted"):
            refine_with_boundary(np.zeros(10, dtype=int), np.zeros(10), 0.5, mode="snap")

    def test_every_mode_is_reachable(self) -> None:
        pred = _dense([(10, 20, 1)], 40)
        prob = np.zeros(40)
        prob[10] = prob[20] = 1.0
        for mode in REFINEMENT_MODES:
            candidates = np.array([10, 20]) if mode == "hybrid" else None
            out = refine_with_boundary(pred, prob, 0.5, mode=mode, candidates=candidates, max_shift=2)
            assert out.shape == pred.shape


class TestScores:
    def test_a_perfect_prediction_scores_one_hundred(self) -> None:
        gt = _dense([(10, 20, 1)], 40)
        prob = np.zeros(40)
        prob[10] = prob[20] = 1.0
        scores = boundary_scores(gt, prob, 0.5, tolerance=0)
        assert scores["boundary_f1"] == pytest.approx(100.0)

    def test_a_near_miss_counts_inside_the_tolerance(self) -> None:
        gt = _dense([(10, 20, 1)], 40)
        prob = np.zeros(40)
        prob[12] = prob[22] = 1.0
        assert boundary_scores(gt, prob, 0.5, tolerance=0)["boundary_recall"] == 0.0
        assert boundary_scores(gt, prob, 0.5, tolerance=3)["boundary_recall"] == pytest.approx(100.0)

    def test_one_peak_cannot_claim_two_transitions(self) -> None:
        gt = _dense([(10, 12, 1)], 40)
        prob = np.zeros(40)
        prob[11] = 1.0
        scores = boundary_scores(gt, prob, 0.5, tolerance=5)
        assert scores["boundary_recall"] == pytest.approx(50.0)
        assert scores["boundary_precision"] == pytest.approx(100.0)

    def test_nothing_to_find_and_nothing_found_is_not_a_failure(self) -> None:
        assert boundary_scores(np.zeros(40, dtype=int), np.zeros(40), 0.5, 0)["boundary_f1"] == 100.0
