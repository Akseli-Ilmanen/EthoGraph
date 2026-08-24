"""The class-agnostic boundary channel: targets, loss, peaks, refinement.

Reweighting a frame-wise objective adds no *localisation* gradient — the
question asked of the model stays "is frame t class c?", never "where is the
transition?". ASRF's answer is a second output channel that predicts, for
every frame, whether a segment boundary sits there
(`arXiv:2007.06866 <https://arxiv.org/abs/2007.06866>`_). This module is that
channel, end to end:

* :func:`boundary_targets` — the binary target, a 1 at every class transition
  of the dense labels, **dilated in seconds** (never in frames: at 200 Hz the
  literature's ±4 frames is ±20 ms, which is not what those papers meant).
* :class:`BoundaryLoss` — masked BCE over the model's boundary stages,
  positive-weighted because boundaries are ~1 % of frames.
* :func:`boundary_peaks` / :func:`refine_with_boundary` — inference. The
  timeline is cut at the peaks of the predicted boundary probability and each
  span takes the majority class, which is ASRF's refinement step.
* :func:`boundary_scores` — how well the channel itself localises, in
  seconds of tolerance, independent of the class prediction.

**The hybrid mode is the point of having this next to a changepoint
detector.** ``refine_with_boundary(..., mode="hybrid", candidates=...)`` keeps
only the predicted peaks that land within ``max_shift`` frames of a detected
changepoint and moves them onto it: the physical prior (a syllable boundary
sits at a speed minimum) stays a hard constraint, and the network only has to
*select* which of the overspecified changepoints are real. That is the learned
version of the hand rule in
:func:`~ethograph.features.changepoints.correct_changepoints`.

Covered by ``tests/test_unit/test_segment_boundary.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ethograph.segment.dataset import PAD_TARGET

REFINEMENT_MODES = ("none", "predicted", "hybrid")
"""What :func:`refine_with_boundary` may be asked to do.

``none`` leaves the dense prediction alone (the existing purge → stitch →
snap-to-changepoint pipeline is the whole of post-processing); ``predicted``
cuts at the model's own boundary peaks; ``hybrid`` cuts at the peaks that
coincide with a detected changepoint, snapped onto it.
"""


def tolerance_frames(tolerance_s: float, fs: float) -> int:
    """Seconds of tolerance → a half-width in frames, at least 0.

    Every boundary setting in this module is spelled in seconds and resolved
    against the materialised dataset's own sampling rate. Nothing here has a
    frame default.
    """
    if fs <= 0:
        raise ValueError(f"Sampling rate must be positive to resolve a tolerance of {tolerance_s} s, got fs={fs}")
    return max(int(round(float(tolerance_s) * float(fs))), 0)


def boundary_targets(y: torch.Tensor, mask: torch.Tensor, tolerance: int = 0) -> torch.Tensor:
    """``(B, T)`` dense class indices → ``(B, 1, T)`` boundary targets in ``{0, 1}``.

    A boundary is a frame whose class differs from the frame before it, so
    onsets and offsets both count (an offset is a transition *to* background).
    Frame 0 is not a boundary: the trial simply starts, and marking it would
    train the head on a transition that is not one.

    *tolerance* dilates each boundary by that many frames either side — the
    target the model can actually hit when the label itself is only accurate
    to a few milliseconds. ``0`` is the single-frame target ASRF uses.

    Padded frames (``mask == 0``) are zeroed, and so is the step into the
    padding, which is not a boundary of the behaviour.
    """
    if y.dim() != 2:
        raise ValueError(f"boundary_targets expects y of shape (B, T), got {tuple(y.shape)}")
    valid = (mask[:, :1, :] > 0).float()
    labels = torch.where(y == PAD_TARGET, torch.zeros_like(y), y).unsqueeze(1)
    changed = torch.zeros_like(labels, dtype=torch.float32)
    changed[:, :, 1:] = (labels[:, :, 1:] != labels[:, :, :-1]).float() * valid[:, :, 1:] * valid[:, :, :-1]
    if tolerance > 0:
        changed = F.max_pool1d(changed, kernel_size=2 * tolerance + 1, stride=1, padding=tolerance)
    return changed * valid


class BoundaryLoss(nn.Module):
    """Masked binary cross-entropy on the boundary stages of a model output.

    ``pos_weight=None`` (the default) recomputes ``n_negative / n_positive``
    per batch, which is the honest weight for a channel whose positives are a
    handful of frames in a few thousand and whose density changes from trial
    to trial. Pass a number to pin it — a run logs whichever was used.

    With *focal*, the BCE is replaced by its focal form (``gamma``), the other
    remedy the ASRF paper offers for the same imbalance; the two are
    alternatives, not a stack.
    """

    def __init__(
        self,
        tolerance: int = 0,
        pos_weight: float | None = None,
        focal: bool = False,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.tolerance = int(tolerance)
        self.pos_weight = pos_weight
        self.focal = bool(focal)
        self.gamma = float(gamma)

    def forward(self, boundary: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """*boundary* is ``(S, B, 1, T)`` logits; the loss averages over stages."""
        if boundary.dim() != 4:
            raise ValueError(f"Boundary logits must be (S, B, 1, T), got {tuple(boundary.shape)}")
        target = boundary_targets(y, mask, self.tolerance)
        valid = (mask[:, :1, :] > 0).expand_as(target)
        total = boundary.new_zeros(())
        for stage in boundary:
            total = total + self._stage_loss(stage, target, valid)
        return total / boundary.shape[0]

    def _stage_loss(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        logits, target = logits[valid], target[valid]
        if logits.numel() == 0:
            return logits.new_zeros(())
        weight = self._pos_weight(target)
        if self.focal:
            prob = torch.sigmoid(logits)
            p_t = prob * target + (1 - prob) * (1 - target)
            alpha = target * weight + (1 - target)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            return (alpha * (1 - p_t).pow(self.gamma) * bce).mean()
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)

    def _pos_weight(self, target: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            return target.new_tensor(float(self.pos_weight))
        positives = target.sum()
        if positives <= 0:
            return target.new_tensor(1.0)
        return ((target.numel() - positives) / positives).detach()


def boundary_probabilities(boundary: torch.Tensor) -> torch.Tensor:
    """``(S, B, 1, T)`` boundary logits → the last stage's probabilities, ``(B, T)``."""
    return torch.sigmoid(boundary[-1][:, 0, :])


def boundary_peaks(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Indices of the local maxima of *prob* that clear *threshold*.

    ASRF's ``argrelmax``, minus its habit of always returning frame 0: the cut
    at the start of the trial is added by :func:`refine_with_boundary`, where
    it belongs, rather than pretending the model predicted it.
    """
    prob = np.asarray(prob, dtype=float)
    if prob.size < 3:
        return np.zeros(0, dtype=np.int64)
    gated = np.where(prob < float(threshold), 0.0, prob)
    interior = (gated[:-2] < gated[1:-1]) & (gated[2:] < gated[1:-1])
    return np.flatnonzero(interior) + 1


def snap_to_candidates(peaks: np.ndarray, candidates: np.ndarray, max_shift: int) -> np.ndarray:
    """The peaks within *max_shift* frames of a *candidate*, moved onto it.

    A peak with no candidate nearby is dropped, not kept where it is: the
    point of the hybrid mode is that a boundary the physics does not see is
    not a boundary. Returns sorted unique candidate indices.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    candidates = np.unique(np.asarray(candidates, dtype=np.int64))
    if peaks.size == 0 or candidates.size == 0:
        return np.zeros(0, dtype=np.int64)
    nearest = np.searchsorted(candidates, peaks)
    left = np.clip(nearest - 1, 0, candidates.size - 1)
    right = np.clip(nearest, 0, candidates.size - 1)
    take_left = np.abs(candidates[left] - peaks) <= np.abs(candidates[right] - peaks)
    chosen = np.where(take_left, candidates[left], candidates[right])
    within = np.abs(chosen - peaks) <= int(max_shift)
    return np.unique(chosen[within])


def refine_with_boundary(
    indices: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    mode: str = "predicted",
    candidates: np.ndarray | None = None,
    max_shift: int = 0,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    """Re-cut a dense prediction at boundary peaks and vote each span's class.

    *indices* is the argmax prediction ``(T,)``, *prob* the predicted boundary
    probability ``(T,)``. The peaks split the trial into spans; each span takes
    the class holding the most frames in it, ties broken by summed probability
    when *scores* ``(T, C)`` is given.

    ``mode="hybrid"`` first restricts the peaks to the *candidates* (frame
    indices of detected changepoints) within *max_shift* frames — see the
    module docstring.
    """
    if mode not in REFINEMENT_MODES:
        raise ValueError(f"Unknown boundary refinement mode {mode!r}; expected one of {REFINEMENT_MODES}")
    indices = np.asarray(indices, dtype=np.int64)
    if mode == "none":
        return indices
    peaks = boundary_peaks(prob, threshold)
    if mode == "hybrid":
        if candidates is None:
            raise ValueError(
                "boundary_refinement='hybrid' needs the changepoint candidates; turn on "
                "infer.postprocess.changepoint_correction so they are computed."
            )
        peaks = snap_to_candidates(peaks, candidates, max_shift)
    cuts = np.unique(np.concatenate([[0], peaks, [len(indices)]])).astype(np.int64)
    out = np.zeros_like(indices)
    for start, stop in zip(cuts[:-1], cuts[1:]):
        span = indices[start:stop]
        if span.size == 0:
            continue
        counts = np.bincount(span)
        modes = np.flatnonzero(counts == counts.max())
        if modes.size == 1 or scores is None:
            out[start:stop] = int(modes[0])
        else:
            out[start:stop] = int(modes[np.argmax(scores[start:stop, modes].sum(axis=0))])
    return out


def boundary_scores(gt: np.ndarray, prob: np.ndarray, threshold: float, tolerance: int) -> dict[str, float]:
    """Precision / recall / F1 of the predicted boundary peaks against *gt*'s transitions.

    *gt* is the dense ground truth ``(T,)``; a predicted peak counts as a hit
    when a true transition sits within *tolerance* frames of it, each true
    transition matching at most one peak. This scores the channel on its own
    terms — where the transitions are — with no class prediction involved.
    """
    truth = np.flatnonzero(np.diff(np.asarray(gt, dtype=np.int64)) != 0) + 1
    peaks = boundary_peaks(prob, threshold)
    if truth.size == 0 and peaks.size == 0:
        return {"boundary_precision": 100.0, "boundary_recall": 100.0, "boundary_f1": 100.0}
    hit = np.zeros(truth.size, dtype=bool)
    tp = 0
    for peak in peaks:
        if truth.size == 0:
            break
        distance = np.abs(truth - peak).astype(np.int64)
        distance[hit] = np.iinfo(np.int64).max
        best = int(np.argmin(distance))
        if distance[best] <= tolerance:
            hit[best] = True
            tp += 1
    fp = peaks.size - tp
    fn = truth.size - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "boundary_precision": 100.0 * precision,
        "boundary_recall": 100.0 * recall,
        "boundary_f1": 100.0 * f1,
    }
