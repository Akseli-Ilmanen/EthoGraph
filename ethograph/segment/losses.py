"""The training loss: DLC2Action's own :class:`MS_TCN_Loss`, plus the circle term.

:func:`build_objective` is what training calls. It composes the two terms a
run can have, each weighted by the config and each reported separately so a
metrics row says where the loss went:

* the **frame** loss below (``train.frame_weight``) — upstream's, near enough
  unmodified;
* the **circle** loss (``train.circle.weight``) — a deep metric-learning term
  over the finest-stage logits, see :class:`CircleLoss`. Architecture-agnostic,
  since every registered model produces logits.

The frame loss itself is cross-entropy (optionally focal) plus upstream's consistency term — the
truncated MSE between consecutive log-probabilities, weighted by ``alpha`` —
averaged over the model's stages. Nothing here reimplements any of that. This
module only reads upstream's ``config/losses.yaml`` and fills in the two
things a config file cannot know:

* ``exclusive`` — upstream takes it from the task's single- vs multi-label
  problem type, which lives in the toolbox layer we do not vendor. Ours are
  single-label.
* ``weights`` — upstream's YAML says ``dataset_inverse_weights``, a sentinel
  its dataset layer resolves. That layer is not vendored, so the sentinel
  cannot be honoured and the default here is ``None`` (unweighted
  cross-entropy). Pass an explicit list to ``train.loss.weights`` to weight
  classes by hand.
* ``tau`` — the truncation threshold of the consistency term. Upstream writes
  it into the arithmetic (``clamp(..., max=16)``, i.e. τ = 4, MS-TCN's own
  value), so no config file can reach it. It is the second half of what makes
  that term a boundary-blurring regulariser — ``alpha`` says how much it
  counts, ``tau`` how large a log-probability jump it still penalises — and
  both were tuned in the literature at 15–30 fps. At 200 Hz they are worth
  re-tuning, so ``tau`` is exposed here beside ``alpha``.
  :class:`TruncatedMSTCNLoss` overrides that one method and nothing else;
  at the default τ it is upstream's loss to the last bit (asserted by
  ``tests/test_unit/test_segment_losses.py::TestTau``).

Padded frames need no mask: cross-entropy ignores them through
``ignore_index`` (:data:`~ethograph.segment.dataset.PAD_TARGET` is upstream's
own ``-100``), and the adapter zeroes padded logits, so a constant
log-softmax makes their contribution to the consistency term exactly zero.
Covered by ``tests/test_unit/test_segment_losses.py``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from ethograph.segment.config import DLC2ACTION_CONFIG, SegmentConfig
from ethograph.segment.dlc2action.loss import MS_TCN_Loss
from ethograph.segment.models import ModelOutput

LOSS_CONFIG = DLC2ACTION_CONFIG / "losses.yaml"
"""Upstream's ``dlc2action/config/losses.yaml`` — the defaults, verbatim."""

LOSS_KEY = "ms_tcn"
"""The block of that file our loss is built from."""

DATASET_INVERSE_WEIGHTS = "dataset_inverse_weights"
"""Upstream's sentinel for class weights computed from the dataset.

Unsupported here: resolving it is the job of DLC2Action's dataset layer,
which this project does not vendor.
"""

UPSTREAM_CLAMP = 16.0
"""What ``MS_TCN_Loss.consistency_loss`` clamps the squared log-probability difference to.

Upstream's one hard-coded number in that term, and MS-TCN's own: the
truncation ``min(delta, tau)`` is applied to the difference, so the clamp on
its square is ``tau ** 2`` — 16 for MS-TCN's tau of 4.
"""

DEFAULT_TAU = math.sqrt(UPSTREAM_CLAMP)
"""Upstream's truncation threshold, τ = 4 — the default, so nothing changes unasked."""

TAU_KEY = "tau"
"""``train.loss.tau``: ours, and the one key :class:`MS_TCN_Loss` does not take."""

#: Everything ``MS_TCN_Loss.__init__`` accepts besides ``num_classes``, plus
#: our own ``tau``, so a typo in ``train.loss`` is refused rather than
#: silently ignored.
LOSS_KEYWORDS = frozenset(
    {"weights", "exclusive", "ignore_index", "focal", "gamma", "alpha", "hard_negative_weight", TAU_KEY}
)


class TruncatedMSTCNLoss(MS_TCN_Loss):
    """Upstream's loss with the consistency term's truncation threshold exposed.

    The term penalises ``|log p[t] - log p[t - 1]|`` per class, truncated at
    *tau* so a genuine class change is not punished without limit. Everything
    else — the cross-entropy, the focal weighting, the averaging over stages —
    is inherited untouched.
    """

    def __init__(self, *, tau: float = DEFAULT_TAU, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not tau > 0:
            raise ValueError(f"train.loss.tau={tau} — the truncation threshold must be positive (upstream's is 4).")
        self.tau = float(tau)

    def consistency_loss(self, p: torch.Tensor) -> torch.Tensor:
        """Upstream's, with its ``max=16`` replaced by ``tau ** 2``."""
        mse = self.mse(self.log_nl(p[:, :, 1:]), self.log_nl(p.detach()[:, :, :-1]))
        return torch.mean(torch.clamp(mse, min=0, max=self.tau**2))


def upstream_defaults() -> dict[str, Any]:
    """DLC2Action's own loss defaults, straight from ``config/losses.yaml``."""
    if not LOSS_CONFIG.is_file():
        raise FileNotFoundError(f"No vendored DLC2Action loss config at {LOSS_CONFIG}")
    config = yaml.safe_load(LOSS_CONFIG.read_text(encoding="utf-8")) or {}
    if LOSS_KEY not in config:
        raise KeyError(f"{LOSS_CONFIG} has no {LOSS_KEY!r} block; found {sorted(config)}")
    return dict(config[LOSS_KEY])


def build_loss(overrides: dict[str, Any], n_classes: int) -> tuple[nn.Module, dict[str, Any]]:
    """Upstream's loss for this run; returns it with the resolved settings.

    *overrides* is the project config's ``train.loss`` — merged over
    :func:`upstream_defaults` key by key, exactly as ``model.params`` is
    merged over an architecture's YAML. The settings come back so the run can
    log what it actually trained with.
    """
    settings = {**upstream_defaults(), "exclusive": True, "weights": None, TAU_KEY: DEFAULT_TAU, **overrides}
    unknown = set(settings) - LOSS_KEYWORDS
    if unknown:
        raise ValueError(f"train.loss: unknown key(s) {sorted(unknown)}; MS_TCN_Loss takes {sorted(LOSS_KEYWORDS)}")
    if isinstance(settings.get("weights"), str):
        raise ValueError(
            f"train.loss.weights={settings['weights']!r} is a DLC2Action sentinel its dataset layer "
            "resolves, and that layer is not vendored here. Give an explicit list of "
            f"{n_classes} weights, or leave it unset for unweighted cross-entropy."
        )
    tau = float(settings[TAU_KEY])
    kwargs = {k: v for k, v in settings.items() if k != TAU_KEY}
    return TruncatedMSTCNLoss(num_classes=n_classes, tau=tau, **kwargs), settings


def _label_similarity_pairs(normed: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Every unordered pair's cosine similarity, split by whether the two frames share a label.

    *normed* is ``(N, C)``, already L2-normalised per row. Ported from
    CETNet's ``convert_label_to_similarity`` (``segment/archive/cetnet_encoder.py``).
    """
    similarity = (normed @ normed.transpose(1, 0)).view(-1)
    same_label = label.unsqueeze(1) == label.unsqueeze(0)
    positive = same_label.triu(diagonal=1).view(-1)
    negative = same_label.logical_not().triu(diagonal=1).view(-1)
    return similarity[positive], similarity[negative]


class CircleLoss(nn.Module):
    """Deep metric-learning loss (Sun et al., CVPR 2020, ``arXiv:2002.10857``).

    Pulls same-class pairs' cosine similarity toward ``1 - m`` and pushes
    different-class pairs' toward ``m``, weighting each pair by how far it
    already sits from that margin — so pairs the model already gets right
    contribute almost nothing and gradient goes to the pairs still confused.

    Ported unmodified from an older CETNet training script
    (github.com/Wangjhdeveloper/CETNet/blob/main/model.py), which applied it to an encoder
    trunk's L2-normalised feature map — a representation this project's
    architecture contract does not expose (every registered model returns
    only class logits, see :class:`~ethograph.segment.models.ModelOutput`).
    :func:`build_objective` instead feeds it the finest-stage logits,
    L2-normalised per frame.
    """

    def __init__(self, m: float = 0.25, gamma: float = 128.0) -> None:
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m
        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        return self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))


def circle_term(
    circle_loss: CircleLoss,
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    max_frames: int | None,
) -> torch.Tensor | None:
    """The circle loss over one batch's finest-stage logits, or ``None`` if there is nothing to compare.

    Pools every unpadded frame across the whole batch into one set of
    (logit-vector, label) pairs — same-class frames are drawn together and
    different-class frames pushed apart regardless of which sample or
    timestep they came from, exactly as :func:`_label_similarity_pairs` does
    with the trunk feature it was ported from. Returns ``None`` when the pool
    has no positive pair (every frame the same class) or no negative pair
    (every frame a different class), since :class:`CircleLoss` would
    otherwise ``logsumexp`` an empty tensor.
    """
    frame_mask = mask[:, 0, :] > 0
    embeddings = logits[-1].permute(0, 2, 1)[frame_mask]  # (N, C)
    labels = y[frame_mask]  # (N,)
    if max_frames is not None and embeddings.shape[0] > max_frames:
        keep = torch.randperm(embeddings.shape[0], device=embeddings.device)[:max_frames]
        embeddings, labels = embeddings[keep], labels[keep]
    sp, sn = _label_similarity_pairs(F.normalize(embeddings, dim=-1), labels)
    if sp.numel() == 0 or sn.numel() == 0:
        return None
    return circle_loss(sp, sn)


class Objective(nn.Module):
    """The whole training loss: frame + circle, weighted and itemised.

    ``forward`` returns ``(total, parts)`` where *parts* holds each term's own
    value as a plain float — that is what the run's ``metrics.tsv`` and log
    line report, so a loss that stops moving can be traced to the term that
    stopped moving.
    """

    def __init__(
        self,
        frame_loss: nn.Module,
        frame_weight: float = 1.0,
        circle_loss: CircleLoss | None = None,
        circle_weight: float = 0.0,
        circle_max_frames: int | None = None,
    ) -> None:
        super().__init__()
        self.frame_loss = frame_loss
        self.frame_weight = float(frame_weight)
        self.circle_loss = circle_loss
        self.circle_weight = float(circle_weight)
        self.circle_max_frames = circle_max_frames

    def forward(
        self, output: ModelOutput, y: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = output.logits.new_zeros(())
        parts: dict[str, float] = {}
        if self.frame_weight:
            frame = self.frame_loss(output.logits, y)
            total = total + self.frame_weight * frame
            parts["frame"] = float(frame.detach())
        if self.circle_weight:
            assert self.circle_loss is not None
            circle = circle_term(self.circle_loss, output.logits, y, mask, self.circle_max_frames)
            if circle is not None:
                total = total + self.circle_weight * circle
                parts["circle"] = float(circle.detach())
        if not parts:
            raise ValueError(
                "Every loss term is switched off (train.frame_weight=0 and train.circle.weight=0) — "
                "there is nothing to train on."
            )
        parts["total"] = float(total.detach())
        return total, parts


def build_objective(config: SegmentConfig, n_classes: int) -> tuple[Objective, dict[str, Any]]:
    """The objective this run trains against, with the settings it resolved to."""
    tcfg = config.train
    frame_loss, frame_settings = build_loss(tcfg.loss, n_classes)
    ccfg = tcfg.circle
    circle_loss = CircleLoss(m=ccfg.m, gamma=ccfg.gamma) if ccfg.weight else None
    objective = Objective(
        frame_loss=frame_loss,
        frame_weight=tcfg.frame_weight,
        circle_loss=circle_loss,
        circle_weight=ccfg.weight,
        circle_max_frames=ccfg.max_frames,
    )
    settings = {
        "frame_weight": tcfg.frame_weight,
        "frame": frame_settings,
        "circle": {"weight": ccfg.weight, "m": ccfg.m, "gamma": ccfg.gamma, "max_frames": ccfg.max_frames},
    }
    return objective, settings
