"""``asrf`` — any vendored backbone plus a class-agnostic boundary branch.

ASRF (`arXiv:2007.06866 <https://arxiv.org/abs/2007.06866>`_) is not an
architecture so much as a second head: the trunk features feed both the usual
classifier and a one-channel branch that predicts *where the transitions are*.
That is the whole difference, and it is why this is a wrapper rather than a
new network — ``model.params.backbone`` names any registered vendored
architecture and everything else about that model stays exactly as it was::

    model:
      architecture: asrf
      params:
        backbone: asformer
        backbone_params: {num_decoders: 0}   # the encoder-only baseline
        brb_stages: 1                        # ASRF's boundary refinement branch

The branch is upstream's: a 1×1 convolution off the shared trunk, optionally
refined by ``brb_stages`` single-stage dilated TCNs each taking the previous
stage's *probabilities*. Every stage is returned, so the loss supervises all
of them and inference reads the last — the same rule as the class logits.

Deliberately **not** here: boundary-weighted cross-entropy. It was ablated and
did not move F1; a head that is asked where the transition is supersedes
reweighting a head that is only ever asked what class a frame is.

Covered by ``tests/test_unit/test_segment_architectures.py``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ethograph.segment.models import ARCHITECTURES, ModelOutput, register_architecture
from ethograph.segment.models.vendored import DLC2ActionModel

DEFAULT_BACKBONE = "asformer"
"""The encoder this was designed around; ``model.params.backbone`` overrides it."""

ASRF_KEYS = frozenset({"backbone", "backbone_params", "brb_stages", "brb_layers"})
"""What ``model.params`` accepts here. A backbone's own keys go in ``backbone_params``."""


class DilatedResidualLayer(nn.Module):
    """MS-TCN's dilated residual block — the unit every stage of the branch is built from."""

    def __init__(self, dilation: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        return x + self.dropout(self.conv_in(out))


class SingleStageTCN(nn.Module):
    """One refinement stage: 1×1 in, ``n_layers`` doubling dilations, 1×1 out."""

    def __init__(self, in_channels: int, n_features: int, out_channels: int, n_layers: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, n_features, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class ASRFModel(nn.Module):
    """A DLC2Action backbone tapped at its trunk, with a boundary branch beside the classifier.

    The tap point is upstream's own split: ``extract_features`` produces the
    shared representation and ``predictor`` turns it into class logits, so the
    branch attaches without touching either.
    """

    def __init__(
        self,
        adapter: DLC2ActionModel,
        n_trunk_features: int,
        brb_stages: int = 1,
        brb_layers: int = 10,
    ) -> None:
        super().__init__()
        self.inner = adapter.inner
        self.per_sample = adapter.per_sample
        self.reverse_stages = adapter.reverse_stages
        self.conv_bound = nn.Conv1d(n_trunk_features, 1, 1)
        self.brb = nn.ModuleList([SingleStageTCN(1, n_trunk_features, 1, brb_layers) for _ in range(brb_stages - 1)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> ModelOutput:
        x = x * mask
        if self.per_sample:
            outputs = [self._run(x[i : i + 1]) for i in range(x.shape[0])]
            logits = torch.cat([o[0] for o in outputs], dim=1)
            boundary = torch.cat([o[1] for o in outputs], dim=1)
        else:
            logits, boundary = self._run(x)
        return ModelOutput(logits=logits * mask, boundary=boundary * mask)

    def _run(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.inner.extract_features(x)
        if features.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"The backbone's trunk returns {features.shape[-1]} frames for an input of {x.shape[-1]}, "
                "so a per-frame boundary branch cannot be attached to it. Use a backbone that keeps the "
                "time axis (asformer, mstcn, edtcn, mlp)."
            )
        logits = self.inner.predictor(features)
        if logits.dim() == 3:
            logits = logits.unsqueeze(0)
        if self.reverse_stages:
            logits = logits.flip(0)
        bound = self.conv_bound(features)
        stages = [bound]
        for stage in self.brb:
            bound = stage(torch.sigmoid(bound))
            stages.append(bound)
        return logits, torch.stack(stages)


@register_architecture("asrf")
def build_asrf(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """ASRF: ``backbone`` + a boundary branch, both fed by the backbone's trunk.

    ``backbone`` (default ``asformer``) and ``backbone_params`` are passed
    straight through to that architecture's own builder, so its defaults,
    its validation and its error messages are unchanged. ``brb_stages``
    (default 1) is how many boundary stages to return, ``brb_layers``
    (default 10) how deep each refinement stage is.

    Train the branch with ``train.boundary.weight``; at 0 it is built but
    contributes nothing, which is the ablation.
    """
    unknown = set(params) - ASRF_KEYS
    if unknown:
        raise ValueError(
            f"model.params {sorted(unknown)} are not settings of the asrf wrapper; it takes "
            f"{sorted(ASRF_KEYS)}. A backbone's own hyperparameters go in model.params.backbone_params."
        )
    backbone = str(params.get("backbone", DEFAULT_BACKBONE))
    if backbone == "asrf":
        raise ValueError("model.params.backbone must be a plain architecture, not 'asrf' itself.")
    if backbone not in ARCHITECTURES:
        raise ValueError(f"Unknown backbone {backbone!r}. Available: {', '.join(sorted(ARCHITECTURES))}")
    adapter = ARCHITECTURES[backbone](dict(params.get("backbone_params", {})), n_features, n_classes)
    if not isinstance(adapter, DLC2ActionModel):
        raise ValueError(
            f"Backbone {backbone!r} is not a vendored DLC2Action model, so it exposes no trunk to attach "
            "a boundary branch to."
        )
    shape = adapter.inner.features_shape()
    if shape is None:
        raise ValueError(
            f"Backbone {backbone!r} declares no trunk feature shape, so there is nothing for a boundary "
            "branch to read. Use a backbone that keeps the time axis and reports its features "
            "(asformer, mstcn, edtcn, mlp)."
        )
    n_trunk = int(shape[0])
    return ASRFModel(
        adapter,
        n_trunk,
        brb_stages=int(params.get("brb_stages", 1)),
        brb_layers=int(params.get("brb_layers", 10)),
    )
