"""Recurrent baseline — the cheapest temporal model in the registry.

One bidirectional GRU or LSTM stack over the whole trial and a linear read-out
per frame. torch's own ``nn.GRU`` / ``nn.LSTM``; nothing vendored. It sits
between ``mlp`` (per-frame, no temporal context) and the TCN / transformer
architectures, and trains on a CPU, which is what makes it a baseline the
heavier models have to beat.

Padding never enters the recurrence: the batch is packed by each sample's real
length (``mask.sum``), so the backward direction of a shorter trial starts at
its last real frame, not at its zero tail. Defaults from ``config/rnn.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ethograph.segment.models import register_architecture

RNN_DEFAULTS = Path(__file__).parent / "config" / "rnn.yaml"

CELLS: dict[str, type[nn.RNNBase]] = {"gru": nn.GRU, "lstm": nn.LSTM}


class RecurrentSegmenter(nn.Module):
    """``(B, F, T)`` + mask → ``(1, B, C, T)``: one recurrent stack, one linear head."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        cell: str,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"model.params.cell must be one of {sorted(CELLS)}, not {cell!r}.")
        self.rnn = CELLS[cell](
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_frames = x.shape[-1]
        lengths = mask[:, 0].sum(dim=1).long().cpu()
        packed = pack_padded_sequence((x * mask).transpose(1, 2), lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = self.rnn(packed)
        padded, _ = pad_packed_sequence(hidden, batch_first=True, total_length=n_frames)
        logits = self.head(self.drop(padded)).transpose(1, 2)
        return (logits * mask).unsqueeze(0)


def _defaults() -> dict[str, Any]:
    return yaml.safe_load(RNN_DEFAULTS.read_text(encoding="utf-8")) or {}


@register_architecture("rnn", defaults=RNN_DEFAULTS)
def build_rnn(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """Bidirectional GRU/LSTM over the trial. Defaults from ``config/rnn.yaml``; returns ``S = 1``.

    A key the model does not take is refused here, naming the ones it does —
    the same contract the vendored builders give.
    """
    defaults = _defaults()
    unknown = set(params) - set(defaults)
    if unknown:
        raise ValueError(
            f"model.params {sorted(unknown)} are not hyperparameters of this architecture (rnn); "
            f"it takes {sorted(defaults)}. See {RNN_DEFAULTS.name}, or eto.segment.tunable_params('rnn')."
        )
    return RecurrentSegmenter(n_features, n_classes, **{**defaults, **params})
