"""Architecture registry for the segmentation pipeline.

An **architecture** is a registered builder that turns a parameter dict into a
``torch.nn.Module`` for any input layout and class count. Every built model
speaks one contract::

    logits = model(x, mask)
    # x:      (B, F, T) float   — F feature columns over T frames
    # mask:   (B, 1, T) float   — 1 where a frame is real, 0 where padded
    # logits: (S, B, C, T) float — S ≥ 1 stages (MS-TCN-style refinement
    #                              returns several; most return one)

The training loss averages over stages; **inference reads the last stage**, so
a builder whose upstream model emits its stages the other way round must flip
them (see ``DLC2ActionModel(reverse_stages=True)`` for the C2F U-Nets).

A builder may return the bare ``(S, B, C, T)`` tensor or a
:class:`ModelOutput` wrapping it; every consumer reads
``as_output(model(x, mask)).logits``, so the two are interchangeable.

Every vendored architecture is a DLC2Action model — see
``dlc2action/NOTICE.md``.

Register a builder with :func:`register_architecture`; third-party packages
can also expose builders through the ``ethograph.segment.architectures``
entry-point group (``name = "pkg.module:builder"``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

Builder = Callable[[dict[str, Any], int, int], nn.Module]


@dataclass(frozen=True)
class ModelOutput:
    """One forward pass' class logits.

    ``logits`` is always ``(S, B, C, T)`` with the last stage the prediction —
    the contract every consumer reads.
    """

    logits: torch.Tensor


def as_output(result: torch.Tensor | ModelOutput) -> ModelOutput:
    """Normalise what a model returned to a :class:`ModelOutput`.

    A builder may return the bare ``(S, B, C, T)`` tensor, so this is the one
    place that difference is absorbed.
    """
    return result if isinstance(result, ModelOutput) else ModelOutput(logits=result)

ARCHITECTURES: dict[str, Builder] = {}

#: Registry name → the YAML its defaults are read from, for the architectures
#: that are ours (``rnn``). A vendored one reads upstream's own config instead.
DEFAULTS_FILES: dict[str, Path] = {}

ENTRY_POINT_GROUP = "ethograph.segment.architectures"


def register_architecture(name: str, defaults: Path | None = None) -> Callable[[Builder], Builder]:
    """Decorator: register ``builder(params, n_features, n_classes) -> nn.Module`` under *name*.

    *defaults* names the YAML an architecture of ours reads its hyperparameters
    from, so ``tunable_params`` can list them; a vendored architecture leaves
    it unset and is read off upstream's config.
    """

    def _register(builder: Builder) -> Builder:
        if name in ARCHITECTURES:
            raise ValueError(f"Architecture {name!r} is already registered.")
        ARCHITECTURES[name] = builder
        if defaults is not None:
            DEFAULTS_FILES[name] = defaults
        return builder

    return _register


def _load_builtin() -> None:
    # Imported for their registration side effect only.
    from ethograph.segment.models import rnn, skeleton_graph, vendored  # noqa: F401


def _load_entry_points() -> None:
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in ARCHITECTURES:
            continue
        try:
            ARCHITECTURES[ep.name] = ep.load()
        except Exception as exc:  # a broken third-party plugin must not take the registry down
            logger.warning("Could not load architecture plugin %r: %s", ep.name, exc)


def available_architectures() -> list[str]:
    _load_builtin()
    _load_entry_points()
    return sorted(ARCHITECTURES)


def build_model(architecture: str, params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """Build a model under the registry contract; unknown names list what exists."""
    names = available_architectures()
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}. Available: {', '.join(names)}")
    return ARCHITECTURES[architecture](dict(params), int(n_features), int(n_classes))
