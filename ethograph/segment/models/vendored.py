"""DLC2Action architectures under the registry contract.

The upstream classes live in :mod:`ethograph.segment.dlc2action.model`
(AGPL-3.0-or-later, see that package's ``NOTICE.md``). Their ``forward(x, ssl_xs)``
returns ``(prediction, ssl_out)`` where the prediction is ``(B, C, T)`` or
``(S, B, C, T)``; :class:`DLC2ActionModel` turns that into
``model(x, mask) -> (S, B, C, T)`` with padded frames zeroed on both sides.

**Hyperparameter defaults are never written here.** They are read from the
vendored copies of upstream's own ``config/model/*.yaml``
(:func:`upstream_defaults`), so refreshing the vendor updates code and
defaults together and the two cannot drift. A builder adds a keyword of its
own only where the YAML cannot carry it — ``exclusive`` for MS-TCN, which
upstream fills in from the task's single- vs multi-label problem type rather
than from a config file. ``params`` from the project config override
everything, key by key, and reach the upstream constructor unchanged, so any
upstream keyword is settable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from ethograph.segment.dlc2action.model.asformer import ASFormer
from ethograph.segment.dlc2action.model.base_model import Model
from ethograph.segment.dlc2action.model.c2f_tcn import C2F_TCN
from ethograph.segment.dlc2action.model.c2f_transformer import C2F_Transformer
from ethograph.segment.dlc2action.model.edtcn import EDTCN
from ethograph.segment.dlc2action.model.mlp import MLP
from ethograph.segment.dlc2action.model.ms_tcn import MS_TCN3
from ethograph.segment.models import register_architecture

FEATURE_KEY = "features"

CONFIG_DIR = Path(__file__).parent.parent / "dlc2action" / "config" / "model"
"""Upstream's ``dlc2action/config/model/`` — one YAML of defaults per model."""

DATASET_FEATURES = "dataset_features"
"""Upstream's sentinel for "fill this in from the dataset's feature count"."""

C2F_MIN_FRAMES = 384
"""C2F models pool the time axis six times (T // 64) and then max-pool that with kernel 6."""

_STEMS = {
    "mstcn": "ms_tcn3",
    "asformer": "asformer",
    "c2f_tcn": "c2f_tcn",
    "c2f_transformer": "c2f_transformer",
    "edtcn": "edtcn",
    "mlp": "mlp",
}
"""Registry name → upstream's file name. They differ only for ``mstcn``."""


def upstream_defaults(stem: str, n_features: int) -> dict[str, Any]:
    """DLC2Action's own defaults for one model, ready to pass to its constructor.

    *stem* is the upstream file name (``ms_tcn3``, not our registry name
    ``mstcn``). Every value equal to :data:`DATASET_FEATURES` becomes this
    session's feature layout — the sentinel is matched by *value*, because
    upstream spells the key ``dims``, ``input_dim`` and ``input_dims`` in
    different files.
    """
    path = CONFIG_DIR / f"{stem}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No vendored DLC2Action config for {stem!r} at {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {k: _dims(n_features) if v == DATASET_FEATURES else v for k, v in config.items()}


class DLC2ActionModel(nn.Module):
    """Adapt a DLC2Action ``Model`` to ``forward(x, mask) -> (S, B, C, T)``.

    ``per_sample`` runs the batch one sample at a time and concatenates the
    results — for inner models whose batching is only correct at ``B == 1``.

    ``reverse_stages`` flips the stage axis. The registry contract is that the
    *last* stage is the prediction to read, which holds for the MS-TCN lineage
    (each stage refines the one before) but is upside down for the C2F U-Nets:
    they emit ``[full-res, T/2↑, T/4↑, T/8↑]``, finest **first**. Flipping here
    fixes every consumer at once rather than teaching each one the exception.
    """

    def __init__(self, inner: Model, per_sample: bool = False, reverse_stages: bool = False) -> None:
        super().__init__()
        self.inner = inner
        self.per_sample = per_sample
        self.reverse_stages = reverse_stages

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask
        if self.per_sample:
            logits = torch.cat([self._run(x[i : i + 1]) for i in range(x.shape[0])], dim=1)
        else:
            logits = self._run(x)
        return logits * mask

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.inner(x, [])
        if logits.dim() == 3:
            logits = logits.unsqueeze(0)
        if self.reverse_stages:
            logits = logits.flip(0)
        return logits


def _dims(n_features: int) -> dict[str, tuple[int]]:
    return {FEATURE_KEY: (n_features,)}


def tunable_params(architecture: str) -> dict[str, Any]:
    """What ``model.params`` accepts for *architecture*, and each key's default.

    Read straight off upstream's own ``config/model/{stem}.yaml``, minus the
    keys filled in from the dataset (the feature count, the segment length).
    This is what tells ``mlp`` (``f_maps_list``, ``dropout_rates``) apart from
    ``mstcn`` (``num_f_maps``, ``num_layers_R``, …) — the two do not share a
    single hyperparameter name, so a search space is per-architecture.
    """
    stem = _STEMS.get(architecture)
    if stem is None:
        raise ValueError(f"Unknown architecture {architecture!r}. Available: {', '.join(sorted(_STEMS))}")
    defaults = yaml.safe_load((CONFIG_DIR / f"{stem}.yaml").read_text(encoding="utf-8")) or {}
    return {k: v for k, v in defaults.items() if not (isinstance(v, str) and v.startswith("dataset_"))}


def _kwargs(stem: str, n_features: int, params: dict[str, Any], **supplied: Any) -> dict[str, Any]:
    """Upstream's defaults for *stem*, overridden key by key by *params*.

    A key the model does not take is an error **here**, naming the ones it
    does — the same contract the config parser gives every other section.
    Left to the constructor it would be a bare ``TypeError`` raised after the
    dataset was materialised and the run directory created, which in a search
    means a study dying on some later trial.
    """
    defaults = upstream_defaults(stem, n_features)
    unknown = set(params) - set(defaults) - set(supplied)
    if unknown:
        raise ValueError(
            f"model.params {sorted(unknown)} are not hyperparameters of this architecture "
            f"({stem}); it takes {sorted(set(defaults) | set(supplied))}. "
            f"See ethograph/segment/dlc2action/config/model/{stem}.yaml, or "
            "eto.segment.tunable_params(architecture)."
        )
    return {**defaults, **supplied, **params}


@register_architecture("mstcn")
def build_mstcn(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """MS-TCN++ (``MS_TCN3``): a dilated prediction stage refined ``num_R`` times.

    Defaults from ``config/ms_tcn3.yaml``. Returns ``S = num_R + 1`` stages;
    any batch size and any ``T >= 1``. ``exclusive`` is the one keyword the
    YAML does not carry — upstream takes it from the task's problem type.
    ``True`` passes a softmax between stages (single-label), ``False`` a
    sigmoid (multi-label); ours are single-label, and it is overridable.
    """
    kwargs = _kwargs("ms_tcn3", n_features, params, exclusive=True)
    return DLC2ActionModel(MS_TCN3(num_classes=n_classes, **kwargs))


@register_architecture("asformer")
def build_asformer(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """ASFormer: sliding-window attention encoder plus ``num_decoders`` refining decoders.

    Defaults from ``config/asformer.yaml``.
    Returns ``S = num_decoders + 1`` stages. Upstream's sliding-window attention
    does not assert on the batch size, but it lays queries out batch-major and
    keys block-major, so for ``B > 1`` samples would be cross-wired (the
    original trains at batch size 1). The adapter therefore runs samples one at
    a time (``per_sample=True``) — correct for any batch, at the cost of a
    Python loop over ``B``. Any ``T >= 1``; the window length of the deepest
    layer is ``2 ** (num_layers - 1)`` frames, shorter inputs are zero-padded.
    A frame whose features sum to exactly zero is treated as padding upstream.
    """
    kwargs = _kwargs("asformer", n_features, params)
    return DLC2ActionModel(ASFormer(num_classes=n_classes, **kwargs), per_sample=True)


@register_architecture("c2f_tcn")
def build_c2f_tcn(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """C2F-TCN: a U-Net over time whose four finest decoder outputs are the stages.

    Defaults from ``config/c2f_tcn.yaml``.
    Returns ``S = 4`` stages, ordered coarsest (``T/8`` upsampled) to finest
    (full resolution), so ``logits[-1]`` is the prediction like every other
    architecture — upstream emits them the other way round.

    Needs ``T >= C2F_MIN_FRAMES`` (384): six 2x poolings followed by a 6-wide
    pooling; upstream recommends ``T >= 512``. Uses BatchNorm, so train with
    ``B * T // 64 > 1``. Any batch size.
    """
    kwargs = _kwargs("c2f_tcn", n_features, params)
    return DLC2ActionModel(
        C2F_TCN(num_classes=n_classes, **kwargs),
        reverse_stages=True,
    )


@register_architecture("c2f_transformer")
def build_c2f_transformer(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """C2F-Transformer: C2F-TCN with multi-head self-attention before each upsampling.

    Defaults from ``config/c2f_transformer.yaml``.
    Returns ``S = 4`` stages, coarsest to finest like ``c2f_tcn``. Same
    ``T >= C2F_MIN_FRAMES`` (384) and BatchNorm constraints; ``num_f_maps``
    must be divisible by ``heads``. The bottleneck positional encoding covers
    512 positions, so ``T <= 512 * 64``. Any batch size.
    """
    kwargs = _kwargs("c2f_transformer", n_features, params)
    return DLC2ActionModel(
        C2F_Transformer(num_classes=n_classes, **kwargs),
        reverse_stages=True,
    )


@register_architecture("edtcn")
def build_edtcn(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """ED-TCN: two-level encoder-decoder with wide kernels and normalised ReLU.

    Defaults from ``config/edtcn.yaml``.
    Returns ``S = 1``. Two 2x poolings, so ``T >= 4``. Any batch size.
    """
    kwargs = _kwargs("edtcn", n_features, params)
    return DLC2ActionModel(EDTCN(num_classes=n_classes, **kwargs))


@register_architecture("mlp")
def build_mlp(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """Per-frame MLP (1x1 convolutions) — no temporal context.

    Defaults from ``config/mlp.yaml``. Returns ``S = 1``; any ``T`` and batch
    size. ``dropout_rates`` may be one float or one per hidden layer; a float
    is expanded here rather than upstream, which sizes that list by the
    *feature* count and so needs at least as many features as hidden layers.
    """
    kwargs = _kwargs("mlp", n_features, params)
    kwargs["f_maps_list"] = list(kwargs["f_maps_list"])
    rates = kwargs.get("dropout_rates")
    if rates is not None and not isinstance(rates, list):
        kwargs["dropout_rates"] = [float(rates)] * len(kwargs["f_maps_list"])
    return DLC2ActionModel(MLP(num_classes=n_classes, **kwargs))
