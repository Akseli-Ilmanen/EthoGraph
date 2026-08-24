"""``baformer`` — the ASFormer encoder with a boundary-aware query-voting head.

BaFormer (`arXiv:2405.15995 <https://arxiv.org/abs/2405.15995>`_) keeps the
validated frame-wise encoder and replaces only what sits on top of it: a set
of *instance queries* that each claim a soft mask over the timeline and emit a
class, plus one **global class-agnostic boundary query** that cuts the
timeline into spans. Each span is then classified by letting the queries vote.

Structurally that is the pipeline this project already runs by hand — cut at
changepoints, classify what lies between — learned end to end. The encoder is
the vendored DLC2Action ASFormer's, layer for layer: this module reuses its
``AttModule`` and only exposes the per-layer features the decoder needs as
multi-scale keys and values.

**The dense logits mean the same thing in both modes, but are produced
differently.** In training they are the soft query composition
``softmax(class) · sigmoid(mask)``, which is what the set criterion's
gradients flow through. In eval they are the boundary-aware vote: the
timeline is cut at the peaks of the boundary query, each span takes the query
whose mask covers it best, and that query's class distribution fills the span.
Upstream does exactly this, and it is why ``model.eval()`` is not optional
here. Either way the value returned is a log-probability, so
``softmax(logits)`` is a distribution and every downstream consumer —
metrics, confidence, post-processing — is untouched.

The upstream code depends on detectron2, einops, timm and fvcore. None of
those are dependencies here; the attention is plain ``torch``.

Covered by ``tests/test_unit/test_segment_architectures.py``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ethograph.segment.dlc2action.model.asformer import AttModule
from ethograph.segment.models import ModelOutput, register_architecture
from ethograph.segment.models.vendored import upstream_defaults

LOG_EPS = 1e-8
"""Floor under the composed probabilities before taking their log."""

BAFORMER_KEYS = frozenset(
    {
        "num_layers",
        "r1",
        "r2",
        "num_f_maps",
        "channel_masking_rate",
        "num_queries",
        "num_decode",
        "nheads",
        "dropout",
        "boundary_threshold",
    }
)
"""What ``model.params`` accepts. The encoder keys mirror ``asformer``'s."""


class ASFormerTrunk(nn.Module):
    """The vendored ASFormer encoder, returning every layer's features.

    Identical to ``dlc2action.model.asformer.Encoder`` except that the
    intermediate activations are kept: the query decoder cross-attends to one
    encoder layer per decode level, which is what "multi-scale" means here.
    """

    def __init__(
        self,
        n_features: int,
        num_layers: int,
        r1: float,
        r2: float,
        num_f_maps: int,
        channel_masking_rate: float,
    ) -> None:
        super().__init__()
        self.conv_1x1 = nn.Conv1d(n_features, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2**i, num_f_maps, num_f_maps, r1, r2, "sliding_att", "encoder", 1) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate
        self.mask_features = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self.channel_masking_rate > 0:
            x = self.dropout(x.unsqueeze(2)).squeeze(2)
        feature = self.conv_1x1(x)
        per_layer = []
        for layer in self.layers:
            feature = layer(feature, None, mask)
            per_layer.append(feature)
        return per_layer, self.mask_features(feature)


class MultiHeadAttention(nn.Module):
    """Pre-norm multi-head attention with a residual; ``mask`` scales the logits."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"model.params.num_f_maps ({dim}) must be divisible by nheads ({heads}).")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.pre_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, dim * 2)
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(dim, dim)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x_q
        q = self._split(self.query(self.pre_norm(x_q)))
        k, v = (self._split(t) for t in self.key_value(x_kv).chunk(2, dim=-1))
        energy = torch.einsum("bhqc,bhkc->bhqk", q, k) * self.scale
        if mask is not None:
            energy = energy * mask
        attention = self.drop(F.softmax(energy, dim=-1))
        out = torch.einsum("bhqk,bhkc->bhqc", attention, v)
        b, h, n, c = out.shape
        return residual + self.projection(out.transpose(1, 2).reshape(b, n, h * c))


class FeedForward(nn.Module):
    """Pre-norm MLP with a residual."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.pre_norm(x))


class DecoderLayer(nn.Module):
    """Cross-attend the queries to one encoder level, then self-attend, then FFN."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attention = MultiHeadAttention(dim, heads, dropout)
        self.self_attention = MultiHeadAttention(dim, heads, dropout)
        self.ffn = FeedForward(dim, dim * 2, dropout)

    def forward(self, queries: torch.Tensor, source: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.cross_attention(queries, source, attention_mask)
        out = self.self_attention(out, out)
        return self.ffn(out)


class MLP(nn.Module):
    """The three-layer projection upstream uses for the mask and video embeddings."""

    def __init__(self, dim: int, out_dim: int, n_layers: int = 3) -> None:
        super().__init__()
        sizes = [dim] * n_layers + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


def sinusoidal_positions(n: int, dim: int) -> torch.Tensor:
    """``(1, n, dim)`` sinusoidal encoding — the queries' only sense of order."""
    position = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    angle = position / torch.pow(10000, 2 * torch.div(torch.arange(dim), 2, rounding_mode="floor") / dim)
    encoding = torch.zeros(n, dim)
    encoding[:, 0::2] = torch.sin(angle[:, 0::2])
    encoding[:, 1::2] = torch.cos(angle[:, 1::2])
    return encoding.unsqueeze(0)


class QueryDecoder(nn.Module):
    """Instance queries plus one global boundary query, refined level by level.

    Each level cross-attends the queries to one encoder layer, then re-reads
    the three heads: a class per query, a mask per query, and — by pooling the
    queries into a single video-level embedding — one boundary curve over the
    whole timeline. Every level's outputs are kept for deep supervision.
    """

    def __init__(self, dim: int, num_queries: int, num_decode: int, heads: int, dropout: float, n_classes: int):
        super().__init__()
        self.num_decode = num_decode
        self.heads = heads
        self.query = nn.Parameter(torch.randn(1, num_queries, dim))
        self.register_buffer("query_pos", sinusoidal_positions(num_queries, dim), persistent=False)
        self.input_projection = nn.ModuleList([nn.Conv1d(dim, dim, 3, padding=1) for _ in range(num_decode)])
        self.levels = nn.ModuleList([DecoderLayer(dim, heads, dropout) for _ in range(num_decode)])
        self.decoder_norm = nn.LayerNorm(dim)
        self.class_embed = nn.Linear(dim, n_classes + 1)
        self.mask_embed = MLP(dim, dim)
        self.video_embed_before = MLP(dim, dim)
        self.video_embed = nn.Linear(num_queries, 1)

    def forward(
        self, per_layer: list[torch.Tensor], mask_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(per_layer) < self.num_decode:
            raise ValueError(
                f"The decoder has {self.num_decode} levels but the encoder gives {len(per_layer)} layers; "
                "model.params.num_decode must not exceed model.params.num_layers."
            )
        sources = [self.input_projection[i](per_layer[i]).transpose(-1, -2) for i in range(self.num_decode)]
        output = self.query + self.query_pos
        output = output.expand(mask_features.shape[0], -1, -1)
        classes, masks, boundaries = [], [], []
        cls, msk, bnd = self._heads(output, mask_features)
        classes.append(cls)
        masks.append(msk)
        boundaries.append(bnd)
        for i, level in enumerate(self.levels):
            attention_mask = msk.sigmoid().unsqueeze(1).repeat(1, self.heads, 1, 1)
            output = level(output, sources[i], attention_mask)
            cls, msk, bnd = self._heads(output, mask_features)
            classes.append(cls)
            masks.append(msk)
            boundaries.append(bnd)
        return torch.stack(classes), torch.stack(masks), torch.stack(boundaries)

    def _heads(
        self, output: torch.Tensor, mask_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normed = self.decoder_norm(output)
        cls = self.class_embed(normed)
        msk = torch.einsum("bqc,bcl->bql", self.mask_embed(normed), mask_features)
        pooled = self.video_embed(self.video_embed_before(output).transpose(-1, -2)).transpose(-1, -2)
        bnd = torch.einsum("bqc,bcl->bql", pooled, mask_features)
        return cls, msk, bnd


class BaFormerModel(nn.Module):
    """ASFormer trunk + :class:`QueryDecoder`, under the registry contract."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        num_layers: int,
        r1: float,
        r2: float,
        num_f_maps: int,
        channel_masking_rate: float,
        num_queries: int,
        num_decode: int,
        nheads: int,
        dropout: float,
        boundary_threshold: float,
    ) -> None:
        super().__init__()
        self.trunk = ASFormerTrunk(n_features, num_layers, r1, r2, num_f_maps, channel_masking_rate)
        self.decoder = QueryDecoder(num_f_maps, num_queries, num_decode, nheads, dropout, n_classes)
        self.n_classes = n_classes
        self.boundary_threshold = float(boundary_threshold)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> ModelOutput:
        x = x * mask
        per_layer, mask_features = self.trunk(x, mask)
        mask_features = mask_features * mask
        query_logits, query_masks, boundary = self.decoder(per_layer, mask_features)
        boundary = boundary * mask.unsqueeze(0)
        dense = self._compose(query_logits, query_masks, boundary, mask)
        return ModelOutput(
            logits=dense * mask,
            boundary=boundary,
            query_logits=query_logits,
            query_masks=query_masks,
        )

    def _compose(
        self,
        query_logits: torch.Tensor,
        query_masks: torch.Tensor,
        boundary: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(L, B, Q, ·)`` heads → dense ``(L, B, C, T)`` log-probabilities."""
        probabilities = query_logits.softmax(dim=-1)[..., :-1]
        masks = query_masks.sigmoid() * mask.unsqueeze(0)
        if not self.training:
            masks = self._vote(masks, boundary)
        dense = torch.einsum("lbqc,lbqt->lbct", probabilities, masks)
        return torch.log(dense.clamp_min(LOG_EPS))

    def _vote(self, masks: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """Harden the query masks into one winner per boundary-delimited span.

        This is the "query voting" of the paper's title, and it is why the
        model has to be in eval mode to be evaluated: a soft composition
        averages the queries and blurs precisely the edges F1@90 measures.
        """
        from ethograph.segment.boundary import boundary_peaks

        hardened = torch.zeros_like(masks)
        probabilities = torch.sigmoid(boundary).detach().cpu().numpy()
        for level in range(masks.shape[0]):
            for sample in range(masks.shape[1]):
                peaks = boundary_peaks(probabilities[level, sample, 0], self.boundary_threshold)
                cuts = [0, *peaks.tolist(), masks.shape[-1]]
                for start, stop in zip(cuts[:-1], cuts[1:]):
                    if stop <= start:
                        continue
                    winner = int(masks[level, sample, :, start:stop].sum(dim=-1).argmax())
                    hardened[level, sample, winner, start:stop] = 1.0
        return hardened


@register_architecture("baformer")
def build_baformer(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """BaFormer: the ASFormer encoder under a boundary-aware query-voting head.

    The encoder keys (``num_layers``, ``r1``, ``r2``, ``num_f_maps``,
    ``channel_masking_rate``) default to the vendored ASFormer's own YAML, so
    the trunk is the validated one unless you say otherwise. The head's own
    settings are ``num_queries`` (upstream's 100 — raise it towards twice the
    worst trial's segment count, which the set criterion will tell you),
    ``num_decode`` (defaults to one level per encoder layer), ``nheads``,
    ``dropout`` and ``boundary_threshold``.
    """
    unknown = set(params) - BAFORMER_KEYS
    if unknown:
        raise ValueError(f"model.params {sorted(unknown)} are not baformer settings; it takes {sorted(BAFORMER_KEYS)}.")
    encoder_defaults = upstream_defaults("asformer", n_features)
    num_layers = int(params.get("num_layers", encoder_defaults["num_layers"]))
    num_f_maps = int(params.get("num_f_maps", encoder_defaults["num_f_maps"]))
    return BaFormerModel(
        n_features=n_features,
        n_classes=n_classes,
        num_layers=num_layers,
        r1=float(params.get("r1", encoder_defaults["r1"])),
        r2=float(params.get("r2", encoder_defaults["r2"])),
        num_f_maps=num_f_maps,
        channel_masking_rate=float(params.get("channel_masking_rate", encoder_defaults["channel_masking_rate"])),
        num_queries=int(params.get("num_queries", 100)),
        num_decode=int(params.get("num_decode", num_layers)),
        nheads=int(params.get("nheads", 4)),
        dropout=float(params.get("dropout", 0.1)),
        boundary_threshold=float(params.get("boundary_threshold", 0.3)),
    )
