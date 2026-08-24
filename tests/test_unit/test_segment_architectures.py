"""Every vendored DLC2Action architecture honours the registry contract.

``model(x: (B, F, T), mask: (B, 1, T)) -> (S, B, C, T)`` with S >= 1, finite
values and zeros on padded frames.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import ethograph as eto  # noqa: E402
import ethograph.segment.models.vendored  # noqa: E402, F401
from ethograph.segment.models import ARCHITECTURES  # noqa: E402
from ethograph.segment.models.vendored import C2F_MIN_FRAMES  # noqa: E402

VENDORED = ["mstcn", "asformer", "c2f_tcn", "c2f_transformer", "edtcn", "mlp"]
HEADS = ["asrf", "baformer"]
"""Ours: an extra head over a vendored encoder, returning a ``ModelOutput``."""
N_FEATURES = 7
N_CLASSES = 3
B = 2
T = 1024
N_PADDED = 100


def _inputs(n_frames: int = T) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(B, N_FEATURES, n_frames)
    mask = torch.ones(B, 1, n_frames)
    mask[1, :, -N_PADDED:] = 0
    return x, mask


def test_registry_contains_vendored_names() -> None:
    assert set(VENDORED) <= set(ARCHITECTURES)


@pytest.mark.parametrize("name", VENDORED)
def test_forward_contract(name: str) -> None:
    model = ARCHITECTURES[name]({}, N_FEATURES, N_CLASSES)
    x, mask = _inputs()
    model.train()
    out = model(x, mask)
    assert out.dim() == 4
    assert out.shape[0] >= 1
    assert tuple(out.shape[1:]) == (B, N_CLASSES, T)
    assert torch.isfinite(out).all()
    assert (out[:, 1, :, -N_PADDED:] == 0).all()
    assert (out[:, :, :, : T - N_PADDED] != 0).any()
    out.mean().backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", VENDORED)
def test_eval_is_deterministic(name: str) -> None:
    model = ARCHITECTURES[name]({}, N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs()
    with torch.no_grad():
        first = model(x, mask)
        second = model(x, mask)
    assert torch.equal(first, second)


def test_asformer_batch_matches_single_sample() -> None:
    model = ARCHITECTURES["asformer"]({}, N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs()
    with torch.no_grad():
        batched = model(x, mask)
        singles = [model(x[i : i + 1], mask[i : i + 1]) for i in range(B)]
    assert torch.allclose(batched, torch.cat(singles, dim=1), atol=1e-6)


def test_params_override_defaults() -> None:
    model = ARCHITECTURES["mstcn"]({"num_R": 1}, N_FEATURES, N_CLASSES)
    x, mask = _inputs()
    assert model(x, mask).shape[0] == 2


@pytest.mark.parametrize("name", ["c2f_tcn", "c2f_transformer"])
def test_c2f_minimum_frames(name: str) -> None:
    model = ARCHITECTURES[name]({}, N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs(C2F_MIN_FRAMES)
    with torch.no_grad():
        assert tuple(model(x, mask).shape[1:]) == (B, N_CLASSES, C2F_MIN_FRAMES)
        x, mask = _inputs(C2F_MIN_FRAMES - 64)
        with pytest.raises(RuntimeError):
            model(x, mask)


#: Registry name -> the upstream config file the builder reads its defaults from.
CONFIG_STEMS = {
    "mstcn": "ms_tcn3",
    "asformer": "asformer",
    "c2f_tcn": "c2f_tcn",
    "c2f_transformer": "c2f_transformer",
    "edtcn": "edtcn",
    "mlp": "mlp",
}


@pytest.mark.parametrize("name,stem", sorted(CONFIG_STEMS.items()))
def test_defaults_come_from_the_vendored_upstream_yaml(name: str, stem: str) -> None:
    """No hyperparameter default is written in ``vendored.py``.

    They live in the vendored copies of upstream's own ``config/model``, so a
    vendor refresh moves code and defaults together and the two cannot drift.
    """
    import yaml

    from ethograph.segment.models.vendored import CONFIG_DIR, DATASET_FEATURES, upstream_defaults

    raw = yaml.safe_load((CONFIG_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))
    resolved = upstream_defaults(stem, N_FEATURES)
    assert set(resolved) == set(raw), "a key was invented or dropped on the way out of the YAML"
    for key, value in raw.items():
        if value == DATASET_FEATURES:
            assert resolved[key] == {"features": (N_FEATURES,)}
        else:
            assert resolved[key] == value, f"{stem}.{key} was overridden instead of passed through"
    assert DATASET_FEATURES not in resolved.values()


def test_upstream_defaults_refuses_an_unknown_model() -> None:
    from ethograph.segment.models.vendored import upstream_defaults

    with pytest.raises(FileNotFoundError, match="No vendored DLC2Action config"):
        upstream_defaults("not_a_model", N_FEATURES)


def test_every_vendored_architecture_maps_to_an_upstream_config() -> None:
    """The vendored ones read their defaults from upstream's YAML; the heads do not.

    ``asrf`` and ``baformer`` are ours — a wrapper and a head over one of
    those encoders — so they have no upstream config file of their own and are
    the only registered names outside ``_STEMS``.
    """
    from ethograph.segment.models import available_architectures
    from ethograph.segment.models.vendored import _STEMS

    assert _STEMS == CONFIG_STEMS
    assert sorted(set(_STEMS) | set(HEADS)) == available_architectures()


@pytest.mark.parametrize("name", sorted(CONFIG_STEMS))
def test_tunable_params_are_exactly_what_build_model_accepts(name: str) -> None:
    """What a search space may name, per architecture.

    The architectures share almost no hyperparameter names, so a benchmark
    over several of them needs this to build one space per architecture.
    """
    from ethograph.segment.models import build_model

    tunable = eto.segment.tunable_params(name)
    assert tunable, name
    assert not any(isinstance(v, str) and v.startswith("dataset_") for v in tunable.values()), (
        "a dataset-derived key is not a hyperparameter the user sets"
    )
    for key, default in tunable.items():
        build_model(name, {key: default}, N_FEATURES, N_CLASSES)


def test_a_param_the_architecture_does_not_take_is_refused_before_training() -> None:
    """Left to the constructor this is a bare ``TypeError``, raised after the
    dataset is materialised and the run directory created — which in a search
    kills the study on some later trial."""
    from ethograph.segment.models import build_model

    with pytest.raises(ValueError, match="not hyperparameters of this architecture"):
        build_model("mlp", {"num_f_maps": 64}, N_FEATURES, N_CLASSES)
    # the message names what it does take
    with pytest.raises(ValueError, match="f_maps_list"):
        build_model("mlp", {"num_f_maps": 64}, N_FEATURES, N_CLASSES)
    # a keyword the builder supplies itself is still settable
    build_model("mstcn", {"exclusive": False}, N_FEATURES, N_CLASSES)


def test_params_override_the_upstream_defaults() -> None:
    from ethograph.segment.models import build_model

    model = build_model("mlp", {"f_maps_list": [8]}, N_FEATURES, N_CLASSES)
    x, mask = _inputs()
    assert tuple(model(x, mask).shape) == (1, B, N_CLASSES, T)
    assert [layer.out_channels for layer in model.inner.feature_extractor.layers] == [8, N_CLASSES]


@pytest.mark.parametrize("name", ["c2f_tcn", "c2f_transformer"])
def test_c2f_puts_its_full_resolution_stage_last(name: str) -> None:
    """The contract is that ``logits[-1]`` is the prediction to read.

    C2F emits ``[full-res, T/2, T/4, T/8]`` — finest first — so the adapter
    flips it. Read the wrong end and inference silently gets an 8x-coarse map
    that cannot place a boundary: it is smooth to the point of never changing
    its argmax. Guard on that smoothness, which holds at initialisation.
    """
    from ethograph.segment.models import build_model

    model = build_model(name, {}, N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs(C2F_MIN_FRAMES * 4)
    with torch.no_grad():
        logits = model(x, mask)
    step = [(s[0, :, 1:] - s[0, :, :-1]).abs().mean().item() for s in logits]
    assert step[-1] == max(step), f"stage -1 is not the finest: per-stage mean|Δt| = {step}"
    assert step[-1] > 10 * step[0], f"stages are not ordered coarse -> fine: {step}"


def test_build_model_dispatches_vendored() -> None:
    from ethograph.segment.models import available_architectures, build_model

    assert set(VENDORED) <= set(available_architectures())
    model = build_model("mlp", {}, N_FEATURES, N_CLASSES)
    x, mask = _inputs()
    assert tuple(model(x, mask).shape) == (1, B, N_CLASSES, T)


# ---------------------------------------------------------------------------
# The heads: asrf and baformer
# ---------------------------------------------------------------------------

HEAD_T = 512
"""Shorter than the vendored sweep: the query head is O(Q x T) per level."""

BAFORMER_PARAMS = {"num_layers": 3, "num_decode": 3, "num_queries": 16, "num_f_maps": 32, "nheads": 4}
"""A small but complete BaFormer — every level, every head, few enough to be quick."""


def _head_inputs(n_frames: int = HEAD_T) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(B, N_FEATURES, n_frames)
    mask = torch.ones(B, 1, n_frames)
    mask[1, :, -N_PADDED:] = 0
    return x, mask


def _build_head(name: str):
    from ethograph.segment.models import build_model

    params = BAFORMER_PARAMS if name == "baformer" else {"backbone_params": {"num_decoders": 0, "num_layers": 4}}
    return build_model(name, params, N_FEATURES, N_CLASSES)


@pytest.mark.parametrize("name", HEADS)
def test_head_honours_the_logits_contract(name: str) -> None:
    """A ``ModelOutput`` still carries ``(S, B, C, T)`` logits, zeroed on padding.

    Everything downstream of the model reads ``.logits`` and nothing else, so
    a head that broke this would break metrics, confidence and inference at
    once.
    """
    from ethograph.segment.models import ModelOutput, as_output

    model = _build_head(name)
    x, mask = _head_inputs()
    out = model(x, mask)
    assert isinstance(out, ModelOutput)
    assert as_output(out) is out
    assert out.logits.dim() == 4
    assert tuple(out.logits.shape[1:]) == (B, N_CLASSES, HEAD_T)
    assert torch.isfinite(out.logits).all()
    assert (out.logits[:, 1, :, -N_PADDED:] == 0).all()


@pytest.mark.parametrize("name", HEADS)
def test_head_emits_a_boundary_channel(name: str) -> None:
    model = _build_head(name)
    x, mask = _head_inputs()
    out = model(x, mask)
    assert out.boundary is not None
    assert out.boundary.dim() == 4
    assert tuple(out.boundary.shape[1:]) == (B, 1, HEAD_T)
    assert torch.isfinite(out.boundary).all()
    assert (out.boundary[:, 1, :, -N_PADDED:] == 0).all()


@pytest.mark.parametrize("name", HEADS)
def test_head_trains_both_outputs(name: str) -> None:
    model = _build_head(name)
    x, mask = _head_inputs()
    out = model(x, mask)
    (out.logits.mean() + out.boundary.mean()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_asrf_keeps_its_backbone_intact() -> None:
    """The wrapper adds a head; it must not change what the encoder computes.

    An ASRF whose class logits differ from the plain backbone's would make
    every comparison against the encoder-only baseline meaningless.
    """
    from ethograph.segment.models import build_model

    backbone_params = {"num_decoders": 0, "num_layers": 4}
    torch.manual_seed(0)
    plain = build_model("asformer", backbone_params, N_FEATURES, N_CLASSES).eval()
    torch.manual_seed(0)
    wrapped = build_model("asrf", {"backbone_params": backbone_params}, N_FEATURES, N_CLASSES).eval()
    wrapped.inner.load_state_dict(plain.inner.state_dict())
    x, mask = _head_inputs()
    with torch.no_grad():
        assert torch.allclose(plain(x, mask), wrapped(x, mask).logits, atol=1e-5)


def test_asrf_defaults_to_the_asformer_encoder() -> None:
    from ethograph.segment.models.asrf import DEFAULT_BACKBONE, ASRFModel, build_asrf
    from ethograph.segment.models.vendored import ASFormer

    assert DEFAULT_BACKBONE == "asformer"
    model = build_asrf({"backbone_params": {"num_layers": 2}}, N_FEATURES, N_CLASSES)
    assert isinstance(model, ASRFModel)
    assert isinstance(model.inner, ASFormer)


def test_asrf_boundary_stages_follow_brb_stages() -> None:
    from ethograph.segment.models import build_model

    model = build_model(
        "asrf",
        {"backbone_params": {"num_decoders": 0, "num_layers": 3}, "brb_stages": 3, "brb_layers": 2},
        N_FEATURES,
        N_CLASSES,
    )
    x, mask = _head_inputs(256)
    assert model(x, mask).boundary.shape[0] == 3


def test_asrf_refuses_a_backbone_with_no_usable_trunk() -> None:
    """C2F pools the timeline and declares no trunk shape.

    Caught when the model is built, not on the first forward pass — by then
    the dataset is materialised and the run directory exists.
    """
    from ethograph.segment.models import build_model

    with pytest.raises(ValueError, match="keeps the time axis"):
        build_model("asrf", {"backbone": "c2f_tcn"}, N_FEATURES, N_CLASSES)


def test_asrf_refuses_to_wrap_itself() -> None:
    from ethograph.segment.models import build_model

    with pytest.raises(ValueError, match="not 'asrf' itself"):
        build_model("asrf", {"backbone": "asrf"}, N_FEATURES, N_CLASSES)


@pytest.mark.parametrize(
    "name,params,match",
    [
        ("asrf", {"num_f_maps": 64}, "backbone_params"),
        ("baformer", {"num_decoders": 3}, "baformer settings"),
    ],
)
def test_a_head_param_it_does_not_take_is_refused_before_training(name: str, params: dict, match: str) -> None:
    from ethograph.segment.models import build_model

    with pytest.raises(ValueError, match=match):
        build_model(name, params, N_FEATURES, N_CLASSES)


def test_baformer_exposes_one_query_set_per_decoder_level() -> None:
    model = _build_head("baformer")
    x, mask = _head_inputs()
    out = model(x, mask)
    levels = BAFORMER_PARAMS["num_decode"] + 1  # the heads are read before the first level too
    assert tuple(out.query_logits.shape) == (levels, B, BAFORMER_PARAMS["num_queries"], N_CLASSES + 1)
    assert tuple(out.query_masks.shape) == (levels, B, BAFORMER_PARAMS["num_queries"], HEAD_T)
    assert out.boundary.shape[0] == levels


def test_baformer_logits_are_log_probabilities() -> None:
    """``softmax(logits)`` must be a distribution — confidence depends on it."""
    model = _build_head("baformer").eval()
    x, mask = _head_inputs()
    with torch.no_grad():
        probs = torch.softmax(model(x, mask).logits[-1], dim=1)
    valid = probs[0]
    assert torch.allclose(valid.sum(dim=0), torch.ones(HEAD_T), atol=1e-5)


def test_baformer_votes_only_in_eval_mode() -> None:
    """Training composes the queries softly; eval hardens them into spans.

    The hard vote is what makes an edge an edge, and it is not differentiable
    — so evaluating a BaFormer left in train mode silently measures the blurred
    version. The two must therefore differ.
    """
    model = _build_head("baformer")
    x, mask = _head_inputs()
    model.eval()
    with torch.no_grad():
        voted = model(x, mask).logits
    model.train()
    with torch.no_grad():
        soft = model(x, mask).logits
    assert not torch.allclose(voted, soft)
    # The voted prediction is constant inside every boundary-delimited span,
    # so it can only change class where the boundary head put a peak.
    from ethograph.segment.boundary import boundary_peaks

    with torch.no_grad():
        model.eval()
        out = model(x, mask)
    peaks = set(boundary_peaks(torch.sigmoid(out.boundary[-1, 0, 0]).numpy(), model.boundary_threshold).tolist())
    labels = out.logits[-1, 0].argmax(dim=0)
    changes = set((torch.nonzero(labels[1:] != labels[:-1]).flatten() + 1).tolist())
    assert changes <= peaks


def test_baformer_decode_levels_cannot_exceed_encoder_layers() -> None:
    from ethograph.segment.models import build_model

    model = build_model("baformer", {**BAFORMER_PARAMS, "num_layers": 2, "num_decode": 3}, N_FEATURES, N_CLASSES)
    x, mask = _head_inputs(128)
    with pytest.raises(ValueError, match="must not exceed"):
        model(x, mask)
