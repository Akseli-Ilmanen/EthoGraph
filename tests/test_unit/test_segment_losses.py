"""The training loss is DLC2Action's ``MS_TCN_Loss``, built from upstream's
own ``config/losses.yaml``.

Nothing on our side reimplements it, so these tests cover the seam: the
defaults come from the vendored YAML, the two keys a config file cannot carry
are filled in, upstream's `dataset_inverse_weights` sentinel is refused rather
than guessed at, an unknown key is refused, and padded frames cost nothing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ethograph.segment.dataset import PAD_TARGET  # noqa: E402
from ethograph.segment.dlc2action.loss import MS_TCN_Loss  # noqa: E402
from ethograph.segment.losses import (  # noqa: E402
    DATASET_INVERSE_WEIGHTS,
    DEFAULT_TAU,
    build_loss,
    upstream_defaults,
)

N_CLASSES = 3
B, T = 2, 64


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.zeros(4, B, N_CLASSES, T)
    target = torch.zeros(B, T, dtype=torch.long)
    target[:, 20:40] = 1
    return logits, target


def test_defaults_come_from_the_vendored_losses_yaml() -> None:
    """No loss default is written on our side."""
    import yaml

    from ethograph.segment.losses import LOSS_CONFIG, LOSS_KEY

    raw = yaml.safe_load(LOSS_CONFIG.read_text(encoding="utf-8"))[LOSS_KEY]
    assert upstream_defaults() == raw


def test_build_loss_is_upstreams_loss() -> None:
    criterion, settings = build_loss({}, N_CLASSES)
    assert isinstance(criterion, MS_TCN_Loss)
    # the two things the YAML cannot carry
    assert settings["exclusive"] is True
    assert settings["weights"] is None
    # everything else is upstream's, untouched
    for key in ("focal", "gamma", "alpha"):
        assert settings[key] == upstream_defaults()[key]


def test_overrides_beat_the_upstream_defaults() -> None:
    _criterion, settings = build_loss({"focal": False, "gamma": 5}, N_CLASSES)
    assert settings["focal"] is False
    assert settings["gamma"] == 5
    assert settings["alpha"] == upstream_defaults()["alpha"]


def test_an_unknown_loss_key_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        build_loss({"smoothing": 0.15}, N_CLASSES)


def test_the_dataset_weights_sentinel_is_refused_not_guessed() -> None:
    """Resolving it needs DLC2Action's dataset layer, which is not vendored."""
    with pytest.raises(ValueError, match="not vendored"):
        build_loss({"weights": DATASET_INVERSE_WEIGHTS}, N_CLASSES)


def test_explicit_weights_are_passed_through() -> None:
    _criterion, settings = build_loss({"weights": [1.0, 2.0, 3.0]}, N_CLASSES)
    assert settings["weights"] == [1.0, 2.0, 3.0]


def test_padded_frames_cost_nothing() -> None:
    """CE ignores them via ``ignore_index``; the adapter zeroes their logits,
    so a constant log-softmax makes their consistency term exactly zero."""
    criterion, _ = build_loss({}, N_CLASSES)
    logits, target = _batch()

    full = criterion(logits, target)
    padded_target = target.clone()
    padded_target[1, -8:] = PAD_TARGET  # logits there are already zero
    padded = criterion(logits, padded_target)
    assert torch.isfinite(padded)
    assert padded <= full + 1e-6


def test_loss_accepts_the_registry_contract_shape() -> None:
    criterion, _ = build_loss({}, N_CLASSES)
    logits, target = _batch()
    stacked = criterion(logits, target)
    single = criterion(logits[-1], target)
    assert torch.isfinite(stacked) and torch.isfinite(single)


class TestTau:
    """``train.loss.tau`` — the truncation threshold of the consistency term.

    Upstream writes it into the arithmetic (``clamp(..., max=16)``), so the
    contract is: exposing it changes nothing at the default, and moving it
    moves only that term.
    """

    @staticmethod
    def _logits(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """A batch with real boundaries, so the consistency term is not zero."""
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(4, B, N_CLASSES, T, generator=generator) * 4
        target = torch.zeros(B, T, dtype=torch.long)
        target[:, 20:40] = 1
        return logits, target

    def test_the_default_is_upstreams_loss_to_the_last_bit(self) -> None:
        criterion, settings = build_loss({}, N_CLASSES)
        upstream = MS_TCN_Loss(num_classes=N_CLASSES, **{k: v for k, v in settings.items() if k != "tau"})
        logits, target = self._logits()
        assert settings["tau"] == DEFAULT_TAU
        assert torch.equal(criterion(logits, target), upstream(logits, target))

    def test_a_larger_tau_truncates_less(self) -> None:
        """τ bounds the penalty on one frame's jump, so raising it cannot lower the term."""
        logits, _ = self._logits()
        default, _ = build_loss({}, N_CLASSES)
        loose, _ = build_loss({"tau": 48.0}, N_CLASSES)
        tight, _ = build_loss({"tau": 0.5}, N_CLASSES)
        p = logits[-1]
        assert tight.consistency_loss(p) < default.consistency_loss(p) < loose.consistency_loss(p)

    def test_tau_moves_only_the_consistency_term(self) -> None:
        logits, target = self._logits()
        off_default, _ = build_loss({"alpha": 0.0}, N_CLASSES)
        off_loose, _ = build_loss({"alpha": 0.0, "tau": 48.0}, N_CLASSES)
        assert torch.equal(off_default(logits, target), off_loose(logits, target))

    def test_a_non_positive_tau_is_refused(self) -> None:
        with pytest.raises(ValueError, match="truncation threshold"):
            build_loss({"tau": 0.0}, N_CLASSES)


class TestCircleLoss:
    """The ported deep metric-learning term, and its wiring into :func:`build_objective`."""

    @staticmethod
    def _batch(n_classes: int = N_CLASSES, b: int = B, t: int = T, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(1, b, n_classes, t, generator=generator, requires_grad=True)
        target = torch.zeros(b, t, dtype=torch.long)
        target[:, t // 2 :] = 1
        mask = torch.ones(b, 1, t)
        return logits, target, mask

    def test_same_class_pairs_are_positive_and_different_class_pairs_negative(self) -> None:
        from ethograph.segment.losses import _label_similarity_pairs

        normed = torch.nn.functional.normalize(torch.randn(6, 4), dim=-1)
        label = torch.tensor([0, 0, 0, 1, 1, 1])
        sp, sn = _label_similarity_pairs(normed, label)
        # 3 same-class pairs within each group of 3: C(3,2) * 2 groups = 6
        assert sp.numel() == 6
        # every cross pair: 3 * 3 = 9
        assert sn.numel() == 9

    def test_circle_term_is_finite_and_differentiable(self) -> None:
        from ethograph.segment.losses import CircleLoss, circle_term

        logits, target, mask = self._batch()
        loss = circle_term(CircleLoss(), logits, target, mask, max_frames=None)
        assert loss is not None
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_a_single_class_batch_has_nothing_to_push_apart(self) -> None:
        from ethograph.segment.losses import CircleLoss, circle_term

        logits, target, mask = self._batch()
        target = torch.zeros_like(target)  # every frame the same class
        assert circle_term(CircleLoss(), logits, target, mask, max_frames=None) is None

    def test_padded_frames_are_excluded(self) -> None:
        from ethograph.segment.losses import CircleLoss, circle_term

        logits, target, mask = self._batch()
        mask = mask.clone()
        mask[:, :, T // 4 :] = 0  # only the always-background prefix remains
        target = target.clone()
        target[:, : T // 4] = 0
        assert circle_term(CircleLoss(), logits, target, mask, max_frames=None) is None

    def test_max_frames_subsamples_rather_than_erroring(self) -> None:
        from ethograph.segment.losses import CircleLoss, circle_term

        logits, target, mask = self._batch(t=256)
        loss = circle_term(CircleLoss(), logits, target, mask, max_frames=32)
        assert loss is not None and torch.isfinite(loss)

    @staticmethod
    def _config(**circle_kwargs):
        from ethograph.segment.config import CircleConfig, SegmentConfig, TrainConfig

        cfg = SegmentConfig(sessions=[])
        cfg.train = TrainConfig(circle=CircleConfig(**circle_kwargs))
        return cfg

    def test_default_weight_leaves_circle_loss_unbuilt(self) -> None:
        from ethograph.segment.losses import build_objective

        objective, settings = build_objective(self._config(), n_classes=N_CLASSES, fs=50.0)
        assert objective.circle_loss is None
        assert settings["circle"]["weight"] == 0.0

    def test_a_positive_weight_builds_and_records_the_loss(self) -> None:
        from ethograph.segment.losses import CircleLoss, build_objective

        objective, settings = build_objective(self._config(weight=0.5, gamma=64.0), n_classes=N_CLASSES, fs=50.0)
        assert isinstance(objective.circle_loss, CircleLoss)
        assert objective.circle_loss.gamma == 64.0
        assert settings["circle"] == {"weight": 0.5, "m": 0.25, "gamma": 64.0, "max_frames": 2048}

    def test_an_out_of_range_margin_is_refused(self) -> None:
        from ethograph.segment.config import CircleConfig

        with pytest.raises(ValueError, match="train.circle.m"):
            CircleConfig(m=1.5)

    def test_forward_reports_the_circle_part(self) -> None:
        from ethograph.segment.losses import build_objective
        from ethograph.segment.models import ModelOutput

        cfg = self._config(weight=0.5)
        cfg.train.frame_weight = 0.0
        objective, _ = build_objective(cfg, n_classes=N_CLASSES, fs=50.0)
        logits, target, mask = self._batch()
        _total, parts = objective(ModelOutput(logits=logits), target, mask)
        assert "circle" in parts and "frame" not in parts
