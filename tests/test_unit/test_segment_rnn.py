"""The recurrent baseline honours the registry contract, and padding never enters its recurrence."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import ethograph as eto  # noqa: E402
from ethograph.segment.models import available_architectures, build_model  # noqa: E402
from ethograph.segment.models.rnn import RNN_DEFAULTS  # noqa: E402

N_FEATURES = 5
N_CLASSES = 3
B = 2
T = 256
N_PADDED = 40


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(B, N_FEATURES, T)
    mask = torch.ones(B, 1, T)
    mask[1, :, -N_PADDED:] = 0
    return x, mask


def test_registered_with_its_own_defaults() -> None:
    assert "rnn" in available_architectures()
    tunable = eto.segment.tunable_params("rnn")
    assert tunable
    assert RNN_DEFAULTS.is_file()
    for key, default in tunable.items():
        build_model("rnn", {key: default}, N_FEATURES, N_CLASSES)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_forward_contract(cell: str) -> None:
    model = build_model("rnn", {"cell": cell}, N_FEATURES, N_CLASSES)
    x, mask = _inputs()
    model.train()
    out = model(x, mask)
    assert tuple(out.shape) == (1, B, N_CLASSES, T)
    assert torch.isfinite(out).all()
    assert (out[:, 1, :, -N_PADDED:] == 0).all()
    assert (out[:, :, :, : T - N_PADDED] != 0).any()
    out.mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_padding_never_enters_the_recurrence() -> None:
    """A padded sample's real frames read the same as when it is run alone.

    Without packing, the backward direction would start on the zero tail and
    the last real frames would differ between the two runs.
    """
    model = build_model("rnn", {"bidirectional": True}, N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs()
    n_real = T - N_PADDED
    with torch.no_grad():
        batched = model(x, mask)[0, 1, :, :n_real]
        alone = model(x[1:, :, :n_real], torch.ones(1, 1, n_real))[0, 0]
    assert torch.allclose(batched, alone, atol=1e-5)


def test_eval_is_deterministic() -> None:
    model = build_model("rnn", {}, N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs()
    with torch.no_grad():
        assert torch.equal(model(x, mask), model(x, mask))


def test_unknown_hyperparameter_is_refused() -> None:
    with pytest.raises(ValueError, match="num_f_maps"):
        build_model("rnn", {"num_f_maps": 64}, N_FEATURES, N_CLASSES)


def test_unknown_cell_is_refused() -> None:
    with pytest.raises(ValueError, match="cell"):
        build_model("rnn", {"cell": "rnn"}, N_FEATURES, N_CLASSES)
