"""Torch device resolution: CUDA → MPS → CPU, never hardcoded."""

from __future__ import annotations


def resolve_device(preferred: str | None = None) -> str:
    """Pick the best available torch device, honouring *preferred* when usable."""
    try:
        import torch
    except ImportError:
        return "cpu"

    available = ["cpu"]
    if torch.cuda.is_available():
        available.insert(0, "cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        available.insert(0 if "cuda" not in available else 1, "mps")

    if preferred:
        base = preferred.split(":")[0]
        if base in available:
            return preferred
    return available[0]
