"""Energy envelope computation."""

from __future__ import annotations

import numpy as np


ENERGY_DISPLAY_NAMES = {
    "energy_lowpass": "SOS lowpass envelope",
    "energy_highpass": "SOS highpass envelope",
    "energy_band": "SOS bandpass envelope",
    "energy_meansquared": "Vocalpy meansquared (amplitude)",
    "energy_ava": "Vocalpy AVA (spectral power)",
}


def compute_energy_envelope(
    data: np.ndarray, rate: float, metric: str, app_state,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute energy envelope using registry-driven dispatch.

    Looks up the wrapper function and cached user params for the given metric,
    then returns (env_time, envelope).
    """
    import inspect

    from ethograph.features.energy import (
        bandpass_envelope,
        env_ava,
        env_meansquared,
        highpass_envelope,
        lowpass_envelope,
    )

    _METRIC_FUNCS = {
        "energy_lowpass": lowpass_envelope,
        "energy_highpass": highpass_envelope,
        "energy_band": bandpass_envelope,
        "energy_meansquared": env_meansquared,
        "energy_ava": env_ava,
    }

    func = _METRIC_FUNCS.get(metric, lowpass_envelope)
    registry_key = metric

    cache = getattr(app_state, "function_params_cache", None) or {}
    cached = cache.get(registry_key, {})

    sig = inspect.signature(func)
    valid_keys = set(sig.parameters) - {"data", "rate"}
    params = {k: v for k, v in cached.items() if k in valid_keys}

    return func(data, rate, **params)
