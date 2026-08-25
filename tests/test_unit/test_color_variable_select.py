"""A colour variable must never be able to crash ``select``.

The panel's selections are sanitized against its *feature's* dims only
(``PanelStateMixin._sanitize_selections``), but ``XarrayLoader.select`` applies
the same selections to the saved colour variable — which can carry dims the
feature doesn't. A multi-value dim the colour var alone has (or one whose
pinned value the colour var lacks, e.g. restored from a stale
``local_settings.yaml`` layout) used to reach ``sel_valid`` free, hand it a 3-D
block and blow its ``(time,)``/``(time, dim)`` assertion — aborting the whole
dataset load. ``select`` now pins such dims to their first value, the same rule
``_sanitize_selections`` applies to extra feature dims.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ethograph.io.catalog import XarrayLoader

N = 10


def _make_ds() -> xr.Dataset:
    time = np.arange(N) / 10.0
    rgb = np.zeros((N, 2, 3))
    rgb[:, 0, :] = 0.25  # individual "a"
    rgb[:, 1, :] = 0.75  # individual "b"
    return xr.Dataset(
        {
            "speed": (("time",), np.arange(float(N))),
            "rgb_state": (("time", "individual", "RGB"), rgb),
            "rgb_bare": (("time", "dim_0"), np.stack([time, -time], axis=1)),
        },
        coords={"time": time, "individual": ["a", "b"]},
    )


def test_color_var_with_unpinned_extra_dim_pins_first_value():
    """Feature is 1-D, colour var carries an ``individual`` dim the selections
    never mention: previously AssertionError (3-D block), now the first
    individual is pinned and colour data comes back ``(time, RGB)``."""
    loader = XarrayLoader(_make_ds())
    pd = loader.select("speed", {}, color_variable="rgb_state")
    assert pd is not None
    assert pd.color_data is not None
    assert pd.color_data.shape == (N, 3)
    assert np.allclose(pd.color_data, 0.25)  # individual "a", the first value


def test_color_var_with_stale_pinned_value_repins():
    """A selection value the colour var's coord lacks (stale layout) used to
    raise KeyError out of ``.sel``; it re-pins to the first value instead."""
    loader = XarrayLoader(_make_ds())
    pd = loader.select("speed", {"individual": "gone"}, color_variable="rgb_state")
    assert pd is not None
    assert pd.color_data is not None
    assert pd.color_data.shape == (N, 3)
    assert np.allclose(pd.color_data, 0.25)


def test_color_var_stale_value_on_single_value_dim_repins():
    """The stale value can name a dim the colour var has only ONE of.

    A session recorded on one animal, replayed on another: the layout (or a
    curation workflow copied between animals) still pins ``individual`` to a
    name this session's colour var does not carry, and the dim is size 1
    because there is only the one animal. Skipping single-value dims left that
    name to reach ``.sel`` and raise KeyError.
    """
    ds = _make_ds().sel(individual=["b"])
    loader = XarrayLoader(ds)
    pd = loader.select("speed", {"individual": "a"}, color_variable="rgb_state")
    assert pd is not None
    assert pd.color_data is not None
    assert np.allclose(pd.color_data, 0.75)  # the one individual present


def test_color_var_valid_pinned_value_is_respected():
    loader = XarrayLoader(_make_ds())
    pd = loader.select("speed", {"individual": "b"}, color_variable="rgb_state")
    assert pd is not None
    assert pd.color_data is not None
    assert np.allclose(pd.color_data, 0.75)


def test_color_var_coordless_extra_dim_pins_index_zero():
    """An extra dim with no coordinate goes through the isel path."""
    loader = XarrayLoader(_make_ds())
    pd = loader.select("speed", {}, color_variable="rgb_bare")
    assert pd is not None
    assert pd.color_data is not None
    assert pd.color_data.ndim == 1
    assert np.allclose(pd.color_data, np.arange(N) / 10.0)  # column 0
