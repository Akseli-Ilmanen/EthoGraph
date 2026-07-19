"""Tests for ethograph.io.pynapple loading and DataLoader access."""

import numpy as np
import pynapple as nap
import pytest

from ethograph.io.catalog import (
    PynappleLoader as PynappleStore,
)
from ethograph.io.catalog import (
    _compute_shared_column_dims,
    catalog_from_pynapple,
)
from ethograph.io.pynapple import detect_trials

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_nap_data():
    """Pynapple data dict with speed (Tsd), velocity (TsdFrame), and trials."""
    trials = nap.IntervalSet(start=[0, 15], end=[10, 25])
    t = np.linspace(0, 25, 2500)
    speed = nap.Tsd(t=t, d=np.random.randn(len(t)), time_support=trials)
    velocity = nap.TsdFrame(
        t=t,
        d=np.random.randn(len(t), 3),
        columns=["x", "y", "z"],
        time_support=trials,
    )
    return {"speed": speed, "velocity": velocity, "trials": trials}


@pytest.fixture
def multi_tsdframe_data():
    """Multiple TsdFrames sharing the same columns (x, y, z)."""
    t = np.linspace(0, 10, 1000)
    return {
        "position": nap.TsdFrame(t=t, d=np.random.randn(1000, 3), columns=["x", "y", "z"]),
        "velocity": nap.TsdFrame(t=t, d=np.random.randn(1000, 3), columns=["x", "y", "z"]),
        "speed": nap.Tsd(t=t, d=np.random.randn(1000)),
    }


# ---------------------------------------------------------------------------
# detect_trials
# ---------------------------------------------------------------------------


def test_detect_trials_found(simple_nap_data):
    ep = detect_trials(simple_nap_data)
    assert ep is not None
    assert len(ep) == 2


def test_detect_trials_not_found():
    data = {"speed": nap.Tsd(t=np.arange(10), d=np.zeros(10))}
    assert detect_trials(data) is None


# ---------------------------------------------------------------------------
# Shared column dimensions
# ---------------------------------------------------------------------------


def test_shared_columns_merged():
    """TsdFrames with identical columns produce one shared dimension name."""
    t = np.linspace(0, 10, 100)
    objs = {
        "position": nap.TsdFrame(t=t, d=np.random.randn(100, 3), columns=["x", "y", "z"]),
        "velocity": nap.TsdFrame(t=t, d=np.random.randn(100, 3), columns=["x", "y", "z"]),
    }
    dim_map = _compute_shared_column_dims(objs)
    assert dim_map["position"] == dim_map["velocity"] == "columns"


def test_different_columns_stay_separate():
    """TsdFrames with different columns get separate dimension names."""
    t = np.linspace(0, 10, 100)
    objs = {
        "position": nap.TsdFrame(t=t, d=np.random.randn(100, 3), columns=["x", "y", "z"]),
        "emg": nap.TsdFrame(t=t, d=np.random.randn(100, 2), columns=["ch1", "ch2"]),
    }
    dim_map = _compute_shared_column_dims(objs)
    assert dim_map["position"] != dim_map["emg"]


def test_single_tsdframe_keeps_prefixed_dim():
    """A lone TsdFrame gets '{name}_columns' (no merging needed)."""
    t = np.linspace(0, 10, 100)
    objs = {
        "velocity": nap.TsdFrame(t=t, d=np.random.randn(100, 3), columns=["x", "y", "z"]),
    }
    dim_map = _compute_shared_column_dims(objs)
    assert dim_map["velocity"] == "velocity_columns"


# ---------------------------------------------------------------------------
# catalog_from_pynapple (replaces extract_type_vars_pynapple)
# ---------------------------------------------------------------------------


def test_catalog_basic(simple_nap_data):
    cat = catalog_from_pynapple(simple_nap_data)
    assert "speed" in cat.features
    assert "velocity" in cat.features
    assert "individuals" in cat.combos
    assert cat.trial_conditions == []


def test_catalog_detects_tsdframe_columns(simple_nap_data):
    cat = catalog_from_pynapple(simple_nap_data)
    assert "velocity_columns" in cat.combos
    assert list(cat.combo_values("velocity_columns")) == ["x", "y", "z"]


def test_catalog_shared_columns(multi_tsdframe_data):
    """Shared columns produce one 'columns' combo instead of two."""
    cat = catalog_from_pynapple(multi_tsdframe_data)
    assert "columns" in cat.combos
    assert "position_columns" not in cat.combos
    assert "velocity_columns" not in cat.combos
    assert list(cat.combo_values("columns")) == ["x", "y", "z"]


def test_catalog_detects_changepoints():
    cp_times = np.array([10.0, 25.0, 50.0, 75.0])
    group = nap.TsGroup({0: nap.Ts(t=cp_times)})
    group.set_info(type=["changepoints"])
    data = {"cp_group": group}
    cat = catalog_from_pynapple(data)
    assert "cp_group" in cat.changepoints


def test_catalog_skips_intervalset(simple_nap_data):
    cat = catalog_from_pynapple(simple_nap_data)
    assert "trials" not in cat.features


# ---------------------------------------------------------------------------
# PynappleStore
# ---------------------------------------------------------------------------


def test_store_features(multi_tsdframe_data):
    store = PynappleStore(multi_tsdframe_data)
    assert set(store.features) == {"position", "velocity", "speed"}


def test_store_dims_shared(multi_tsdframe_data):
    store = PynappleStore(multi_tsdframe_data)
    assert "columns" in store.dims
    assert list(store.dims["columns"]) == ["x", "y", "z"]
    assert "individuals" in store.dims


def test_store_get_type_vars(multi_tsdframe_data):
    store = PynappleStore(multi_tsdframe_data)
    tvd = store.get_type_vars()
    assert "features" in tvd
    assert "columns" in tvd
    assert "individuals" in tvd


def test_store_select_tsd(multi_tsdframe_data):
    """Selecting a Tsd feature returns 1-D data."""
    store = PynappleStore(multi_tsdframe_data)
    pd = store.select("speed", {}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.data.ndim == 1
    assert len(pd.time) == len(pd.data)


def test_store_select_tsdframe_all_columns(multi_tsdframe_data):
    """Without column selection, TsdFrame returns 2-D data."""
    store = PynappleStore(multi_tsdframe_data)
    pd = store.select("position", {}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.data.ndim == 2
    assert pd.data.shape[1] == 3
    assert pd.dim_labels == ["x", "y", "z"]


def test_store_select_tsdframe_single_column(multi_tsdframe_data):
    """With column selection, TsdFrame returns 1-D data."""
    store = PynappleStore(multi_tsdframe_data)
    pd = store.select("position", {"columns": "x"}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.data.ndim == 1


def test_store_select_with_time_window(multi_tsdframe_data):
    """Narrower time window restricts the returned data (time stays absolute)."""
    store = PynappleStore(multi_tsdframe_data)
    full = store.select("speed", {}, t0=0.0, t1=10.0)
    windowed = store.select("speed", {}, t0=2.0, t1=5.0)
    assert windowed is not None
    assert len(windowed.time) < len(full.time)
    assert windowed.time[0] >= 2.0 - 0.1
    assert windowed.time[-1] <= 5.0 + 0.1


def test_store_absolute_time():
    """Returned time keeps absolute session coordinates (never re-based to t0),
    so a viewport starting anywhere (e.g. a sliding fixed window) aligns with
    the plot x-axis."""
    trials = nap.IntervalSet(start=[100, 200], end=[110, 210])
    t = np.linspace(100, 210, 11000)
    speed = nap.Tsd(t=t, d=np.random.randn(len(t)), time_support=trials)
    data = {"speed": speed}

    store = PynappleStore(data)
    pd = store.select("speed", {}, t0=100.0, t1=110.0)
    assert pd is not None
    assert pd.time[0] >= 100.0 - 0.1
    assert pd.time[-1] <= 110.0 + 0.1

    pd2 = store.select("speed", {}, t0=200.0, t1=210.0)
    assert pd2 is not None
    assert pd2.time[0] >= 200.0 - 0.1


def test_store_select_nonexistent_feature(multi_tsdframe_data):
    store = PynappleStore(multi_tsdframe_data)
    assert store.select("nonexistent", {}, t0=0.0, t1=10.0) is None


def test_store_select_sparse_changepoints():
    """Sparse Ts changepoints should only mark actual CP frames, not every frame."""
    t = np.linspace(0, 10, 2000)
    speed = nap.Tsd(t=t, d=np.random.randn(2000))

    cp_times = np.array([1.0, 3.5, 7.2])
    group = nap.TsGroup({0: nap.Ts(t=cp_times)})
    group.set_info(
        type=["changepoints"],
        target_feature=["speed"],
        source_label=["unit_0"],
    )

    store = PynappleStore({"speed": speed, "cps": group})
    pd = store.select("speed", {}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.changepoints is not None
    cp_binary = list(pd.changepoints.values())[0]
    assert cp_binary.sum() == 3
    assert len(cp_binary) == len(pd.time)


def test_store_select_dense_tsd_changepoints():
    """Dense Tsd changepoints (legacy) should only mark non-zero frames."""
    t = np.linspace(0, 10, 2000)
    speed = nap.Tsd(t=t, d=np.random.randn(2000))

    mask = np.zeros(2000, dtype=np.float32)
    mask[100] = 1.0
    mask[500] = 1.0
    mask[1500] = 1.0
    dense_cp = nap.Tsd(t=t, d=mask)
    group = nap.TsGroup({0: dense_cp})
    group.set_info(
        type=["changepoints"],
        target_feature=["speed"],
        source_label=["unit_0"],
    )

    store = PynappleStore({"speed": speed, "cps": group})
    pd = store.select("speed", {}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.changepoints is not None
    cp_binary = list(pd.changepoints.values())[0]
    assert cp_binary.sum() == 3
    assert len(cp_binary) == len(pd.time)


# ---------------------------------------------------------------------------
# get_cp_times (for changepoint correction)
# ---------------------------------------------------------------------------


def test_store_get_cp_times_sparse():
    """get_cp_times returns sparse trial-relative timestamps for correction."""
    t = np.linspace(0, 10, 2000)
    speed = nap.Tsd(t=t, d=np.random.randn(2000))

    cp_times = np.array([1.0, 3.5, 7.2])
    group = nap.TsGroup({0: nap.Ts(t=cp_times)})
    group.set_info(
        type=["changepoints"],
        target_feature=["speed"],
        source_label=["unit_0"],
    )

    store = PynappleStore({"speed": speed, "cps": group})
    result = store.get_cp_times("speed", t0=0.0, t1=10.0)
    assert len(result) == 3
    np.testing.assert_allclose(result, cp_times, atol=1e-6)


def test_store_get_cp_times_no_changepoints():
    t = np.linspace(0, 10, 1000)
    store = PynappleStore({"speed": nap.Tsd(t=t, d=np.random.randn(1000))})
    result = store.get_cp_times("speed", t0=0.0, t1=10.0)
    assert len(result) == 0
