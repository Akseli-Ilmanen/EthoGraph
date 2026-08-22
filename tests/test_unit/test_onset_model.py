"""Unit tests for the GradBoost onset model (labels/onset_model.py)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ethograph.io.catalog import XarrayLoader
from ethograph.labels import onset_model as om


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point ~/.ethograph at a temp dir so tests never touch the real store."""
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path / ".ethograph"))


def _make_ds(t_event: float, fs: float = 50.0, dur: float = 10.0, seed: int = 0) -> xr.Dataset:
    """One synthetic trial: a Gaussian bump on (x, y) at *t_event* plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur, 1.0 / fs)
    bump = np.exp(-0.5 * ((t - t_event) / 0.05) ** 2)
    x = 5.0 * bump + rng.normal(0, 0.1, t.size)
    y = -3.0 * bump + rng.normal(0, 0.1, t.size)
    return xr.Dataset(
        {"signal": (("time", "space"), np.column_stack([x, y]))},
        coords={"time": t, "space": ["x", "y"]},
    )


FEATURES = {"signal": {"space": ["x", "y"]}}


def _make_config(name: str = "test-model") -> om.OnsetModelConfig:
    return om.OnsetModelConfig(
        name=name,
        target_label=3,
        target_name="peck",
        features=FEATURES,
        window_s=0.4,
        tolerance_s=0.1,
        max_iter=60,
    )


# ---------------------------------------------------------------------------
# Columns + config
# ---------------------------------------------------------------------------


def test_enumerate_columns_is_deterministic():
    cols = om.enumerate_columns({"signal": {"space": ["x", "y"]}, "flat": {}})
    assert [c.name for c in cols] == ["signal|space=x", "signal|space=y", "flat"]
    assert cols[0].selections == {"space": "x"}
    assert cols[2].selections == {}


def test_config_roundtrip():
    config = _make_config()
    om.save_config(config)
    assert om.list_models() == ["test-model"]
    loaded = om.load_config("test-model")
    assert loaded == config


def test_session_id_stable_and_distinct(tmp_path):
    a = tmp_path / "sess_a.nc"
    b = tmp_path / "sub" / "sess_a.nc"
    assert om.session_id(a) == om.session_id(a)
    assert om.session_id(a) != om.session_id(b)
    assert om.session_id(a).startswith("sess_a-")


# ---------------------------------------------------------------------------
# Windowing + targets
# ---------------------------------------------------------------------------


def test_lag_offsets_symmetric_and_capped():
    offsets = om.lag_offsets(fs=1000.0, window_s=1.0)
    assert len(offsets) <= om.MAX_LAGS
    assert offsets[0] == -offsets[-1]
    assert 0 in offsets


def test_build_windows_nan_edges():
    data = np.arange(10.0)
    x = om.build_windows(data, np.array([-1, 0, 1]))
    assert x.shape == (10, 3)
    assert np.isnan(x[0, 0]) and np.isnan(x[-1, 2])
    assert x[5, 0] == 4.0 and x[5, 1] == 5.0 and x[5, 2] == 6.0


def test_make_targets_gaussian_peak():
    time = np.arange(0.0, 1.0, 0.01)
    y, w = om.make_targets(time, 0.5, tolerance_s=0.05)
    assert y.sum() > 0
    assert np.argmax(w * y) == np.argmin(np.abs(time - 0.5))
    assert np.all(w[y == 0] == 1.0)


def test_make_targets_outside_range_raises():
    time = np.arange(0.0, 1.0, 0.01)
    with pytest.raises(ValueError, match="outside"):
        om.make_targets(time, 5.0, tolerance_s=0.05)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extract_features_shape():
    loader = XarrayLoader(_make_ds(3.0))
    time, data = om.extract_features(loader, FEATURES)
    assert data.shape == (len(time), 2)
    assert np.isclose(om.sampling_rate(time), 50.0)


def test_extract_features_rejects_rate_mismatch():
    ds = _make_ds(3.0)
    t_aux = np.arange(0.0, 10.0, 1.0 / 100.0)
    ds["audio"] = ("time_aux", np.zeros(t_aux.size))
    ds = ds.assign_coords(time_aux=t_aux)
    loader = XarrayLoader(ds)
    with pytest.raises(ValueError, match="[Ss]ampling-rate mismatch"):
        om.extract_features(loader, {**FEATURES, "audio": {}})


def test_extract_features_missing_feature_raises():
    loader = XarrayLoader(_make_ds(3.0))
    with pytest.raises(ValueError, match="not available"):
        om.extract_features(loader, {"nope": {}})


# ---------------------------------------------------------------------------
# Train + predict end-to-end
# ---------------------------------------------------------------------------


def test_train_and_predict_recovers_event():
    config = _make_config()
    om.save_config(config)

    event_times = [1.5, 3.2, 5.0, 6.7, 8.1, 2.4, 4.9, 7.3]
    for i, t_event in enumerate(event_times):
        loader = XarrayLoader(_make_ds(t_event, seed=i))
        time, data = om.extract_features(loader, config.features)
        om.write_trial_training_data(config.name, "sess-a", i + 1, time, data, t_event)
    om.write_session_meta(config.name, "sess-a", {"n_trials": len(event_times)})

    summary = om.train_model(config.name)
    assert summary["n_trials"] == len(event_times)
    assert summary["n_sessions"] == 1
    assert summary["n_positive"] > 0
    assert om.is_trained(config.name)

    bundle = om.load_bundle(config.name)
    t_true = 4.2
    loader = XarrayLoader(_make_ds(t_true, seed=99))
    time, data = om.extract_features(loader, config.features)
    t_pred, confidence = om.predict_onset(bundle, time, data)
    assert abs(t_pred - t_true) <= 0.2
    assert 0.0 <= confidence <= 1.0


def test_train_without_data_raises():
    om.save_config(_make_config())
    with pytest.raises(ValueError, match="[Nn]o training data"):
        om.train_model("test-model")


def test_predict_rejects_rate_mismatch():
    config = _make_config()
    om.save_config(config)
    loader = XarrayLoader(_make_ds(2.0))
    time, data = om.extract_features(loader, config.features)
    om.write_trial_training_data(config.name, "sess-a", 1, time, data, 2.0)
    om.train_model(config.name)
    bundle = om.load_bundle(config.name)

    other = XarrayLoader(_make_ds(2.0, fs=100.0))
    time_other, data_other = om.extract_features(other, config.features)
    with pytest.raises(ValueError, match="[Ss]ampling-rate mismatch"):
        om.predict_onset(bundle, time_other, data_other)
