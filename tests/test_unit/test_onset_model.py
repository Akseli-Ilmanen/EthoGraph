"""Unit tests for the LightGBM onset model (labels/onset_model.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr
import yaml

from ethograph.io.catalog import XarrayLoader
from ethograph.labels import onset_model as om


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point ~/.ethograph at a temp dir so tests never touch the real store."""
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path / ".ethograph"))


def _make_ds(
    t_event: float,
    fs: float = 50.0,
    dur: float = 10.0,
    seed: int = 0,
    t_second: float | None = None,
) -> xr.Dataset:
    """One synthetic trial: Gaussian bumps on (x, y) plus noise.

    *t_event* drives both channels; *t_second*, when given, adds a bump only
    on y — a second point-event class with its own signature.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur, 1.0 / fs)
    bump = np.exp(-0.5 * ((t - t_event) / 0.05) ** 2)
    x = 5.0 * bump + rng.normal(0, 0.1, t.size)
    y = -3.0 * bump + rng.normal(0, 0.1, t.size)
    if t_second is not None:
        y = y + 8.0 * np.exp(-0.5 * ((t - t_second) / 0.05) ** 2)
    return xr.Dataset(
        {"signal": (("time", "space"), np.column_stack([x, y]))},
        coords={"time": t, "space": ["x", "y"]},
    )


FEATURES = {"signal": {"space": ["x", "y"]}}


def _make_config(name: str = "test-model", targets: dict[int, str] | None = None) -> om.OnsetModelConfig:
    return om.OnsetModelConfig(
        name=name,
        targets=targets if targets is not None else {3: "peck"},
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


def test_legacy_single_target_config_upgrades(tmp_path):
    """A config written before multi-target support still loads."""
    raw = {
        "name": "old-model",
        "target_label": 7,
        "target_name": "land",
        "features": FEATURES,
    }
    d = om.model_dir("old-model")
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = om.load_config("old-model")
    assert config.targets == {7: "land"}
    assert config.target_name(7) == "land"


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
        om.write_trial_training_data(config.name, "sess-a", i + 1, time, data, {3: t_event})
    om.write_session_meta(config.name, "sess-a", {"n_trials": len(event_times)})

    summary = om.train_model(config.name)
    assert summary["n_trials"] == len(event_times)
    assert summary["n_sessions"] == 1
    assert summary["targets"][3]["n_positive"] > 0
    assert om.is_trained(config.name)

    bundle = om.load_bundle(config.name)
    t_true = 4.2
    loader = XarrayLoader(_make_ds(t_true, seed=99))
    time, data = om.extract_features(loader, config.features)
    predictions = om.predict_events(bundle, time, data)
    assert set(predictions) == {3}
    assert abs(predictions[3].time - t_true) <= 0.2
    assert 0.0 <= predictions[3].confidence <= 1.0


def test_two_targets_predicted_together():
    """One model, two point-event classes, one pass over the features."""
    config = _make_config(targets={3: "peck", 4: "land"})
    om.save_config(config)

    events = [(1.5, 6.0), (3.2, 8.0), (5.0, 1.2), (6.7, 2.5), (8.1, 4.0), (2.4, 7.1)]
    for i, (t_peck, t_land) in enumerate(events):
        loader = XarrayLoader(_make_ds(t_peck, seed=i, t_second=t_land))
        time, data = om.extract_features(loader, config.features)
        om.write_trial_training_data(config.name, "sess-a", i + 1, time, data, {3: t_peck, 4: t_land})

    summary = om.train_model(config.name)
    assert set(summary["targets"]) == {3, 4}

    bundle = om.load_bundle(config.name)
    assert set(bundle["models"]) == {3, 4}
    loader = XarrayLoader(_make_ds(4.2, seed=99, t_second=8.4))
    time, data = om.extract_features(loader, config.features)
    predictions = om.predict_events(bundle, time, data)
    assert abs(predictions[3].time - 4.2) <= 0.3
    assert abs(predictions[4].time - 8.4) <= 0.3
    assert predictions[3].name == "peck"


def test_target_without_training_data_raises():
    """A target no stored trial carries cannot be trained — say so."""
    config = _make_config(targets={3: "peck", 4: "land"})
    om.save_config(config)
    loader = XarrayLoader(_make_ds(2.0))
    time, data = om.extract_features(loader, config.features)
    om.write_trial_training_data(config.name, "sess-a", 1, time, data, {3: 2.0})
    with pytest.raises(ValueError, match="No training trial carries 'land'"):
        om.train_model(config.name)


class TestConfidence:
    def test_spike_is_sharper_than_flat(self):
        spike = np.zeros(500)
        spike[250] = 1.0
        assert om.curve_sharpness(spike) > 0.95
        assert om.curve_sharpness(np.ones(500)) == 0.0

    def test_confidence_needs_peak_and_sharpness(self):
        """A confident model believes strongly AND localises that belief."""
        sure = om.OnsetPrediction(1, "a", 1.0, peak=0.9, sharpness=0.9)
        weak_peak = om.OnsetPrediction(1, "a", 1.0, peak=0.1, sharpness=0.9)
        smeared = om.OnsetPrediction(1, "a", 1.0, peak=0.9, sharpness=0.1)
        assert sure.confidence > weak_peak.confidence
        assert sure.confidence > smeared.confidence
        assert 0.0 <= smeared.confidence <= 1.0

    def test_empty_curve_is_not_confident(self):
        assert om.curve_sharpness(np.zeros(100)) == 0.0
        assert om.curve_sharpness(np.array([1.0])) == 0.0


def test_train_without_data_raises():
    om.save_config(_make_config())
    with pytest.raises(ValueError, match="[Nn]o training data"):
        om.train_model("test-model")


def test_predict_rejects_rate_mismatch():
    config = _make_config()
    om.save_config(config)
    loader = XarrayLoader(_make_ds(2.0))
    time, data = om.extract_features(loader, config.features)
    om.write_trial_training_data(config.name, "sess-a", 1, time, data, {3: 2.0})
    om.train_model(config.name)
    bundle = om.load_bundle(config.name)

    other = XarrayLoader(_make_ds(2.0, fs=100.0))
    time_other, data_other = om.extract_features(other, config.features)
    with pytest.raises(ValueError, match="[Ss]ampling-rate mismatch"):
        om.predict_events(bundle, time_other, data_other)

# ---------------------------------------------------------------------------
# Sequence model (CRF)
# ---------------------------------------------------------------------------
#
# Fitting is the expensive part of these tests: scikit-learn's boosting is an
# order of magnitude slower with sample weights, and cross-fitting refits every
# target once per fold. So the A-B-C model is trained once for the whole module
# and the tests work off the returned bundle, which needs no model store.

#: Three classes, each with its own signal channel.
SEQ_FEATURES = {"signal": {"chan": ["a", "b", "c"]}}
SEQ_TARGETS = {1: "A", 2: "B", 3: "C"}
SEQ_FS = 50.0


def _make_seq_ds(times, seed=0, dur=9.0, decoy=None) -> xr.Dataset:
    """A trial whose channel *k* carries a bump at ``times[k]``.

    *decoy* is ``(channel, time, amplitude)`` — an extra bump on one channel,
    used to build a trial where reading each channel on its own goes wrong.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur, 1.0 / SEQ_FS)
    columns = [5.0 * np.exp(-0.5 * ((t - te) / 0.06) ** 2) + rng.normal(0, 0.3, t.size) for te in times]
    if decoy is not None:
        channel, t_decoy, amplitude = decoy
        columns[channel] = columns[channel] + amplitude * np.exp(-0.5 * ((t - t_decoy) / 0.06) ** 2)
    return xr.Dataset(
        {"signal": (("time", "chan"), np.column_stack(columns))},
        coords={"time": t, "chan": ["a", "b", "c"]},
    )


def _seq_config(name: str, use_crf: bool = True) -> om.OnsetModelConfig:
    return om.OnsetModelConfig(
        name=name,
        targets=SEQ_TARGETS,
        features=SEQ_FEATURES,
        window_s=0.2,
        tolerance_s=0.1,
        max_iter=40,
        use_crf=use_crf,
    )


def _write_seq_trials(name: str, n_trials: int, seed: int = 0) -> None:
    """Trials whose events always run A → B → C, at jittered times."""
    config = om.load_config(name)
    rng = np.random.default_rng(seed)
    for i in range(n_trials):
        base = 0.8 + rng.uniform(0.0, 0.8)
        times = [base, base + 2.5 + rng.uniform(0, 0.4), base + 5.0 + rng.uniform(0, 0.4)]
        loader = XarrayLoader(_make_seq_ds(times, seed=i))
        time, data = om.extract_features(loader, config.features)
        om.write_trial_training_data(name, "sess-a", i, time, data, dict(zip(SEQ_TARGETS, times)))
    om.write_session_meta(name, "sess-a", {"n_trials": n_trials})


@pytest.fixture(scope="module")
def sequence_model(tmp_path_factory):
    """The A→B→C model, trained once: ``(bundle, summary)``.

    Returns the loaded bundle rather than a model name, so the tests using it
    are independent of which store the per-test fixture points at.
    """
    home = tmp_path_factory.mktemp("seq-home")
    previous = os.environ.get("ETHOGRAPH_HOME")
    os.environ["ETHOGRAPH_HOME"] = str(home)
    try:
        om.save_config(_seq_config("seq"))
        _write_seq_trials("seq", n_trials=8)
        summary = om.train_model("seq")
        return om.load_bundle("seq"), summary
    finally:
        if previous is None:
            del os.environ["ETHOGRAPH_HOME"]
        else:
            os.environ["ETHOGRAPH_HOME"] = previous


def _seq_features(bundle: dict, times, seed: int, decoy=None):
    """Assemble one test trial the way the predict path does."""
    loader = XarrayLoader(_make_seq_ds(times, seed=seed, decoy=decoy))
    return om.extract_features(loader, om.bundle_config(bundle).features)


class TestPhaseTags:
    def test_tag_is_the_most_recent_event(self):
        time = np.arange(0.0, 1.0, 0.1)
        tags = om.phase_tags(time, {7: 0.3, 9: 0.6})
        assert tags == ["none", "none", "none", "7", "7", "7", "9", "9", "9", "9"]

    def test_no_events_is_all_none(self):
        assert set(om.phase_tags(np.arange(5.0), {})) == {om.CRF_NONE_TAG}

    def test_tags_follow_time_not_label_order(self):
        """B before A in time means the tags read none → B → A."""
        time = np.arange(0.0, 1.0, 0.25)
        assert om.phase_tags(time, {1: 0.75, 2: 0.25}) == ["none", "2", "2", "1"]


class TestCrfFeatures:
    def test_one_dict_per_frame_with_neighbours(self):
        time = np.arange(0.0, 0.5, 0.1)
        sequence = om.crf_features(time, {1: np.array([0.0, 0.2, 0.9, 0.3, 0.1])})
        assert len(sequence) == 5
        assert sequence[2]["p:1"] == pytest.approx(0.9)
        assert sequence[2]["p:1@-1"] == pytest.approx(0.2)
        assert sequence[2]["p:1@+1"] == pytest.approx(0.3)
        # Edges repeat rather than wrap, and position spans the trial.
        assert sequence[0]["p:1@-1"] == pytest.approx(0.0)
        assert sequence[0]["t"] == pytest.approx(0.0)
        assert sequence[-1]["t"] == pytest.approx(1.0)


class TestObservedSequences:
    def test_counts_orders_by_event_time(self):
        config = om.OnsetModelConfig(name="m", targets={1: "A", 2: "B"})
        trials = [
            om.TrainingTrial(np.arange(3.0), np.zeros((3, 1)), {1: 0.0, 2: 1.0}, 1.0),
            om.TrainingTrial(np.arange(3.0), np.zeros((3, 1)), {1: 0.0, 2: 1.0}, 1.0),
            om.TrainingTrial(np.arange(3.0), np.zeros((3, 1)), {2: 0.0, 1: 1.0}, 1.0),
        ]
        assert om.observed_sequences(trials, config) == {"1-2": 2, "2-1": 1}


class TestSequenceModel:
    def test_bundle_carries_a_crf_over_the_phase_tags(self, sequence_model):
        bundle, _ = sequence_model
        assert bundle["crf"] is not None
        assert set(bundle["crf"].classes_) == {om.CRF_NONE_TAG, "1", "2", "3"}

    def test_summary_reports_the_order_it_saw(self, sequence_model):
        _, summary = sequence_model
        assert summary["crf"]["sequences"] == {"1-2-3": 8}

    def test_crf_rejects_an_order_the_training_never_showed(self, sequence_model):
        """The point of the sequence model.

        Channel b carries a decoy bump *before* A, so reading each class on
        its own puts B first — an order the training trials never contain.
        Decoding the trial as a sequence rejects it and finds the real B.
        """
        bundle, _ = sequence_model
        time, data = _seq_features(bundle, [1.2, 3.8, 6.4], seed=99, decoy=(1, 0.3, 6.0))

        alone = om.predict_events(bundle, time, data, use_crf=False)
        assert alone[2].time < alone[1].time  # B before A — out of order

        ordered = om.predict_events(bundle, time, data, use_crf=True)
        assert ordered[1].time < ordered[2].time < ordered[3].time
        assert abs(ordered[2].time - 3.8) <= 0.3

    def test_predictions_stay_confident_and_in_range(self, sequence_model):
        bundle, _ = sequence_model
        time, data = _seq_features(bundle, [1.2, 3.8, 6.4], seed=77)
        predictions = om.predict_events(bundle, time, data)
        assert set(predictions) == set(SEQ_TARGETS)
        for prediction in predictions.values():
            assert 0.0 <= prediction.confidence <= 1.0
            assert time[0] <= prediction.time <= time[-1]

    def test_no_crf_without_the_flag(self):
        """A model trained with ``use_crf`` off carries no sequence model."""
        config = _make_config()
        om.save_config(config)
        loader = XarrayLoader(_make_ds(2.0))
        time, data = om.extract_features(loader, config.features)
        om.write_trial_training_data(config.name, "s", 1, time, data, {3: 2.0})
        summary = om.train_model(config.name)
        assert "crf" not in summary
        assert om.load_bundle(config.name).get("crf") is None

    def test_single_trial_cannot_train_a_sequence_model(self):
        config = _make_config(name="lonely")
        config.use_crf = True
        om.save_config(config)
        loader = XarrayLoader(_make_ds(2.0))
        time, data = om.extract_features(loader, config.features)
        om.write_trial_training_data("lonely", "s", 1, time, data, {3: 2.0})
        with pytest.raises(ValueError, match="at least 2 training trials"):
            om.train_model("lonely")

    def test_absurdly_long_trials_are_refused(self):
        config = om.OnsetModelConfig(name="m", targets={1: "A"})
        with pytest.raises(ValueError, match="capped at"):
            om.decode_crf(object(), np.arange(om.CRF_MAX_FRAMES + 1.0), {}, config)
