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
# Derivative columns
# ---------------------------------------------------------------------------


def test_derivative_column_follows_the_value_it_is_taken_from():
    cols = om.enumerate_columns({"signal": {"space": ["x", "y"]}, "flat": {}}, ["signal"])
    assert [c.name for c in cols] == [
        "signal|space=x",
        "signal|space=x|d/dt",
        "signal|space=y",
        "signal|space=y|d/dt",
        "flat",
    ]
    assert [c.derivative for c in cols] == [False, True, False, True, False]


def test_extract_features_derivative_is_np_gradient():
    loader = XarrayLoader(_make_ds(3.0))
    time, data = om.extract_features(loader, FEATURES, derivatives=["signal"])
    assert data.shape == (len(time), 4)
    for value, derivative in ((0, 1), (2, 3)):
        assert np.allclose(data[:, derivative], np.gradient(data[:, value], time))


def test_extract_model_features_reads_the_config_derivatives():
    config = _make_config()
    config.derivatives = ["signal"]
    loader = XarrayLoader(_make_ds(3.0))
    _time, data = om.extract_model_features(loader, config)
    assert data.shape[1] == len(config.columns()) == 4


def test_a_config_written_before_derivatives_reads_back_with_none():
    config = _make_config()
    om.save_config(config)
    path = om.model_dir(config.name) / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["derivatives"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert om.load_config(config.name).derivatives == []


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
    result = om.predict_trial(bundle, time, data)
    predictions = result.events
    assert abs(predictions[3].time - 4.2) <= 0.3
    assert abs(predictions[4].time - 8.4) <= 0.3
    assert predictions[3].name == "peck"
    # The curves the events were read off come back with them: one per
    # target, on the caller's clock, so review can draw what the single
    # confidence number compressed.
    assert set(result.curves) == {3, 4}
    assert all(len(curve) == len(time) for curve in result.curves.values())
    assert om.predict_events(bundle, time, data)[3].time == predictions[3].time


def test_target_without_training_data_raises():
    """A target no stored trial carries cannot be trained — say so."""
    config = _make_config(targets={3: "peck", 4: "land"})
    om.save_config(config)
    loader = XarrayLoader(_make_ds(2.0))
    time, data = om.extract_features(loader, config.features)
    om.write_trial_training_data(config.name, "sess-a", 1, time, data, {3: 2.0})
    with pytest.raises(ValueError, match="No training trial carries 'land'"):
        om.train_model(config.name)


def _bump(n_frames: int, at: int, amplitude: float = 0.8, width: float = 2.5) -> np.ndarray:
    """A Gaussian bump of *width* frames on a low baseline."""
    t = np.arange(n_frames)
    return 0.02 + amplitude * np.exp(-0.5 * ((t - at) / width) ** 2)


def _hump(n_frames: int, at: int, amplitude: float, width: float) -> np.ndarray:
    """A bare Gaussian, to add as a rival to a :func:`_bump`."""
    t = np.arange(n_frames)
    return amplitude * np.exp(-0.5 * ((t - at) / width) ** 2)


class TestConfidence:
    """Confidence is the height of the curve's tallest peak — nothing else.

    The number has to be readable straight off the curve the review draws,
    because that is what lets a user set a threshold by looking at it.
    """

    def test_the_tallest_peak_is_the_event(self):
        curve = _bump(3000, 1500, amplitude=0.8) + _hump(3000, 750, 0.4, 2.5)
        index, height = om.tallest_peak(curve)
        assert index == 1500
        assert height == pytest.approx(0.82, abs=0.01)

    def test_a_rising_edge_is_not_a_peak(self):
        """A curve still climbing at the trial's end must not read as certain."""
        ramp = np.linspace(0.0, 0.9, 500)
        index, height = om.tallest_peak(ramp + _hump(500, 200, 0.3, 5.0))
        assert index == 200
        assert height == pytest.approx(0.66, abs=0.02)

    def test_a_flat_curve_scores_its_own_level(self):
        assert om.tallest_peak(np.zeros(100))[1] == 0.0
        assert om.tallest_peak(np.full(100, 0.3))[1] == pytest.approx(0.3)

    def test_an_empty_curve_is_not_confident(self):
        assert om.tallest_peak(np.array([]))[1] == 0.0
        assert om.tallest_peak(np.array([0.7]))[1] == pytest.approx(0.7)

    def test_height_is_the_confidence_the_curve_shows(self):
        """What lands in the TSV is a point on the drawn curve."""
        curve = _bump(3000, 1500, amplitude=0.55)
        index, height = om.tallest_peak(curve)
        assert om.OnsetPrediction(3, "peck", 15.0, confidence=height).confidence == pytest.approx(curve[index])


class TestPlacement:
    """A point label goes at the frame the curve names — no derived statistic."""

    def test_the_event_is_placed_at_the_tallest_peak(self):
        time = np.arange(0, 10, 0.01)
        curve = _bump(len(time), 640, amplitude=0.7)
        index, height = om.tallest_peak(curve)
        assert index == 640
        assert float(time[index]) == pytest.approx(6.40)
        assert height == pytest.approx(curve[640])

    def test_no_class_moves_another(self):
        """Every target is read off its own curve, in isolation.

        A surprising order is reported by check_expectations, never fixed by
        pulling an event away from its own evidence.
        """
        time = np.arange(0, 10, 0.01)
        curves = {3: _bump(len(time), 700), 4: _bump(len(time), 300)}
        placed = {label: float(time[om.tallest_peak(curve)[0]]) for label, curve in curves.items()}
        assert placed[3] == pytest.approx(7.0)
        assert placed[4] == pytest.approx(3.0)


class TestCalibration:
    """How often the model is right is a verdict on the **model**.

    It is fitted on trials the classifiers did not see and reported when the
    model trains. It is deliberately *not* applied to a label's confidence:
    that number stays the height of the drawn curve, so the user can set a
    threshold by looking at the picture rather than by trusting a transform.
    """

    def test_a_perfect_record_still_admits_doubt(self):
        """8 of 8 held out is 9/10, never 1.0 — eight trials cannot say more."""
        cal = om.TargetCalibration(n_trials=8, n_hits=8, slope=1.0, intercept=0.0)
        assert cal.hit_rate == pytest.approx(9 / 10)

    def test_nothing_held_out_knows_nothing(self):
        """One training trial: nothing is known about how often it is right."""
        assert om.TargetCalibration(n_trials=0, n_hits=0, slope=1.0, intercept=0.0).hit_rate == 0.5

    def test_the_record_does_not_touch_a_label(self):
        """A label's confidence is its curve's peak — the model's record is
        reported at training, never folded into the number on a label."""
        curve = _bump(3000, 1500, amplitude=0.9)
        assert om.OnsetPrediction(1, "a", 1.0, confidence=om.tallest_peak(curve)[1]).confidence == pytest.approx(
            curve.max()
        )

    def test_fit_reads_the_held_out_curves(self):
        """A hit is a prediction within the tolerance of the labelled event."""
        config = _make_config()
        time = np.arange(0, 10, 0.01)
        trials = [om.TrainingTrial(time, np.zeros((len(time), 2)), {3: 5.0}, 100.0) for _ in range(4)]
        curves = [
            {3: _bump(len(time), 500)},  # right on the event
            {3: _bump(len(time), 505)},  # within the tolerance
            {3: _bump(len(time), 900)},  # nowhere near
            {3: _bump(len(time), 100)},
        ]
        cal = om.fit_confidence_calibration(trials, curves, config)[3]
        assert (cal.n_trials, cal.n_hits) == (4, 2)
        assert cal.hit_rate == pytest.approx(3 / 6)

    def test_training_records_the_held_out_score(self):
        config = _make_config()
        om.save_config(config)
        for i, t_event in enumerate([1.5, 3.2, 5.0, 6.7, 8.1, 2.4]):
            loader = XarrayLoader(_make_ds(t_event, seed=i))
            time, data = om.extract_features(loader, config.features)
            om.write_trial_training_data(config.name, "sess-a", i + 1, time, data, {3: t_event})
        summary = om.train_model(config.name)
        assert summary["calibration"][3]["n_trials"] == 6

        bundle = om.load_bundle(config.name)
        assert isinstance(bundle["calibration"][3], om.TargetCalibration)
        loader = XarrayLoader(_make_ds(4.2, seed=99))
        time, data = om.extract_features(loader, config.features)
        result = om.predict_trial(bundle, time, data)
        # The label's number is a point on the curve that was drawn for it.
        assert result.events[3].confidence == pytest.approx(om.tallest_peak(result.curves[3])[1])


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


class TestObservedSequences:
    def test_counts_orders_by_event_time(self):
        config = om.OnsetModelConfig(name="m", targets={1: "A", 2: "B"})
        trials = [
            om.TrainingTrial(np.arange(3.0), np.zeros((3, 1)), {1: 0.0, 2: 1.0}, 1.0),
            om.TrainingTrial(np.arange(3.0), np.zeros((3, 1)), {1: 0.0, 2: 1.0}, 1.0),
            om.TrainingTrial(np.arange(3.0), np.zeros((3, 1)), {2: 0.0, 1: 1.0}, 1.0),
        ]
        assert om.observed_sequences(trials, config) == {"1-2": 2, "2-1": 1}


class TestExpectations:
    """What the user declared, checked after the fact — never enforced.

    The model has no idea what order things come in; the user does. Declaring
    it turns a surprise into a flag on the trial, which the trials table can
    filter on, instead of silently moving an event.
    """

    def _config(self, order, together=False):
        config = _make_config(targets={3: "peck", 4: "land"})
        config.expected_order = order
        config.expect_together = together
        return config

    def test_nothing_declared_flags_nothing(self):
        config = _make_config(targets={3: "peck", 4: "land"})
        assert om.check_expectations({3: 5.0, 4: 1.0}, config) == []

    def test_the_declared_order_passes(self):
        assert om.check_expectations({3: 1.0, 4: 5.0}, self._config([3, 4])) == []

    def test_the_wrong_order_is_flagged(self):
        assert om.check_expectations({3: 5.0, 4: 1.0}, self._config([3, 4])) == [om.FLAG_ORDER]

    def test_a_class_left_out_is_not_an_order_problem(self):
        """One class present cannot be out of order with anything."""
        assert om.check_expectations({3: 5.0}, self._config([3, 4])) == []

    def test_one_of_a_coupled_set_implies_the_rest(self):
        config = self._config([3, 4], together=True)
        assert om.check_expectations({3: 1.0}, config) == [om.FLAG_MISSING]
        assert om.check_expectations({3: 1.0, 4: 5.0}, config) == []

    def test_a_trial_with_none_of_them_is_not_flagged(self):
        """Coupling says "one implies the rest", not "every trial has them".

        A trial where the behaviour did not happen at all is not a surprise,
        and flagging it would bury the trials that are.
        """
        assert om.check_expectations({}, self._config([3, 4], together=True)) == []

    def test_both_flags_can_land_together(self):
        config = _make_config(targets={3: "a", 4: "b", 5: "c"})
        config.expected_order = [3, 4, 5]
        config.expect_together = True
        flags = om.check_expectations({3: 5.0, 4: 1.0}, config)
        assert flags == [om.FLAG_ORDER, om.FLAG_MISSING]
        assert om.expectation_verdict(flags) == "order+missing"

    def test_the_verdict_is_a_string_the_funnel_filter_can_group(self):
        assert om.expectation_verdict([]) == om.EXPECTED_OK
        assert om.expectation_verdict([om.FLAG_ORDER]) == "order"

    def test_a_config_asking_for_the_old_sequence_model_still_loads(self):
        """`use_crf` was replaced by declarations; a stored config survives."""
        config = _make_config()
        om.save_config(config)
        path = om.model_dir(config.name) / "config.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["use_crf"] = True
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        assert om.load_config(config.name).expected_order == []

    def test_describe_reads_back_what_was_declared(self):
        assert self._config([3, 4]).describe_expectations() == "peck → land"
        assert self._config([3, 4], True).describe_expectations() == "peck → land, one implies the rest"
        assert _make_config().describe_expectations() == ""
