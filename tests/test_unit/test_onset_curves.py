"""The onset-curve store: what each prediction run believed, frame by frame."""

import numpy as np

from ethograph.labels import onset_curves as oc


def _curves(n=50):
    return np.linspace(0.0, 1.0, n), {3: np.linspace(0.0, 0.9, n), 4: np.linspace(0.9, 0.0, n)}


def _write(session, timestamp, per_trial):
    return oc.write_curves(oc.run_dir(session, timestamp) / oc.CURVES_FILE, per_trial)


def test_a_run_writes_under_the_sessions_labels_folder(tmp_path):
    """The same `labels/` folder the label backups use."""
    folder = oc.run_dir(tmp_path / "Trial_data.nc", "20260824_151107")
    assert folder.parent == tmp_path / "labels"
    assert folder.name == "predictions_lightgbm_20260824_151107"


def test_provenance_lands_beside_the_curves(tmp_path):
    """A run folder says what wrote it: the model's config (file or data) and how it was applied."""
    import yaml

    folder = oc.run_dir(tmp_path / "s.nc", "20260824_120000")
    trained = tmp_path / "config.yaml"
    trained.write_text("model: {architecture: rny008_gsm}\n", encoding="utf-8")
    oc.write_provenance(folder, model_config=trained, inference={"run": "ctx2s", "epoch": 3})
    assert yaml.safe_load((folder / oc.CONFIG_FILE).read_text()) == {"model": {"architecture": "rny008_gsm"}}
    assert yaml.safe_load((folder / oc.INFERENCE_FILE).read_text()) == {"run": "ctx2s", "epoch": 3}
    oc.write_provenance(folder, model_config={"targets": {31: "a"}}, inference={"model": "lightgbm"})
    assert yaml.safe_load((folder / oc.CONFIG_FILE).read_text()) == {"targets": {31: "a"}}


def test_missing_file_reads_as_empty(tmp_path):
    assert oc.read_curves(tmp_path / "nothing.npz") == {}
    assert oc.read_all_curves(tmp_path / "Trial_data.nc") == {}


def test_roundtrip(tmp_path):
    time, curves = _curves()
    back = oc.read_curves(_write(tmp_path / "s.nc", "20260824_120000", {7: (time, curves)}))
    assert set(back) == {"7"}
    stored_time, stored_curves = back["7"]
    assert np.allclose(stored_time, time)
    assert set(stored_curves) == {3, 4}
    assert np.allclose(stored_curves[3], curves[3], atol=1e-6)


class TestManyRuns:
    """A run is filtered by the trials table and by which classes are missing,
    so no one run holds everything — reading has to add them up."""

    def test_a_later_run_adds_its_trials(self, tmp_path):
        session = tmp_path / "s.nc"
        time, curves = _curves()
        _write(session, "20260824_120000", {"1": (time, curves)})
        _write(session, "20260824_130000", {"2": (time, curves)})
        assert set(oc.read_all_curves(session)) == {"1", "2"}

    def test_a_later_run_wins_the_class_it_repredicted(self, tmp_path):
        session = tmp_path / "s.nc"
        time, _ = _curves()
        _write(session, "20260824_120000", {"1": (time, {3: np.zeros_like(time)})})
        _write(session, "20260824_130000", {"1": (time, {3: np.ones_like(time)})})
        _time, curves = oc.read_all_curves(session)["1"]
        assert np.allclose(curves[3], 1.0)

    def test_a_later_run_keeps_the_class_it_did_not_touch(self, tmp_path):
        """Predicting class 4 into a trial must not erase class 3's curve."""
        session = tmp_path / "s.nc"
        time, _ = _curves()
        _write(session, "20260824_120000", {"1": (time, {3: np.zeros_like(time)})})
        _write(session, "20260824_130000", {"1": (time, {4: np.ones_like(time)})})
        assert set(oc.read_all_curves(session)["1"][1]) == {3, 4}

    def test_runs_are_ordered_by_their_timestamp(self, tmp_path):
        session = tmp_path / "s.nc"
        time, curves = _curves()
        for stamp in ("20260824_130000", "20260824_120000", "20260901_090000"):
            _write(session, stamp, {"1": (time, curves)})
        assert [p.name.split("_", 1)[1] for p in oc.run_dirs(session)] == [
            "lightgbm_20260824_120000",
            "lightgbm_20260824_130000",
            "lightgbm_20260901_090000",
        ]

    def test_a_folder_without_curves_is_skipped(self, tmp_path):
        session = tmp_path / "s.nc"
        (oc.run_dir(session, "20260824_120000")).mkdir(parents=True)
        assert oc.run_dirs(session) == []


def test_many_trials_do_not_collide(tmp_path):
    """Key prefixes are index-based — trial 1 must not read trial 11's curves."""
    time, _ = _curves()
    path = _write(
        tmp_path / "s.nc",
        "20260824_120000",
        {str(i): (time, {3: np.full_like(time, float(i))}) for i in range(1, 13)},
    )
    back = oc.read_curves(path)
    assert len(back) == 12
    for i in range(1, 13):
        assert np.allclose(back[str(i)][1][3], float(i))


def test_unreadable_file_is_ignored(tmp_path):
    path = tmp_path / "broken.npz"
    path.write_bytes(b"not an npz")
    assert oc.read_curves(path) == {}


class TestManyModels:
    """One folder convention for every model, ordered by when it ran."""

    def test_run_dir_names_the_model(self, tmp_path):
        session = tmp_path / "s.nc"
        assert oc.run_dir(session, "20260101_000000").name == "predictions_lightgbm_20260101_000000"
        assert oc.run_dir(session, "20260101_000000", model="spot_A2").name == "predictions_spot_A2_20260101_000000"

    def test_runs_order_by_timestamp_not_by_model(self, tmp_path):
        session = tmp_path / "s.nc"
        for model, ts in (
            ("spot_A2", "20260101_000000"),
            ("lightgbm", "20260102_000000"),
            ("spot_A2", "20260103_000000"),
        ):
            oc.write_curves(oc.run_dir(session, ts, model=model) / oc.CURVES_FILE, {})
        assert [oc.run_timestamp(p) for p in oc.run_dirs(session)] == [
            "20260101_000000",
            "20260102_000000",
            "20260103_000000",
        ]
