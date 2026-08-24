"""``features.changepoint_features`` — expanding named raw changepoint masks
at ``open_session`` time so ``more_changepoint_features``'s proximity/
segment-ID columns are auto-merged into ``features.columns`` without the
user spelling out every derived name. Xarray-only; pynapple sessions must
refuse it clearly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import ethograph as eto
from ethograph.io import schema
from ethograph.labels.tsv_store import save_labels_tsv
from ethograph.segment.config import load_config
from ethograph.segment.materialise import materialise, read_index, read_layout
from ethograph.segment.sessions import open_session

FS = 50.0
DURATION = 4.0


def _empty_labels() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trial",
            "individual",
            "individual_rec",
            "labels",
            "onset_s",
            "offset_s",
            "event_type",
            "confidence",
            "labeling_method",
            "changepoint_corrected",
            "prediction_source",
            "n_samples",
        ]
    )


def _session_with_changepoints(folder: Path, *, legacy: bool = False) -> Path:
    """One trial with ``speed`` + its ``speed_troughs`` mask.

    ``legacy=True`` writes the pre-schema ``attrs["type"] = "changepoints"``
    spelling most real session files still carry.
    """
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    speed = np.abs(np.sin(t * 2.0))[:, None]
    mask = np.zeros(t.size, dtype=np.int8)[:, None]
    mask[[40, 120, 160], 0] = 1
    ds = xr.Dataset(
        {"speed": (("time", "individual"), speed), "speed_troughs": (("time", "individual"), mask)},
        coords={"time": t, "individual": ["A"]},
        attrs={"trial": 1, "fps": FS},
    )
    ds["speed_troughs"].attrs = (
        {"type": "changepoints", "target_feature": "speed"}
        if legacy
        else schema.changepoint_attrs(target_feature="speed")
    )
    dt = eto.from_datasets([ds])
    nc_path = folder / "cp.nc"
    dt.save(str(nc_path))
    save_labels_tsv(folder / "cp_labels.tsv", _empty_labels())
    return nc_path


def _session_with_two_changepoint_masks(folder: Path) -> Path:
    """``speed`` plus two masks over it, firing on different frames."""
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    speed = np.abs(np.sin(t * 2.0))[:, None]
    troughs = np.zeros(t.size, dtype=np.int8)[:, None]
    troughs[[40, 160], 0] = 1
    peaks = np.zeros(t.size, dtype=np.int8)[:, None]
    peaks[[90, 120], 0] = 1
    ds = xr.Dataset(
        {
            "speed": (("time", "individual"), speed),
            "speed_troughs": (("time", "individual"), troughs),
            "speed_peaks": (("time", "individual"), peaks),
        },
        coords={"time": t, "individual": ["A"]},
        attrs={"trial": 1, "fps": FS},
    )
    for var in ("speed_troughs", "speed_peaks"):
        ds[var].attrs = schema.changepoint_attrs(target_feature="speed")
    dt = eto.from_datasets([ds])
    nc_path = folder / "cp2.nc"
    dt.save(str(nc_path))
    save_labels_tsv(folder / "cp_labels.tsv", _empty_labels())  # the name _write_config expects
    return nc_path


def _write_config(
    root: Path,
    nc_path: Path,
    *,
    columns: dict | None = None,
    changepoint_features: dict | None = None,
) -> Path:
    (root / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    cp_cfg = (
        {"sigmas": [2.0], "inputs": {"speed_troughs": {}}} if changepoint_features is None else changepoint_features
    )
    config = {
        "sessions": [{"source": str(nc_path), "labels_path": str(nc_path.with_name("cp_labels.tsv"))}],
        "features": {
            "name": "cp",
            "columns": {"speed": {}} if columns is None else columns,
            "labels": {"mapping": "mapping.txt", "branch": 0},
            "changepoint_features": cp_cfg,
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_open_session_migrates_legacy_attrs_before_expanding(tmp_path: Path):
    """Real session files predate the variable-schema convention and still
    spell a changepoint mask ``attrs["type"] = "changepoints"``; nothing
    migrates that automatically on load, so the expansion must do it itself."""
    nc_path = _session_with_changepoints(tmp_path / "session", legacy=True)
    config_path = _write_config(tmp_path, nc_path)
    cfg = load_config(config_path)
    session = open_session(cfg.sessions[0], cfg)
    trial_ds = session.trial_dataset(1)
    assert "speed_troughs_cp_sigma2" in trial_ds.data_vars
    assert trial_ds["speed_troughs_cp_sigma2"].attrs["normalise"] == 0


def test_open_session_expands_configured_changepoints(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(tmp_path, nc_path)
    cfg = load_config(config_path)
    session = open_session(cfg.sessions[0], cfg)
    trial_ds = session.trial_dataset(1)
    assert "speed_troughs_cp_sigma2" in trial_ds.data_vars
    assert "speed_troughs_cp_binary" in trial_ds.data_vars
    assert "speed_troughs_cp_segment_id" in trial_ds.data_vars
    assert trial_ds["speed_troughs_cp_sigma2"].attrs["normalise"] == 0
    assert trial_ds["speed_troughs_cp_sigma2"].attrs[schema.KIND] == schema.CHANGEPOINT_FEATURE
    # the raw mask stays the only changepoint variable — expansions are ordinary features
    assert schema.changepoint_vars(trial_ds) == ["speed_troughs"]


def test_inputs_generate_columns_without_spelling_them_out(tmp_path: Path):
    """``inputs`` + ``transforms`` are enough — no derived name in ``features.columns``."""
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={},
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["proximity"]},
    )
    cfg = load_config(config_path)
    assert cfg.features.columns == {"speed_troughs_cp_sigma2": {}}
    data_dir = materialise(cfg)
    layout = read_layout(data_dir)
    assert layout.names == ["speed_troughs_cp_sigma2|individual=self"]
    assert layout.normalise == [False]


def test_normalise_zero_columns_are_not_percentile_clipped(tmp_path: Path):
    """A sparse mask clipped to its 2nd/98th percentile would become a constant.

    ``normalise=0`` says the column's values already mean what they say, so it
    gates clipping as well as z-scoring: the mask keeps its ones and the
    proximity feature keeps its peak, while an ordinary column is still clipped.
    """
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={"speed": {}},
        changepoint_features={
            "sigmas": [2.0],
            "inputs": {"speed_troughs": {}},
            "transforms": ["binary", "proximity"],
        },
    )
    cfg = load_config(config_path)
    data_dir = materialise(cfg)
    layout = read_layout(data_dir)
    key = read_index(data_dir)["key"].iloc[0]
    x = np.load(data_dir / "features" / f"{key}.npy")
    column = {name.split("|")[0]: x[i] for i, name in enumerate(layout.names)}

    assert column["speed_troughs_cp_binary"].max() == pytest.approx(1.0)
    assert column["speed_troughs_cp_binary"].sum() == pytest.approx(3.0)
    assert column["speed_troughs_cp_sigma2"].max() == pytest.approx(1.0)

    raw_speed = eto.open(str(nc_path)).trial(1)["speed"].sel(individual="A").values
    assert column["speed"].max() < raw_speed.max()


def test_merge_expands_one_block_for_all_masks(tmp_path: Path):
    """``merge: true`` ORs the named masks into one and expands that once."""
    nc_path = _session_with_two_changepoint_masks(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={},
        changepoint_features={
            "sigmas": [2.0],
            "inputs": {"speed_troughs": {}, "speed_peaks": {}},
            "transforms": ["binary"],
            "merge": True,
        },
    )
    cfg = load_config(config_path)
    assert cfg.features.columns == {"changepoints_cp_binary": {}}

    data_dir = materialise(cfg)
    layout = read_layout(data_dir)
    # the merge collapses the keypoint-like dims but leaves the individual
    # standing, so the sample still pins it to itself
    assert layout.names == ["changepoints_cp_binary|individual=self"]
    key = read_index(data_dir)["key"].iloc[0]
    merged = np.load(data_dir / "features" / f"{key}.npy")[0]
    # the union of both masks' frames, each mask contributing its own
    assert np.flatnonzero(merged).tolist() == [40, 90, 120, 160]


def test_merge_refuses_a_variable_that_is_not_a_mask(tmp_path: Path):
    nc_path = _session_with_two_changepoint_masks(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        changepoint_features={"sigmas": [2.0], "inputs": {"speed": {}}, "merge": True},
    )
    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="not changepoint masks"):
        open_session(cfg.sessions[0], cfg)


def test_transforms_filters_which_columns_are_generated(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={},
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["segment_id"]},
    )
    cfg = load_config(config_path)
    assert cfg.features.columns == {"speed_troughs_cp_segment_id": {}}
    session = open_session(cfg.sessions[0], cfg)
    trial_ds = session.trial_dataset(1)
    assert "speed_troughs_cp_segment_id" in trial_ds.data_vars
    assert "speed_troughs_cp_binary" not in trial_ds.data_vars
    assert "speed_troughs_cp_sigma2" not in trial_ds.data_vars


def test_a_saved_config_reloads(tmp_path: Path):
    """`save_config` -> `load_config` is what a run directory does on every infer.

    The generated columns must not be dumped as explicit entries, or reloading
    would read them as colliding with the expansion that produced them.
    """
    from ethograph.segment.config import save_config

    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={"speed": {}},
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["proximity"]},
    )
    cfg = load_config(config_path)
    reloaded = load_config(save_config(cfg, tmp_path / "run" / "config.yaml"))
    assert reloaded.features.columns == cfg.features.columns
    assert "speed_troughs_cp_sigma2" in reloaded.features.columns


def test_explicit_column_colliding_with_generated_is_an_error(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={"speed_troughs_cp_sigma2": {}},
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["proximity"]},
    )
    with pytest.raises(ValueError, match="also generates"):
        load_config(config_path)


def test_changepoint_features_requires_inputs(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(tmp_path, nc_path, changepoint_features={"sigmas": [2.0]})
    with pytest.raises(ValueError, match="must name at least one changepoint variable"):
        load_config(config_path)


def test_changepoint_features_rejects_unknown_transform(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["bogus"]},
    )
    with pytest.raises(ValueError, match="must be a subset of"):
        load_config(config_path)


def test_expansion_is_required_to_be_configured(tmp_path: Path):
    """Without ``features.changepoint_features`` the expanded column simply doesn't exist."""
    nc_path = _session_with_changepoints(tmp_path / "session")
    (tmp_path / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    config = {
        "sessions": [{"source": str(nc_path), "labels_path": str(nc_path.with_name("cp_labels.tsv"))}],
        "features": {
            "name": "cp",
            "columns": {"speed_troughs_cp_sigma2": {}},
            "labels": {"mapping": "mapping.txt", "branch": 0},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="not available in this session"):
        materialise(cfg)


def test_pynapple_session_refuses_changepoint_expansion(tmp_path: Path):
    nap = pytest.importorskip("pynapple")
    from ethograph.io.nwb_alignment import alignment_from_trials_ep

    folder = tmp_path / "pynapple_session"
    folder.mkdir(parents=True)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    nap.Tsd(t=t, d=np.abs(np.sin(t * 3))).save(str(folder / "speed.npz"))
    group = nap.TsGroup({0: nap.Ts(t=np.array([1.0, 2.0]))})
    group.set_info(source_label=["speed"], **schema.changepoint_metadata(1, target_feature="speed"))
    group.save(str(folder / "speed_troughs.npz"))
    ep = nap.IntervalSet(start=[0.0], end=[DURATION - 1 / FS])
    ep.set_info(trial=[1])
    alignment_from_trials_ep(ep, folder / ".ethograph" / "alignment.nwb")
    save_labels_tsv(folder / "labels.tsv", _empty_labels())

    (tmp_path / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    config = {
        "sessions": [{"source": str(folder), "labels_path": str(folder / "labels.tsv")}],
        "features": {
            "name": "cp",
            "columns": {"speed": {}},
            "labels": {"mapping": "mapping.txt", "branch": 0},
            "changepoint_features": {"sigmas": [2.0], "inputs": {"speed_troughs": {}}},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="only implemented for xarray sessions"):
        open_session(cfg.sessions[0], cfg)
