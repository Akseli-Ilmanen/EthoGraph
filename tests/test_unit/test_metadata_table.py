"""Tests for the metadata table system."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from ethograph.io.metadata_table import (
    condition_columns,
    empty_metadata_df,
    load_metadata_df,
    metadata_from_intervalset,
    metadata_from_nwb_trials,
    load_metadata_tsv,
    save_metadata_tsv,
    metadata_tsv_path,
)


def test_condition_columns():
    df = pd.DataFrame({
        "trial": [1, 2],
        "genotype": ["WT", "KO"],
        "start_time": [0.0, 1.0],
        "video_cam-1": ["a.mp4", "b.mp4"],
    })
    cols = condition_columns(df)
    assert cols == ["genotype"]
    assert "trial" not in cols





def test_save_load_roundtrip():
    df = pd.DataFrame({
        "trial": [1, 2, 3],
        "genotype": ["WT", "KO", "WT"],
        "dose_mg": [0.0, 5.0, 10.0],
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_metadata.tsv"
        save_metadata_tsv(path, df)
        loaded = load_metadata_tsv(path)
        assert list(loaded.columns) == ["trial", "genotype", "dose_mg"]
        assert len(loaded) == 3
        assert loaded.loc[1, "genotype"] == "KO"


def test_metadata_tsv_path():
    p = metadata_tsv_path("/data/experiment.nc")
    assert p.name == "experiment_metadata.tsv"


def test_empty_metadata_df():
    mdf = empty_metadata_df([1])
    assert "trial" in mdf.columns


def test_load_metadata_df_empty_fallback():
    mdf, source = load_metadata_df(trial_ids=[1, 2])
    assert source is None
    assert list(mdf["trial"]) == [1, 2]


def test_metadata_from_nwb_trials_filters_infrastructure():
    trials_df = pd.DataFrame({
        "trial": [1, 2],
        "start_time": [0.0, 1.0],
        "stop_time": [0.5, 1.5],
        "genotype": ["WT", "KO"],
        "video_cam-1": ["a.mp4", "b.mp4"],
    })
    mdf = metadata_from_nwb_trials(trials_df)
    assert list(mdf.columns) == ["trial", "genotype"]
    assert list(mdf["genotype"]) == ["WT", "KO"]


def test_metadata_from_intervalset():
    nap = pytest.importorskip("pynapple")

    intervals = nap.IntervalSet(
        start=[0.0, 1.0],
        end=[0.5, 1.5],
        metadata=pd.DataFrame({
            "trial": [1, 2],
            "condition": ["A", "B"],
            "start_time": [0.0, 1.0],
        }),
    )
    mdf = metadata_from_intervalset(intervals)
    assert list(mdf.columns) == ["trial", "condition"]
    assert list(mdf["condition"]) == ["A", "B"]


def test_load_metadata_df_uses_nwb_source_when_trials_present(monkeypatch, tmp_path):
    nwb_path = tmp_path / "session.nwb"
    nwb_path.write_text("placeholder", encoding="utf-8")

    class _FakeAlignment:
        def __init__(self):
            self.trials_df = pd.DataFrame({
                "trial": [1, 2],
                "genotype": ["WT", "KO"],
                "start_time": [0.0, 1.0],
                "stop_time": [0.5, 1.5],
            })

    monkeypatch.setattr("ethograph.io.metadata_table.make_nwb_alignment", lambda _: _FakeAlignment())

    mdf, source = load_metadata_df(source_path=nwb_path)

    assert source == str(nwb_path)
    assert list(mdf.columns) == ["trial", "genotype"]
    assert list(mdf["trial"]) == [1, 2]
