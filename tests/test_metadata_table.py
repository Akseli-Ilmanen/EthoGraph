"""Tests for the metadata table system."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ethograph.io.metadata_table import (
    condition_columns,
    load_metadata_tsv,
    save_metadata_tsv,
    metadata_tsv_path,
)
from ethograph.io.trialtree import TrialTree


def _make_dt_with_conditions():
    time1 = np.arange(3, dtype=float) / 30.0
    ds1 = xr.Dataset({"x": xr.DataArray([1, 2, 3], dims="time", coords={"time": time1})})
    ds1.attrs["trial"] = 1
    ds1.attrs["fps"] = 30
    ds1.attrs["genotype"] = "WT"
    ds1.attrs["treatment"] = "saline"

    time2 = np.arange(3, dtype=float) / 30.0
    ds2 = xr.Dataset({"x": xr.DataArray([4, 5, 6], dims="time", coords={"time": time2})})
    ds2.attrs["trial"] = 2
    ds2.attrs["fps"] = 30
    ds2.attrs["genotype"] = "KO"
    ds2.attrs["treatment"] = "drug_A"

    return TrialTree.from_datasets([ds1, ds2], validate=False)




def test_condition_columns():
    df = pd.DataFrame({"trial": [1, 2], "genotype": ["WT", "KO"]})
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
    time = np.arange(2, dtype=float) / 30.0
    ds = xr.Dataset({"x": xr.DataArray([1, 2], dims="time", coords={"time": time})})
    ds.attrs["trial"] = 1
    ds.attrs["fps"] = 30
    dt = TrialTree.from_datasets([ds], validate=False)
    # No metadata set → should return DataFrame with just trial column
    mdf = dt.metadata_df
    assert "trial" in mdf.columns
    assert dt.get_trial_metadata(1) == {}
