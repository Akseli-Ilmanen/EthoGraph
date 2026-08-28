"""``features.sin_cos`` — an angle enters a model as its (sin, cos) pair.

The two questions worth guarding: which units the angle is in (declared,
inferred, or refused), and that the columns ``extract_features`` stacks are
the ones ``enumerate_columns`` names — including through the segmentation
pipeline's layout, where the components must escape z-scoring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import ethograph as eto
from ethograph.features import columns as fc
from ethograph.io.catalog import XarrayLoader
from ethograph.labels.tsv_store import save_labels_tsv
from ethograph.segment.config import load_config
from ethograph.segment.materialise import materialise, read_layout

FS = 50.0
DURATION = 4.0


def _angle_ds(units: str | None, degrees: bool) -> xr.Dataset:
    """One trial whose ``heading`` sweeps a full turn, plus a plain feature."""
    t = np.arange(0.0, DURATION, 1.0 / FS)
    turn = np.linspace(-np.pi, np.pi, t.size, endpoint=False)
    heading = np.rad2deg(turn) if degrees else turn
    ds = xr.Dataset(
        {
            "heading": (("time", "individual"), heading[:, None]),
            "speed": (("time", "individual"), np.abs(np.sin(t * 2.0))[:, None]),
        },
        coords={"time": t, "individual": ["A"]},
        attrs={"trial": 1, "fps": FS},
    )
    if units is not None:
        ds["heading"].attrs["units"] = units
    return ds


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "expected"),
    [("rad", fc.RADIANS), ("radians", fc.RADIANS), ("deg", fc.DEGREES), ("Degrees", fc.DEGREES)],
)
def test_declared_units_decide(declared: str, expected: str):
    """Whatever the values look like, the variable's own attr wins."""
    values = np.linspace(-3.0, 3.0, 100)
    assert fc.angle_units(values, declared) == expected


def test_units_are_read_off_the_values_when_the_variable_does_not_say():
    turn = np.linspace(-np.pi, np.pi, 100)
    assert fc.angle_units(turn) == fc.RADIANS
    assert fc.angle_units(np.rad2deg(turn)) == fc.DEGREES


def test_a_non_angular_unit_is_refused():
    with pytest.raises(ValueError, match="units='m'"):
        fc.angle_units(np.zeros(10), "m", "Column 'distance'")


def test_all_nan_values_read_as_radians_rather_than_raising():
    """An angle absent for the whole trial is a gap for interpolation to fill."""
    assert fc.angle_units(np.full(10, np.nan)) == fc.RADIANS


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def test_the_components_replace_the_angle_and_carry_its_derivative():
    cols = fc.enumerate_columns(
        {"heading": {"keypoint": ["beak"]}, "speed": {}},
        derivatives=["heading"],
        sin_cos=["heading"],
    )
    assert [c.name for c in cols] == [
        "heading|keypoint=beak|sin",
        "heading|keypoint=beak|sin|d/dt",
        "heading|keypoint=beak|cos",
        "heading|keypoint=beak|cos|d/dt",
        "speed",
    ]
    assert [c.circular for c in cols] == ["sin", "sin", "cos", "cos", None]


@pytest.mark.parametrize("degrees", [False, True])
def test_extract_features_stacks_what_enumerate_columns_names(degrees: bool):
    """The two functions expand independently; a disagreement would silently
    mislabel every column downstream of it."""
    ds = _angle_ds("deg" if degrees else "rad", degrees)
    loader = XarrayLoader(ds)
    features = {"heading": {}, "speed": {}}
    time, data = fc.extract_features(loader, features, sin_cos=["heading"])

    names = [c.name for c in fc.enumerate_columns(features, sin_cos=["heading"])]
    assert names == ["heading|sin", "heading|cos", "speed"]
    assert data.shape == (len(time), 3)

    turn = np.linspace(-np.pi, np.pi, len(time), endpoint=False)
    assert np.allclose(data[:, 0], np.sin(turn))
    assert np.allclose(data[:, 1], np.cos(turn))


def test_sin_cos_naming_a_feature_that_is_not_selected_is_an_error():
    loader = XarrayLoader(_angle_ds("rad", degrees=False))
    with pytest.raises(ValueError, match="sin_cos names"):
        fc.extract_features(loader, {"speed": {}}, sin_cos=["heading"])


# ---------------------------------------------------------------------------
# Through the segmentation pipeline
# ---------------------------------------------------------------------------


def _project(tmp_path: Path, *, units: str | None, degrees: bool, sin_cos: list[str]) -> Path:
    folder = tmp_path / "session"
    folder.mkdir(parents=True, exist_ok=True)
    eto.from_datasets([_angle_ds(units, degrees)]).save(str(folder / "s.nc"))
    save_labels_tsv(folder / "s_labels.tsv", pd.DataFrame(columns=list(eto.labels.tsv_store.TSV_COLUMNS)))
    (tmp_path / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    config = {
        "sessions": [{"source": str(folder / "s.nc"), "labels_path": str(folder / "s_labels.tsv")}],
        "features": {
            "name": "ang",
            "columns": {"heading": {}, "speed": {}},
            "sin_cos": sin_cos,
            "labels": {"mapping": "mapping.txt", "branch": 0},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_materialised_layout_carries_the_components_unnormalised(tmp_path: Path):
    """sin/cos live in [-1, 1] and mean what they say there — z-scoring or
    percentile-clipping them would undo exactly what the encoding is for."""
    cfg = load_config(_project(tmp_path, units="deg", degrees=True, sin_cos=["heading"]))
    layout = read_layout(materialise(cfg))
    assert layout.names == [
        "heading|individual=self|sin",
        "heading|individual=self|cos",
        "speed|individual=self",
    ]
    assert layout.normalise == [False, False, True]


def test_sin_cos_must_name_a_selected_feature(tmp_path: Path):
    with pytest.raises(ValueError, match="features.sin_cos names"):
        load_config(_project(tmp_path, units="rad", degrees=False, sin_cos=["nope"]))
