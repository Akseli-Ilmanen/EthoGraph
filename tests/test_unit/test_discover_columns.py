"""Feature discovery: pick every feature at a target sampling rate.

One synthetic session with two features on different time dims — ``position``
on ``time`` at 50 Hz, ``slow_feat`` on ``time_aux`` at 10 Hz — checks that
:func:`discover_columns_from_source` keeps only the features matching the
requested rate, and drops the individual dim from the result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

import ethograph as eto
from ethograph.features.columns import columns_to_yaml, enumerate_columns, expand_dim_values
from ethograph.segment.sessions import discover_columns_from_source

FAST_FS = 50.0
SLOW_FS = 10.0
INDIVIDUALS = ["A", "B"]
KEYPOINTS = ["beak", "tail"]


def _make_session(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    fast_t = np.arange(0.0, 2.0, 1.0 / FAST_FS)
    slow_t = np.arange(0.0, 2.0, 1.0 / SLOW_FS)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "position": (
                ("time", "space", "keypoint", "individual"),
                rng.normal(size=(fast_t.size, 2, len(KEYPOINTS), len(INDIVIDUALS))),
            ),
            "slow_feat": (("time_aux", "keypoint"), rng.normal(size=(slow_t.size, len(KEYPOINTS)))),
        },
        coords={
            "time": fast_t,
            "time_aux": slow_t,
            "space": ["x", "y"],
            "keypoint": KEYPOINTS,
            "individual": INDIVIDUALS,
        },
        attrs={"trial": 1, "fps": FAST_FS},
    )
    dt = eto.from_datasets([ds])
    nc_path = folder / "session.nc"
    dt.save(str(nc_path))
    return nc_path


def test_discover_columns_keeps_only_the_matching_rate(tmp_path: Path) -> None:
    source = _make_session(tmp_path)

    fast_columns = discover_columns_from_source(source, fs=FAST_FS)
    assert fast_columns == {"position": {"space": ["x", "y"], "keypoint": KEYPOINTS}}

    slow_columns = discover_columns_from_source(source, fs=SLOW_FS)
    assert slow_columns == {"slow_feat": {"keypoint": KEYPOINTS}}


def test_discover_columns_respects_exclude(tmp_path: Path) -> None:
    source = _make_session(tmp_path)

    columns = discover_columns_from_source(source, fs=FAST_FS, exclude=["position"])

    assert columns == {}


def test_columns_to_yaml_is_paste_ready() -> None:
    columns = {"position": {"space": ["x", "y"], "keypoint": ["beak", "tail"]}}

    text = columns_to_yaml(columns)

    assert text == "  columns:\n    position: {space: [x, y], keypoint: [beak, tail]}\n"


def test_columns_to_yaml_collapses_contiguous_numeric_dims() -> None:
    columns = {"s3d": {"s3d_dims": [str(i) for i in range(20)]}}

    text = columns_to_yaml(columns)

    assert text == "  columns:\n    s3d: {s3d_dims: 0..19}\n"


def test_expand_dim_values_reads_back_a_range() -> None:
    assert expand_dim_values("0..19") == [str(i) for i in range(20)]
    assert expand_dim_values(["a", "b"]) == ["a", "b"]


def test_enumerate_columns_expands_a_range_dim() -> None:
    columns = enumerate_columns({"s3d": {"s3d_dims": "0..2"}})

    assert [c.selections["s3d_dims"] for c in columns] == ["0", "1", "2"]
