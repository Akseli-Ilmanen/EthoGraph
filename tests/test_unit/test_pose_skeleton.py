"""Tests for the NWB config layer of the skeleton module.

These cover only the NWB->config adapter and its integration with the existing
``PrecomputedRenderer`` — the rendering/state code itself is unchanged and
covered by ``test_skeleton.py``.
"""

import numpy as np
import xarray as xr

from ethograph.skeleton.config import config_to_arrays, nwb_skeleton_to_config
from ethograph.skeleton.renderers import PrecomputedRenderer


def test_nwb_skeleton_to_config_basic():
    nodes = ["nose", "ear_left", "ear_right", "tail_base"]
    edges = np.array([[0, 1], [0, 2], [0, 3]])

    config = nwb_skeleton_to_config(nodes, edges)

    assert config["keypoints"] == nodes
    assert len(config["connections"]) == 3
    assert config["connections"][0]["start"] == "nose"
    assert config["connections"][0]["end"] == "ear_left"
    # Each connection carries a hex color and width.
    assert config["connections"][0]["color"].startswith("#")
    assert config["connections"][0]["width"] == 2.0


def test_nwb_skeleton_to_config_drops_unmapped_nodes():
    # A None node (absent from the dataset) drops any edge touching it.
    nodes = ["nose", None, "tail_base"]
    edges = np.array([[0, 1], [0, 2]])

    config = nwb_skeleton_to_config(nodes, edges)

    assert config["keypoints"] == ["nose", "tail_base"]
    assert len(config["connections"]) == 1
    assert config["connections"][0] == {
        "start": "nose",
        "end": "tail_base",
        "color": config["connections"][0]["color"],
        "width": 2.0,
        "segment": "",
    }


def test_nwb_config_feeds_existing_renderer():
    # End-to-end: NWB nodes/edges -> config -> existing renderer.
    keypoints = ["nose", "neck", "tail_base"]
    edges = np.array([[0, 1], [1, 2]])
    config = nwb_skeleton_to_config(keypoints, edges)

    n_frames = 4
    position = np.random.rand(n_frames, 2, len(keypoints), 1) * 100
    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(position, dims=["time", "space", "keypoints", "individuals"]),
        },
        coords={
            "time": np.arange(n_frames),
            "space": ["x", "y"],
            "keypoints": keypoints,
            "individuals": ["ind_0"],
        },
        attrs={"ds_type": "poses"},
    )

    connections, colors, widths, _ = config_to_arrays(config, keypoints)
    renderer = PrecomputedRenderer(ds, connections, colors, widths)
    renderer.prepare()

    # 4 frames * 1 individual * 2 connections = 8 vectors, napari (N, 2, 3).
    assert renderer.vectors.shape == (8, 2, 3)
    assert connections == [(0, 1), (1, 2)]
