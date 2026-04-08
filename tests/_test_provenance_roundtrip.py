"""Ad-hoc test: provenance round-trip through alignment.nwb."""
import tempfile
from pathlib import Path

import pandas as pd

from ethograph.io.nwb_alignment import NWBAlignment
from ethograph.utils.nwb import create_alignment_from_streams


def test_provenance_roundtrip(tmp_path):
    trials = pd.DataFrame(
        {"trial": [1, 2], "start_time": [0.0, 10.0], "stop_time": [8.0, 18.0]}
    )
    streams = [
        {
            "name": "video_cam1",
            "files": ["https://example.com/vid.mp4"],
            "rate": 30.0,
            "starting_time": 0.0,
        }
    ]
    provenance = {
        "nwb_dandiset_id": "000409",
        "nwb_asset_id": "abc123",
        "nwb_pose_keys": ["cam1"],
    }

    nwb_path = tmp_path / "alignment.nwb"
    create_alignment_from_streams(trials, streams, nwb_path, provenance=provenance)

    sio = NWBAlignment(nwb_path)

    # Provenance round-trip
    assert sio.provenance == provenance, f"Mismatch: {sio.provenance}"

    # Camera discovery from acquisition
    assert sio.cameras == ["cam1"], f"cameras: {sio.cameras}"

    # URL returned from resolve_media_path
    url = sio.resolve_media_path(1, "video", device="cam1")
    assert url == "https://example.com/vid.mp4", f"url: {url}"

    # Stream offset: video starts at 0, trial 1 starts at 0
    offset = sio.stream_offset_for_trial(1, "video", "cam1")
    assert offset == 0.0, f"offset: {offset}"

    # Stream offset: video starts at 0, trial 2 starts at 10
    offset2 = sio.stream_offset_for_trial(2, "video", "cam1")
    assert offset2 < 0, f"offset2: {offset2}"

    sio.close()


def test_from_nwb_object(tmp_path):
    trials = pd.DataFrame(
        {"trial": [1, 2], "start_time": [5.0, 20.0], "stop_time": [15.0, 30.0]}
    )
    streams = []
    provenance = {"nwb_dandiset_id": "test"}
    nwb_path = tmp_path / "alignment.nwb"
    create_alignment_from_streams(trials, streams, nwb_path, provenance=provenance)

    from pynwb import NWBHDF5IO

    io = NWBHDF5IO(str(nwb_path), "r")
    nwb_obj = io.read()

    sio = NWBAlignment.from_nwb_object(nwb_obj)
    assert not sio.trials_df.empty
    assert sio.start_time(1) == 5.0
    assert sio.stop_time(1) == 15.0
    assert sio.provenance == {"nwb_dandiset_id": "test"}

    io.close()
