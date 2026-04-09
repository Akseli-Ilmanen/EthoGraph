"""Integration tests for DANDI remote NWB streaming.

These tests require network access and are slow (~30-60s).
They are skipped by default; run with::

    pytest -m slow tests/test_dandi_streaming.py
"""

import pytest

pytestmark = pytest.mark.slow

DANDISET_ID = "000409"
SESSION_EID = "64e3fb86-928c-4079-865c-b364205b502e"
# The processed NWB asset for this session (IBL – Brainwide Map)
EXPECTED_ASSET_ID = "196aa923-79c8-4524-a1b2-344fc30d8cb2"
EXPECTED_POSE_CAMERAS = {"RightCamera", "LeftCamera", "BodyCamera"}


@pytest.fixture(scope="module")
def remote_nwb():
    """Open the example DANDI NWB file once for the whole module."""
    from ethograph.utils.dandi import open_nwb_dandi

    nwb_obj, nwb_io, h5_file, rf = open_nwb_dandi(DANDISET_ID, EXPECTED_ASSET_ID)
    yield nwb_obj, h5_file
    for closeable in (nwb_io, h5_file, rf):
        if closeable is not None:
            try:
                closeable.close()
            except Exception:
                pass


def test_remote_nwb_has_trials(remote_nwb):
    """The example session has 533 trials with real timing."""
    nwb_obj, _ = remote_nwb
    assert nwb_obj.trials is not None
    assert len(nwb_obj.trials) > 500




def test_trial_alignment_from_remote(remote_nwb):
    """NWBAlignment.from_nwb_object returns correct trial start/stop times."""
    from ethograph.io.nwb_alignment import NWBAlignment

    nwb_obj, _ = remote_nwb
    sio = NWBAlignment.from_nwb_object(nwb_obj)
    df = sio.trials_df
    assert not df.empty
    assert len(df) > 500

    # Trial 1 should have real timing (not 0.0 / None)
    start = sio.start_time(1)
    stop = sio.stop_time(1)
    assert start > 0, f"Trial 1 start_time should be >0, got {start}"
    assert stop is not None, "Trial 1 stop_time should not be None"
    assert stop > start, f"stop ({stop}) should be > start ({start})"

    # Trial 2 should differ from trial 1
    start2 = sio.start_time(2)
    assert start2 > start, "Trial 2 should start after trial 1"


def test_nwb_catalog_and_loader(remote_nwb):
    """NWBLoader can be constructed from remote h5py handle and has features."""
    from ethograph.io.catalog import NWBLoader, catalog_from_nwb, read_trial_intervals

    _, h5_file = remote_nwb
    source = h5_file

    catalog, combo_cat = catalog_from_nwb(source)
    trial_intervals = read_trial_intervals(source)

    assert len(catalog.features) > 10, f"Expected >10 features, got {len(catalog.features)}"
    assert len(trial_intervals) > 500, f"Expected >500 trials, got {len(trial_intervals)}"

    loader = NWBLoader(source, catalog, combo_catalog=combo_cat)
    loader.set_trial_intervals(trial_intervals)
    assert loader.backend == "nwb"

    # Verify data can be sliced for a trial
    loader.set_trial(0)
    t0, t1 = loader.trial_bounds
    assert t1 > t0, f"Trial bounds should be valid, got ({t0}, {t1})"


def test_full_project_load(tmp_path):
    """End-to-end: create alignment.nwb with provenance, load, verify alignment + cameras."""
    import pandas as pd

    from ethograph.io.data_loader import _load_remote_nwb
    from ethograph.labels.tsv_store import init_empty_labels, save_labels_tsv
    from ethograph.utils.nwb import create_alignment_from_streams

    project_dir = tmp_path / "test_session"
    ethograph_dir = project_dir / ".ethograph"
    ethograph_dir.mkdir(parents=True)

    # Simulate video_info as stored by the wizard
    video_info = {
        "VideoBodyCamera": {
            "url": "https://dandiarchive.s3.amazonaws.com/blobs/cfa/32c/cfa32c9a-2d2d-4489-bb29-30c7a20fa207",
            "start": 6.577, "end": 4030.423, "fps": 30.0,
        },
        "VideoLeftCamera": {
            "url": "https://dandiarchive.s3.amazonaws.com/blobs/390/b57/390b57b1-8f79-4d19-84ae-6c5d4b20127a",
            "start": 6.533, "end": 4030.406, "fps": 60.0,
        },
        "VideoRightCamera": {
            "url": "https://dandiarchive.s3.amazonaws.com/blobs/f08/c4c/f08c4cf3-729c-4505-8241-2415ddb951a4",
            "start": 6.500, "end": 4030.430, "fps": 150.0,
        },
    }

    # Build alignment.nwb with video URLs as session-wide streams
    # Use a minimal trials table (real trials come from remote NWB at load time)
    trials_df = pd.DataFrame({
        "trial": [1, 2, 3],
        "start_time": [10.0, 20.0, 30.0],
        "stop_time": [18.0, 28.0, 38.0],
    })
    streams = []
    for vname, info in video_info.items():
        streams.append({
            "name": f"video_{vname}",
            "files": [info["url"]],
            "rate": info["fps"],
            "starting_time": info["start"],
        })

    provenance = {
        "nwb_dandiset_id": DANDISET_ID,
        "nwb_asset_id": EXPECTED_ASSET_ID,
        "nwb_pose_keys": list(EXPECTED_POSE_CAMERAS),
    }
    create_alignment_from_streams(
        trials_df, streams, ethograph_dir / "alignment.nwb", provenance=provenance,
    )
    save_labels_tsv(project_dir / "labels.tsv", init_empty_labels(["1", "2", "3"]))

    result = _load_remote_nwb(str(project_dir))

    # Core assertions
    assert result.data_loader is not None
    assert result.data_loader.backend == "nwb"
    assert len(result.trial_ids) > 500
    assert result.nwb_local is None

    # Alignment: real trial timing from remote NWB
    sio = result.nwb_alignment
    assert sio is not None
    assert not sio.trials_df.empty
    start = sio.start_time(1)
    stop = sio.stop_time(1)
    assert start > 0, f"Trial 1 start should be >0, got {start}"
    assert stop is not None and stop > start

    # Cameras: registered from acquisition ImageSeries
    assert set(sio.cameras) == set(video_info.keys()), (
        f"Expected cameras {set(video_info.keys())}, got {set(sio.cameras)}"
    )
    # Video URL accessible via resolve_media_path
    url = sio.resolve_media_path(1, "video", device="VideoBodyCamera")
    assert url and url.startswith("https://"), f"Expected DANDI URL, got {url}"

    # Stream offset: video starts before trial 1 → negative offset
    offset = sio.stream_offset_for_trial(1, "video", "VideoBodyCamera")
    assert offset < 0, f"Video starts before trial 1, offset should be <0, got {offset}"

    # Provenance round-trip
    prov = sio.provenance
    assert prov is not None
    assert prov["nwb_dandiset_id"] == DANDISET_ID
    assert prov["nwb_asset_id"] == EXPECTED_ASSET_ID
    assert set(prov["nwb_pose_keys"]) == EXPECTED_POSE_CAMERAS

    # Source collection
    sc = result.source_collection
    assert sc is not None
    assert sc.n_trials > 500
