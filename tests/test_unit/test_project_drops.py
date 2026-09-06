"""Drops made inside a project are kept and can be reopened.

The record and the restore must agree on which fields make a drop loadable
again, and nothing but this contract forces them to.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from ethograph.gui.project import (
    DROP_RECORD_FILENAME,
    DROP_STATE_FIELDS,
    DropRecord,
    list_drops,
    new_drop_dir,
    record_drop,
    restore_drop,
)


def _state(**overrides) -> SimpleNamespace:
    base = {name: None for name in DROP_STATE_FIELDS}
    base["image_paths"] = []
    base["extra_cameras"] = []
    base["metadata_path"] = "stale.tsv"
    base.update(overrides)
    return SimpleNamespace(**base)


class TestDropDir:
    def test_is_timestamped_under_sessions(self, tmp_path: Path):
        d = new_drop_dir(tmp_path, datetime(2026, 9, 6, 21, 47, 12))
        assert d == tmp_path / "sessions" / "2026-09-06_21-47-12"
        assert d.is_dir()

    def test_same_second_never_collides(self, tmp_path: Path):
        when = datetime(2026, 9, 6, 21, 47, 12)
        a = new_drop_dir(tmp_path, when)
        b = new_drop_dir(tmp_path, when)
        assert a != b and b.is_dir()


class TestRecord:
    def test_round_trip_restores_every_field(self, tmp_path: Path):
        drop = new_drop_dir(tmp_path, datetime(2026, 9, 6, 21, 47, 12))
        state = _state(
            nc_file_path=str(drop / "alignment-abc.tmp.nwb"),
            nwb_file_path=str(drop / "alignment-abc.tmp.nwb"),
            video_folder="D:/raw",
            image_paths=["D:/raw/still.png"],
            primary_camera="cam-1",
            extra_cameras=["cam-2"],
            source_software="DeepLabCut",
        )
        record = record_drop(drop, ["D:/raw/cam0.mp4", "D:/raw/cam1.mp4"], state)
        assert (drop / DROP_RECORD_FILENAME).is_file()

        fresh = _state(video_folder="E:/elsewhere", metadata_path="stale.tsv")
        restore_drop(DropRecord.load(drop), fresh)
        for name in DROP_STATE_FIELDS:
            assert getattr(fresh, name) == getattr(state, name), name
        assert fresh.metadata_path is None
        assert record.title == "2026-09-06 21:47 — cam0.mp4, cam1.mp4"

    def test_a_field_missing_from_an_old_record_restores_as_none(self, tmp_path: Path):
        drop = new_drop_dir(tmp_path, datetime(2026, 9, 6, 21, 47, 12))
        DropRecord(created=drop.name, files=[], state={"nc_file_path": "x.nc"}).save(drop)
        fresh = _state(video_folder="E:/elsewhere")
        restore_drop(DropRecord.load(drop), fresh)
        assert fresh.nc_file_path == "x.nc"
        assert fresh.video_folder is None


class TestListing:
    def test_newest_first_and_unrecorded_folders_skipped(self, tmp_path: Path):
        old = new_drop_dir(tmp_path, datetime(2026, 9, 5, 8, 0, 0))
        new = new_drop_dir(tmp_path, datetime(2026, 9, 6, 8, 0, 0))
        new_drop_dir(tmp_path, datetime(2026, 9, 7, 8, 0, 0))  # a drop that failed before its record
        record_drop(old, ["a.mp4"], _state())
        record_drop(new, ["b.mp4"], _state())
        assert [folder for folder, _ in list_drops(tmp_path)] == [new, old]

    def test_no_project_sessions_is_empty(self, tmp_path: Path):
        assert list_drops(tmp_path / "nowhere") == []
