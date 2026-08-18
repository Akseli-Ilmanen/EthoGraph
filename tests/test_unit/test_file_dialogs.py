"""Browse dialogs remember where the user last was (ethograph.gui.file_dialogs)."""

from pathlib import Path

import pytest

from ethograph.gui.file_dialogs import browse_start_dir, remember_browse_dir


class _State:
    """Stand-in for ObservableAppState — only last_browse_dir matters here."""

    def __init__(self, last_browse_dir: str | None = None):
        self.last_browse_dir = last_browse_dir


def test_start_dir_empty_without_anything_remembered():
    assert browse_start_dir(_State()) == ""


def test_start_dir_uses_remembered_folder(tmp_path: Path):
    state = _State(str(tmp_path))
    assert browse_start_dir(state) == str(tmp_path)


def test_start_dir_skips_a_folder_that_no_longer_exists(tmp_path: Path):
    state = _State(str(tmp_path / "moved_to_another_drive"))
    assert browse_start_dir(state) == ""


def test_preferred_dir_wins_over_the_remembered_one(tmp_path: Path):
    remembered = tmp_path / "old"
    preferred = tmp_path / "current"
    remembered.mkdir()
    preferred.mkdir()
    state = _State(str(remembered))
    assert browse_start_dir(state, preferred) == str(preferred)


def test_preferred_file_resolves_to_its_folder(tmp_path: Path):
    session = tmp_path / "session.nc"
    session.write_bytes(b"")
    assert browse_start_dir(_State(), session) == str(tmp_path)


def test_missing_preferred_dir_falls_back_to_the_remembered_one(tmp_path: Path):
    remembered = tmp_path / "old"
    remembered.mkdir()
    state = _State(str(remembered))
    assert browse_start_dir(state, tmp_path / "gone.nc") == str(remembered)


def test_remember_stores_the_folder_of_a_picked_file(tmp_path: Path):
    session = tmp_path / "session.nc"
    session.write_bytes(b"")
    state = _State()
    remember_browse_dir(state, str(session))
    assert state.last_browse_dir == str(tmp_path)


def test_remember_stores_a_picked_folder_as_is(tmp_path: Path):
    state = _State()
    remember_browse_dir(state, tmp_path)
    assert state.last_browse_dir == str(tmp_path)


@pytest.mark.parametrize("cancelled", ["", None])
def test_a_cancelled_dialog_leaves_the_memory_alone(tmp_path: Path, cancelled):
    state = _State(str(tmp_path))
    remember_browse_dir(state, cancelled)
    assert state.last_browse_dir == str(tmp_path)
