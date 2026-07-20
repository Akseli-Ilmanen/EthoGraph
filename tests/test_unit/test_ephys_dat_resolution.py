"""Locating a Kilosort folder's raw recording when params.py is stale."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ethograph.gui.widgets_ephys import EphysWidget


def _resolve(ks_folder: Path, params: dict | None, ephys_path: str | None = None) -> Path | None:
    widget = EphysWidget.__new__(EphysWidget)
    widget._kilosort_params = None
    object.__setattr__(widget, "app_state", SimpleNamespace(ephys_path=ephys_path))
    return EphysWidget._resolve_dat_path(widget, ks_folder, params)


@pytest.fixture
def session(tmp_path: Path) -> Path:
    """A recording folder with the Kilosort output in a subfolder."""
    ks = tmp_path / "kilosort4"
    ks.mkdir()
    (tmp_path / "amplifier.dat").write_bytes(b"\x00" * 4096)
    (tmp_path / "digitalin.dat").write_bytes(b"\x00" * 16)
    return tmp_path


def test_absolute_path_from_another_machine_falls_back_to_parent(session):
    params = {"dat_path": r"F:\sorted\elsewhere\amplifier.dat"}
    assert _resolve(session / "kilosort4", params) == session / "amplifier.dat"


def test_existing_absolute_path_is_used_as_is(session):
    actual = session / "amplifier.dat"
    assert _resolve(session / "kilosort4", {"dat_path": str(actual)}) == actual


def test_raw_file_beside_kilosort_output_wins_over_parent(session):
    inner = session / "kilosort4" / "amplifier.dat"
    inner.write_bytes(b"\x00" * 32)
    params = {"dat_path": r"F:\gone\amplifier.dat"}
    assert _resolve(session / "kilosort4", params) == inner


def test_without_dat_path_picks_the_largest_raw_file(session):
    """Aux/digital-in companions are much smaller than the recording."""
    assert _resolve(session / "kilosort4", {}) == session / "amplifier.dat"


def test_returns_none_when_no_raw_file_exists(tmp_path):
    ks = tmp_path / "kilosort4"
    ks.mkdir()
    assert _resolve(ks, {"dat_path": r"F:\gone\amplifier.dat"}) is None
