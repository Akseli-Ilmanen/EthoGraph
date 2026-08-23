"""TrialTree.save(): incremental writes for trials touched via update_trial.

A full rewrite costs O(total tree size) no matter how small the edit. These
tests pin the fast path — write only the dirty trial's group in place — and
confirm every case that must fall back to a full rewrite still does (added/
removed trial, a variable's shape changing) so no edit is ever lost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

import ethograph as eto
from ethograph.io.trialtree import TrialTree


def _trial_ds(trial: int, n: int = 20, value: float = 0.0) -> xr.Dataset:
    return xr.Dataset(
        {"speed": (("time", "individual"), np.full((n, 1), value, dtype="float64"))},
        coords={"time": np.arange(n) / 10.0, "individual": ["A"]},
        attrs={"trial": trial},
    )


def _make_tree(path: Path, n_trials: int = 3) -> TrialTree:
    eto.from_datasets([_trial_ds(t) for t in range(n_trials)]).save(str(path))
    return TrialTree.open(str(path))


def test_noop_save_does_no_io(tmp_path: Path, monkeypatch):
    path = tmp_path / "session.nc"
    tree = _make_tree(path)

    def _fail(*a, **k):
        raise AssertionError("full rewrite should not run when nothing changed")

    monkeypatch.setattr(TrialTree, "to_netcdf", _fail)
    tree.save()  # no edits since open() — must be a pure no-op
    tree.close()


def test_attr_edit_uses_the_incremental_path(tmp_path: Path, monkeypatch):
    path = tmp_path / "session.nc"
    tree = _make_tree(path)

    def _fail(*a, **k):
        raise AssertionError("a single attr edit must not trigger a full-tree rewrite")

    monkeypatch.setattr(TrialTree, "to_netcdf", _fail)
    tree.set_trial_attr(1, "note", "hello")
    tree.save()
    tree.close()

    reopened = TrialTree.open(str(path))
    assert reopened.trial(1).attrs["note"] == "hello"
    assert "note" not in reopened.trial(0).attrs
    reopened.close()


def test_value_edit_same_shape_persists_and_leaves_other_trials_untouched(tmp_path: Path):
    path = tmp_path / "session.nc"
    tree = _make_tree(path)

    tree.update_trial(1, lambda ds: ds.assign(speed=ds["speed"] * 0 + 999.0))
    tree.save()
    tree.close()

    reopened = TrialTree.open(str(path))
    np.testing.assert_array_equal(reopened.trial(1)["speed"].values, 999.0)
    np.testing.assert_array_equal(reopened.trial(0)["speed"].values, 0.0)
    reopened.close()


def test_new_variable_on_one_trial_persists(tmp_path: Path):
    path = tmp_path / "session.nc"
    tree = _make_tree(path)

    tree.update_trial(1, lambda ds: ds.assign(extra=(("time",), np.arange(len(ds["time"])).astype("float64"))))
    tree.save()
    tree.close()

    reopened = TrialTree.open(str(path))
    assert "extra" in reopened.trial(1).data_vars
    assert "extra" not in reopened.trial(0).data_vars
    reopened.close()


def test_shape_change_falls_back_to_full_rewrite(tmp_path: Path):
    path = tmp_path / "session.nc"
    tree = _make_tree(path)

    tree.update_trial(1, lambda ds: ds.isel(time=slice(0, 5)))
    tree.save()
    tree.close()

    reopened = TrialTree.open(str(path))
    assert reopened.trial(1)["speed"].shape == (5, 1)
    assert reopened.trial(0)["speed"].shape == (20, 1)
    reopened.close()


def test_added_trial_falls_back_to_full_rewrite(tmp_path: Path):
    path = tmp_path / "session.nc"
    tree = _make_tree(path)

    tree["3"] = xr.DataTree(_trial_ds(3))
    tree.save()
    tree.close()

    reopened = TrialTree.open(str(path))
    assert sorted(reopened.trials) == [0, 1, 2, 3]
    reopened.close()


def test_tree_stays_usable_after_incremental_save(tmp_path: Path):
    path = tmp_path / "session.nc"
    tree = _make_tree(path)

    tree.set_trial_attr(1, "note", "hello")
    tree.save()

    # the in-memory tree itself (not a reopened copy) must still work
    assert tree.trial(1).attrs["note"] == "hello"
    np.testing.assert_array_equal(tree.trial(0)["speed"].values, 0.0)
    assert sorted(tree.trials) == [0, 1, 2]
    tree.close()


def test_save_as_new_path_always_writes_fully_even_with_no_dirty_trials(tmp_path: Path):
    path = tmp_path / "session.nc"
    other = tmp_path / "copy.nc"
    tree = _make_tree(path)

    tree.save(str(other))  # no edits, but a different target: must still materialise the file
    tree.close()

    assert other.exists()
    copied = TrialTree.open(str(other))
    assert sorted(copied.trials) == [0, 1, 2]
    copied.close()


def test_second_incremental_save_after_reload_still_works(tmp_path: Path):
    """A save that goes through the close+reopen dance must leave dirty-tracking consistent."""
    path = tmp_path / "session.nc"
    tree = _make_tree(path)

    tree.set_trial_attr(0, "a", "1")
    tree.save()
    tree.set_trial_attr(2, "b", "2")
    tree.save()
    tree.close()

    reopened = TrialTree.open(str(path))
    assert reopened.trial(0).attrs["a"] == "1"
    assert reopened.trial(2).attrs["b"] == "2"
    reopened.close()
