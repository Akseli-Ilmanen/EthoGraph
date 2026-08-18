"""Tests for the display-basis authority: SourceCollection clock conversions
and app_state.display_basis / to_display / from_display."""

import pandas as pd
import pytest

from ethograph.io.time_model import SourceCollection


@pytest.fixture
def sc():
    """Three trials with gaps: 10-14, 20-26, 40-45 (ids 1, 2, 4)."""
    collection = SourceCollection()
    collection.set_trials(ids=[1, 2, 4], starts=[10.0, 20.0, 40.0], stops=[14.0, 26.0, 45.0])
    return collection


# ---------------------------------------------------------------------------
# SourceCollection conversions
# ---------------------------------------------------------------------------


def test_to_session(sc):
    assert sc.to_session(1, 0.0) == pytest.approx(10.0)
    assert sc.to_session(2, 3.0) == pytest.approx(23.0)
    assert sc.to_session(4, 0.5) == pytest.approx(40.5)


def test_to_session_string_ids_tolerated(sc):
    assert sc.to_session("2", 1.0) == pytest.approx(21.0)


def test_to_session_unknown_trial_passthrough(sc):
    assert sc.to_session(99, 3.0) == pytest.approx(3.0)


def test_to_trial_inside(sc):
    assert sc.to_trial(23.0) == (2, pytest.approx(3.0))
    assert sc.to_trial(10.0) == (1, pytest.approx(0.0))


def test_to_trial_gap_strict_returns_none(sc):
    assert sc.to_trial(17.0, strict=True) is None
    assert sc.to_trial(30.0, strict=True) is None


def test_to_trial_gap_snaps_when_not_strict(sc):
    # 18.0 is closer to trial 2's start (20) than trial 1's end (14)
    trial_id, t_rel = sc.to_trial(18.0)
    assert trial_id == 2
    assert t_rel == pytest.approx(0.0)
    # 15.0 is closer to trial 1's end
    trial_id, t_rel = sc.to_trial(15.0)
    assert trial_id == 1
    assert t_rel == pytest.approx(4.0)


def test_to_trial_no_trials_returns_none():
    assert SourceCollection().to_trial(5.0) is None


def test_roundtrip(sc):
    for tid, t_rel in [(1, 0.0), (2, 5.5), (4, 4.9)]:
        t_abs = sc.to_session(tid, t_rel)
        assert sc.to_trial(t_abs, strict=True) == (tid, pytest.approx(t_rel))


# ---------------------------------------------------------------------------
# app_state display-basis authority
# ---------------------------------------------------------------------------


def test_display_basis_follows_slider_scope(app_state):
    app_state.slider_scope = "trial"
    app_state.navigate_mode = "trial"
    assert app_state.display_basis == "trial"
    app_state.slider_scope = "session"
    assert app_state.display_basis == "session"


def test_display_basis_label_navigation_is_trial_basis(app_state):
    """Label/sequence windows are built from trial-relative onsets, so they
    force trial basis even under session scope."""
    app_state.slider_scope = "session"
    app_state.navigate_mode = "label"
    assert app_state.display_basis == "trial"
    app_state.navigate_mode = "sequence"
    assert app_state.display_basis == "trial"


def test_to_from_display_trial_basis(app_state, sc):
    app_state.slider_scope = "trial"
    app_state.navigate_mode = "trial"
    app_state.source_collection = sc
    app_state.trials_sel = 2
    assert app_state.to_display(2, 3.0) == pytest.approx(3.0)
    assert app_state.from_display(3.0) == (2, pytest.approx(3.0))


def test_to_from_display_session_basis(app_state, sc):
    app_state.slider_scope = "session"
    app_state.navigate_mode = "trial"
    app_state.source_collection = sc
    app_state.trials_sel = 1
    assert app_state.to_display(2, 3.0) == pytest.approx(23.0)
    # from_display finds the trial under the click, not the current trial
    assert app_state.from_display(41.0) == (4, pytest.approx(1.0))
    assert app_state.from_display(17.0, strict=True) is None


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _labels_df(onsets_by_trial: dict) -> pd.DataFrame:
    rows = []
    for trial, onsets in onsets_by_trial.items():
        for o in onsets:
            rows.append({"trial": trial, "individual": "crow1", "labels": 1, "onset_s": o, "offset_s": o + 0.5})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# get_display_intervals
# ---------------------------------------------------------------------------


def test_display_intervals_trial_basis_is_current_trial(app_state, sc):
    app_state.slider_scope = "trial"
    app_state.navigate_mode = "trial"
    app_state.source_collection = sc
    app_state._all_labels_df = _labels_df({1: [0.5], 2: [1.0]})
    app_state.trials_sel = 2
    app_state.label_intervals = app_state.get_trial_intervals(2)

    view = app_state.get_display_intervals()
    assert len(view) == 1
    assert view["onset_s"].iloc[0] == pytest.approx(1.0)


def test_display_intervals_session_basis_shows_all_trials_shifted(app_state, sc):
    app_state.slider_scope = "session"
    app_state.navigate_mode = "trial"
    app_state.source_collection = sc
    app_state._all_labels_df = _labels_df({1: [0.5], 2: [1.0], 4: [2.0]})
    app_state.trials_sel = 1

    view = app_state.get_display_intervals()
    assert len(view) == 3
    by_trial = {row["trial"]: row["onset_s"] for _, row in view.iterrows()}
    assert by_trial[1] == pytest.approx(10.5)
    assert by_trial[2] == pytest.approx(21.0)
    assert by_trial[4] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# XarrayLoader display offset (session basis: trial-local data on an absolute axis)
# ---------------------------------------------------------------------------


def test_xarray_loader_display_offset_renders_at_session_position():
    import numpy as np
    import xarray as xr

    from ethograph.io.catalog import XarrayLoader

    t = np.linspace(0.0, 4.0, 41)  # trial-local coord
    ds = xr.Dataset({"speed": ("time", np.arange(41.0))}, coords={"time": t})
    loader = XarrayLoader(ds)

    native = loader.select("speed", {}, t0=0.0, t1=4.0)
    assert native.time[0] == pytest.approx(0.0)

    # Session basis: the axis is absolute, this trial starts at 20 s there.
    loader.set_display_offset_provider(lambda: -20.0)
    shifted = loader.select("speed", {}, t0=20.0, t1=24.0)
    assert shifted is not None
    assert shifted.time[0] == pytest.approx(20.0)
    assert shifted.time[-1] == pytest.approx(24.0)
    np.testing.assert_array_equal(np.asarray(shifted.data), np.asarray(native.data))

    # A window outside the trial's span selects nothing (other trials absent).
    outside = loader.select("speed", {}, t0=50.0, t1=60.0)
    assert outside is None or len(outside.time) == 0


# ---------------------------------------------------------------------------
# Trial window duration without an alignment file (sc bookmark fallback)
# ---------------------------------------------------------------------------


def test_trial_bounds_from_source_collection_bookmarks(sc):
    """No alignment: the trial window comes from the trial bookmarks, not the
    union range (which made every 'trial' window span the whole session)."""
    from ethograph.io.nwb_alignment import EmpytAlignment
    from ethograph.io.time_model import compute_trial_video_bounds

    tvb = compute_trial_video_bounds(EmpytAlignment(), 2, None, source_collection=sc)
    assert tvb.trial_range is not None
    assert tvb.trial_range.start_s == pytest.approx(0.0)
    assert tvb.trial_range.duration == pytest.approx(6.0)  # trial 2: 20-26 s
