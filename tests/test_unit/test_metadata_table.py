"""Tests for the metadata table system."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ethograph.io.metadata_table import (
    condition_columns,
    empty_metadata_df,
    load_metadata_df,
    load_metadata_tsv,
    metadata_from_intervalset,
    metadata_from_nwb_trials,
    metadata_tsv_path,
    save_metadata_tsv,
)


def test_condition_columns():
    df = pd.DataFrame(
        {
            "trial": [1, 2],
            "genotype": ["WT", "KO"],
            "start_time": [0.0, 1.0],
            "video_cam-1": ["a.mp4", "b.mp4"],
        }
    )
    cols = condition_columns(df)
    assert cols == ["genotype"]
    assert "trial" not in cols


def test_save_load_roundtrip():
    df = pd.DataFrame(
        {
            "trial": [1, 2, 3],
            "genotype": ["WT", "KO", "WT"],
            "dose_mg": [0.0, 5.0, 10.0],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_metadata.tsv"
        save_metadata_tsv(path, df)
        loaded = load_metadata_tsv(path)
        assert list(loaded.columns) == ["trial", "genotype", "dose_mg"]
        assert len(loaded) == 3
        assert loaded.loc[1, "genotype"] == "KO"


def test_metadata_tsv_path():
    p = metadata_tsv_path("/data/experiment.nc")
    assert p.name == "experiment_metadata.tsv"


def test_empty_metadata_df():
    mdf = empty_metadata_df([1])
    assert "trial" in mdf.columns


def test_load_metadata_df_empty_fallback():
    mdf, source = load_metadata_df(trial_ids=[1, 2])
    assert source is None
    assert list(mdf["trial"]) == [1, 2]


def test_metadata_from_nwb_trials_filters_infrastructure():
    trials_df = pd.DataFrame(
        {
            "trial": [1, 2],
            "start_time": [0.0, 1.0],
            "stop_time": [0.5, 1.5],
            "genotype": ["WT", "KO"],
            "video_cam-1": ["a.mp4", "b.mp4"],
        }
    )
    mdf = metadata_from_nwb_trials(trials_df)
    assert list(mdf.columns) == ["trial", "genotype"]
    assert list(mdf["genotype"]) == ["WT", "KO"]


def test_metadata_from_intervalset():
    nap = pytest.importorskip("pynapple")

    intervals = nap.IntervalSet(
        start=[0.0, 1.0],
        end=[0.5, 1.5],
        metadata=pd.DataFrame(
            {
                "trial": [1, 2],
                "condition": ["A", "B"],
                "start_time": [0.0, 1.0],
            }
        ),
    )
    mdf = metadata_from_intervalset(intervals)
    assert list(mdf.columns) == ["trial", "condition"]
    assert list(mdf["condition"]) == ["A", "B"]


def test_load_metadata_df_uses_nwb_source_when_trials_present(monkeypatch, tmp_path):
    nwb_path = tmp_path / "session.nwb"
    nwb_path.write_text("placeholder", encoding="utf-8")

    class _FakeAlignment:
        def __init__(self):
            self.trials_df = pd.DataFrame(
                {
                    "trial": [1, 2],
                    "genotype": ["WT", "KO"],
                    "start_time": [0.0, 1.0],
                    "stop_time": [0.5, 1.5],
                }
            )

    monkeypatch.setattr("ethograph.io.metadata_table.make_nwb_alignment", lambda _: _FakeAlignment())

    mdf, source = load_metadata_df(source_path=nwb_path)

    assert source == str(nwb_path)
    assert list(mdf.columns) == ["trial", "genotype"]
    assert list(mdf["trial"]) == [1, 2]


# ---------------------------------------------------------------------------
# merge_trial_metadata / apply_metadata_choices
# ---------------------------------------------------------------------------


def _alignment_trials_df():
    return pd.DataFrame(
        {
            "start_time": [20.9, 33.6, 46.5],
            "stop_time": [25.5, 38.4, 50.7],
            "trial": [1, 2, 4],
            "video_cam-1": ["a.mp4", "b.mp4", "c.mp4"],
        }
    )


def test_merge_unions_new_columns():
    from ethograph.io.metadata_table import merge_trial_metadata

    user = pd.DataFrame({"trial": [1, 2, 4], "genotype": ["WT", "KO", "WT"]})
    merged, conflicts = merge_trial_metadata(_alignment_trials_df(), user)
    assert conflicts == []
    assert list(merged["genotype"]) == ["WT", "KO", "WT"]
    assert list(merged["trial"]) == [1, 2, 4]
    assert "start_time" in merged.columns


def test_merge_partial_metadata_fills_nan_without_conflict():
    from ethograph.io.metadata_table import merge_trial_metadata

    user = pd.DataFrame({"trial": [2], "genotype": ["KO"]})
    merged, conflicts = merge_trial_metadata(_alignment_trials_df(), user)
    assert conflicts == []
    assert merged.loc[merged["trial"] == 2, "genotype"].iloc[0] == "KO"
    assert merged.loc[merged["trial"] == 1, "genotype"].isna().all()


def test_merge_pynapple_naming_matches_nwb_timing():
    """User 'start'/'end' columns are the same quantity as start_time/stop_time."""
    from ethograph.io.metadata_table import merge_trial_metadata

    user = pd.DataFrame(
        {
            "trial": [1, 2, 4],
            "start": [20.9, 33.6, 46.5],
            "end": [25.5, 38.4, 50.7],
        }
    )
    merged, conflicts = merge_trial_metadata(_alignment_trials_df(), user)
    assert conflicts == []
    assert "start" not in merged.columns
    assert "end" not in merged.columns


def test_merge_unknown_trial_ids_raise():
    from ethograph.io.metadata_table import merge_trial_metadata

    user = pd.DataFrame({"trial": [1, 3], "genotype": ["WT", "KO"]})
    with pytest.raises(ValueError, match="not present in the alignment"):
        merge_trial_metadata(_alignment_trials_df(), user)


def test_merge_duplicate_trial_ids_raise():
    from ethograph.io.metadata_table import merge_trial_metadata

    user = pd.DataFrame({"trial": [1, 1], "genotype": ["WT", "KO"]})
    with pytest.raises(ValueError, match="duplicate"):
        merge_trial_metadata(_alignment_trials_df(), user)


def test_merge_conflict_defaults_to_alignment_and_choice_applies():
    from ethograph.io.metadata_table import apply_metadata_choices, merge_trial_metadata

    user = pd.DataFrame(
        {
            "trial": [1, 2, 4],
            "start_time": [21.0, 33.6, 46.5],  # trial 1 differs
        }
    )
    merged, conflicts = merge_trial_metadata(_alignment_trials_df(), user)
    assert len(conflicts) == 1
    assert conflicts[0].column == "start_time"
    assert conflicts[0].n_differing == 1
    # Default: alignment side kept.
    assert merged.loc[merged["trial"] == 1, "start_time"].iloc[0] == pytest.approx(20.9)

    chosen = apply_metadata_choices(merged, conflicts, ["start_time"])
    assert chosen.loc[chosen["trial"] == 1, "start_time"].iloc[0] == pytest.approx(21.0)
    # Non-conflicting rows untouched.
    assert chosen.loc[chosen["trial"] == 2, "start_time"].iloc[0] == pytest.approx(33.6)


# ---------------------------------------------------------------------------
# Trials from alignment NWB (pynapple load path helpers)
# ---------------------------------------------------------------------------


def test_trial_ids_from_alignment_trials_ep():
    """IntervalSet built from a trials table carries the real trial ids."""
    pytest.importorskip("pynapple")
    from ethograph.io.data_loader import _trial_ids_from_ep
    from ethograph.io.nwb_alignment import _build_trials_ep

    ep = _build_trials_ep(_alignment_trials_df())
    assert ep is not None
    assert _trial_ids_from_ep(ep) == [1, 2, 4]


def test_merged_trials_df_prefers_alignment_table(tmp_path):
    from ethograph.io.data_loader import _merged_trials_df
    from ethograph.io.nwb_alignment import TableAlignment

    alignment = TableAlignment(_alignment_trials_df())
    user_path = tmp_path / "meta.tsv"
    pd.DataFrame({"trial": [1, 2, 4], "genotype": ["WT", "KO", "WT"]}).to_csv(user_path, sep="\t", index=False)

    merged = _merged_trials_df(alignment, str(user_path), None)
    assert merged is not None
    assert list(merged["genotype"]) == ["WT", "KO", "WT"]


def test_merged_trials_df_conflict_resolver_wins(tmp_path):
    from ethograph.io.data_loader import _merged_trials_df
    from ethograph.io.nwb_alignment import TableAlignment

    alignment = TableAlignment(_alignment_trials_df())
    user_path = tmp_path / "meta.tsv"
    pd.DataFrame({"trial": [1, 2, 4], "stop_time": [26.0, 38.4, 50.7]}).to_csv(user_path, sep="\t", index=False)

    resolver_calls = []

    def resolver(conflicts):
        resolver_calls.append([c.column for c in conflicts])
        return ["stop_time"]

    merged = _merged_trials_df(alignment, str(user_path), resolver)
    assert resolver_calls == [["stop_time"]]
    assert merged.loc[merged["trial"] == 1, "stop_time"].iloc[0] == pytest.approx(26.0)


def test_merged_trials_df_no_alignment_returns_none():
    from ethograph.io.data_loader import _merged_trials_df
    from ethograph.io.nwb_alignment import EmpytAlignment

    assert _merged_trials_df(EmpytAlignment(), None, None) is None


def test_intervalset_metadata_matched_by_time_containment():
    """A trials .npz has no trial-id column and may hold fewer trials than the
    alignment table — its metadata rows attach to the alignment trial whose
    window contains each interval."""
    pytest.importorskip("pynapple")
    import pynapple as nap

    from ethograph.io.data_loader import _intervalset_metadata_by_time, _merged_trials_df
    from ethograph.io.nwb_alignment import TableAlignment

    align_df = pd.DataFrame(
        {
            "start_time": [20.9, 33.6, 46.5, 58.8],
            "stop_time": [25.5, 38.4, 50.7, 62.8],
            "trial": [1, 2, 3, 5],
        }
    )
    # Three intervals inside trials 1, 3 and 5 (none for trial 2), slightly
    # tighter than the alignment windows, plus one outside every trial.
    ep = nap.IntervalSet(
        start=[21.5, 47.0, 59.3, 100.0],
        end=[24.7, 50.4, 62.3, 101.0],
        metadata={"condition": ["right", "left", "right", "orphan"]},
    )

    df = _intervalset_metadata_by_time(ep, align_df)
    assert list(df["trial"]) == [1, 3, 5]
    assert list(df["condition"]) == ["right", "left", "right"]

    merged = _merged_trials_df(TableAlignment(align_df), None, None)
    assert "condition" not in merged.columns  # no metadata source -> alignment only
