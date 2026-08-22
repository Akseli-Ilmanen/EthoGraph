"""``labeling_method``: who vouches for a label, and the curation transitions.

Pure pandas — the rules in ``labels/curation.py`` and the column's life in
the interval/TSV stores, with no Qt involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethograph.labels.curation import (
    CURATED_COLUMN,
    build_review_queue,
    curate_label,
    curate_rows,
    curate_trial,
    curated_column,
    curated_column_differs,
    label_key,
    mark_manual,
    method_counts,
    queue_index_of,
    set_method,
    targets_from_seeds,
    trial_curation_status,
)
from ethograph.labels.intervals import (
    HUMAN_CONFIDENCE,
    INTERVAL_COLUMNS,
    LABELING_AUTOMATED,
    LABELING_CURATED,
    LABELING_MANUAL,
    add_interval,
    add_point,
    empty_intervals,
    ensure_labeling_method,
)
from ethograph.labels.tsv_store import (
    TSV_COLUMNS,
    get_trial_from_tsv,
    load_labels_tsv,
    save_labels_tsv,
    set_trial_in_tsv,
)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": [1, 1, 2, 2],
            "labels": [1, 2, 1, 3],
            "onset_s": [0.5, 1.0, 0.2, 2.0],
            "offset_s": [np.nan, 1.5, 0.6, np.nan],
            "individual": ["a", "a", "a", "b"],
            "individual_rec": ["", "", "", ""],
            "event_type": ["point", "state", "state", "point"],
            "confidence": [0.4, 1.0, 1.0, 0.7],
            "labeling_method": [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_CURATED, LABELING_AUTOMATED],
        }
    )


def _methods(df: pd.DataFrame) -> list[str]:
    return df.sort_values(["trial", "onset_s"])["labeling_method"].tolist()


# ---------------------------------------------------------------------------
# The column
# ---------------------------------------------------------------------------


class TestEnsureLabelingMethod:
    def test_missing_column_is_read_off_the_confidence(self):
        df = _labels().drop(columns=["labeling_method"])
        ensure_labeling_method(df)
        # 0.4 and 0.7 are a model's scores; 1.0 is a human's.
        assert _methods(df) == [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_MANUAL, LABELING_AUTOMATED]

    def test_blank_or_unknown_values_are_filled_and_valid_ones_kept(self):
        df = _labels()
        df.loc[0, "labeling_method"] = None
        df.loc[3, "labeling_method"] = "verified?"
        ensure_labeling_method(df)
        assert _methods(df) == [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_CURATED, LABELING_AUTOMATED]

    def test_it_is_an_interval_column(self):
        assert "labeling_method" in INTERVAL_COLUMNS
        assert "labeling_method" in empty_intervals().columns


class TestStore:
    def test_hand_placed_labels_are_manual(self):
        df = add_point(empty_intervals(), 1.0, 1, "a")
        df = add_interval(df, 2.0, 3.0, 2, "a")
        assert df["labeling_method"].tolist() == [LABELING_MANUAL, LABELING_MANUAL]

    def test_a_model_passes_automated(self):
        df = add_point(empty_intervals(), 1.0, 1, "a", confidence=0.3, labeling_method=LABELING_AUTOMATED)
        assert df.loc[0, "labeling_method"] == LABELING_AUTOMATED

    def test_a_trimmed_remnant_keeps_its_method(self):
        df = add_interval(empty_intervals(), 0.0, 2.0, 1, "a", labeling_method=LABELING_AUTOMATED)
        df = add_interval(df, 1.0, 3.0, 2, "a")
        by_label = dict(zip(df["labels"], df["labeling_method"]))
        assert by_label == {1: LABELING_AUTOMATED, 2: LABELING_MANUAL}

    def test_tsv_round_trip_keeps_the_method(self, tmp_path):
        path = tmp_path / "x_labels.tsv"
        save_labels_tsv(path, _labels())
        back = load_labels_tsv(path)
        assert _methods(back) == _methods(_labels())
        assert "labeling_method" in TSV_COLUMNS
        assert "human_verified" not in TSV_COLUMNS

    def test_a_legacy_file_is_read_conservatively(self, tmp_path):
        path = tmp_path / "old_labels.tsv"
        _labels().drop(columns=["labeling_method"]).assign(human_verified=1).to_csv(path, sep="\t", index=False)
        back = load_labels_tsv(path)
        # human_verified=1 says nothing per label; the scores do.
        assert _methods(back) == [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_MANUAL, LABELING_AUTOMATED]
        assert "human_verified" in back.columns

    def test_rewriting_a_trial_carries_a_legacy_column_along(self):
        df = _labels().assign(human_verified=[1, 1, 0, 0])
        rows = get_trial_from_tsv(df, 1)
        rows = add_point(rows, 3.0, 1, "a")
        out = set_trial_in_tsv(df, 1, rows)
        assert out.loc[out["trial"] == 1, "human_verified"].tolist() == [1, 1, 1]
        assert out.loc[out["trial"] == 1, "labeling_method"].tolist()[-1] == LABELING_MANUAL


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


class TestTransitions:
    def test_curate_trial_turns_automated_into_curated_and_nothing_else(self):
        out, n = curate_trial(_labels(), 1)
        assert n == 1
        assert _methods(out) == [LABELING_CURATED, LABELING_MANUAL, LABELING_CURATED, LABELING_AUTOMATED]

    def test_curate_trial_respects_the_scope(self):
        out, n = curate_trial(_labels(), 2, label_ids={1})
        assert n == 0  # label 1 in trial 2 is already curated; label 3 is out of scope
        out, n = curate_trial(_labels(), 2, label_ids={3})
        assert n == 1
        assert out.loc[out["labels"] == 3, "labeling_method"].item() == LABELING_CURATED

    def test_curate_label_touches_one_row(self):
        inst = {"trial": 2, "labels": 3, "onset_s": 2.0, "individual": "b", "individual_rec": ""}
        out, n = curate_label(_labels(), inst)
        assert n == 1
        assert _methods(out) == [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_CURATED, LABELING_CURATED]

    def test_manual_is_never_rewritten_as_curated(self):
        df = _labels()
        out, n = curate_rows(df, pd.Series(True, index=df.index))
        assert n == 2
        assert LABELING_MANUAL in _methods(out)

    def test_an_edit_makes_a_label_manual(self):
        inst = {"trial": 2, "labels": 1, "onset_s": 0.2, "individual": "a", "individual_rec": ""}
        out, n = mark_manual(_labels(), inst)
        assert n == 1
        assert out.loc[(out["trial"] == 2) & (out["labels"] == 1), "labeling_method"].item() == LABELING_MANUAL

    def test_unknown_method_is_a_bug(self):
        df = _labels()
        with pytest.raises(ValueError):
            set_method(df, pd.Series(True, index=df.index), "verified")

    def test_empty_tables_pass_through(self):
        assert curate_trial(None, 1) == (None, 0)
        empty = empty_intervals()
        out, n = curate_trial(empty, 1)
        assert n == 0 and out is empty

    def test_label_key_identifies_a_row(self):
        row = _labels().iloc[3]
        assert label_key(row) == ("2", 3, 2.0, "b", "")


# ---------------------------------------------------------------------------
# Per-trial verdicts
# ---------------------------------------------------------------------------


class TestStatus:
    def test_a_trial_is_curated_when_nothing_is_automated(self):
        assert trial_curation_status(_labels(), [1, 2, 3]) == {"1": False, "2": False, "3": True}
        out, _ = curate_trial(_labels(), 2)
        assert trial_curation_status(out, [1, 2]) == {"1": False, "2": True}

    def test_new_automated_labels_reopen_a_curated_trial(self):
        out, _ = curate_trial(_labels(), 2)
        rows = add_point(get_trial_from_tsv(out, 2), 5.0, 3, "b", confidence=0.5, labeling_method=LABELING_AUTOMATED)
        out = set_trial_in_tsv(out, 2, rows)
        assert trial_curation_status(out, [2]) == {"2": False}

    def test_method_counts(self):
        assert method_counts(_labels(), 2) == {LABELING_MANUAL: 0, LABELING_AUTOMATED: 1, LABELING_CURATED: 1}
        assert method_counts(None, 2) == {LABELING_MANUAL: 0, LABELING_AUTOMATED: 0, LABELING_CURATED: 0}

    def test_curated_column_is_written_per_trial(self):
        mdf = pd.DataFrame({"trial": [1, 2, 3], "genotype": ["wt", "ko", "wt"]})
        status = trial_curation_status(_labels(), [1, 2, 3])
        out = curated_column(mdf, status)
        assert out[CURATED_COLUMN].tolist() == [0, 0, 1]
        assert CURATED_COLUMN not in mdf.columns  # a copy
        assert curated_column(None, status) is None

    def test_differs_only_when_a_verdict_changed(self):
        mdf = pd.DataFrame({"trial": [1, 2], CURATED_COLUMN: [0, 0]})
        assert curated_column_differs(mdf, {"1": False, "2": False}) is False
        assert curated_column_differs(mdf, {"1": False, "2": True}) is True
        assert curated_column_differs(mdf.drop(columns=[CURATED_COLUMN]), {"1": False}) is True
        assert curated_column_differs(None, {"1": False}) is False


# ---------------------------------------------------------------------------
# Frame-by-frame review queue
# ---------------------------------------------------------------------------


class TestQueue:
    def test_queue_visits_each_trial_once_in_time_order(self):
        queue = build_review_queue(_labels(), None)
        got = [(str(t.inst["trial"]), t.inst["labels"], t.field) for t in queue]
        assert got == [
            ("1", 1, "point"),
            ("1", 2, "start"),
            ("1", 2, "end"),
            ("2", 1, "start"),
            ("2", 1, "end"),
            ("2", 3, "point"),
        ]
        start, end = queue[1], queue[2]
        assert start.inst is end.inst  # one label, two boundaries

    def test_scope_individual_and_trials_narrow_the_queue(self):
        assert [t.inst["labels"] for t in build_review_queue(_labels(), {3})] == [3]
        assert [t.inst["trial"] for t in build_review_queue(_labels(), None, individual="b")] == [2]
        assert {str(t.inst["trial"]) for t in build_review_queue(_labels(), None, allowed_trials={"1"})} == {"1"}

    def test_targets_from_seeds_share_the_instance(self):
        seeds = [
            {"trial": "0", "labels": 8, "onset_s": 2.0, "offset_s": 3.0, "field": "end", "event_type": "state"},
            {"trial": "0", "labels": 8, "onset_s": 2.0, "offset_s": 3.0, "field": "start", "event_type": "state"},
            {"trial": "0", "labels": 4, "onset_s": 1.0, "offset_s": None, "field": "point", "event_type": "point"},
        ]
        targets = targets_from_seeds(seeds)
        assert [(t.inst["labels"], t.field) for t in targets] == [(4, "point"), (8, "start"), (8, "end")]
        assert targets[1].inst is targets[2].inst
        assert np.isnan(targets[0].inst["offset_s"])

    def test_queue_index_of(self):
        queue = build_review_queue(_labels(), None)
        inst = {"trial": 2, "labels": 1, "onset_s": 0.2, "individual": "a", "individual_rec": ""}
        assert queue_index_of(queue, inst, "end") == 4
        assert queue_index_of(queue, inst) == 3
        assert queue_index_of(queue, {**inst, "labels": 9}) is None


def test_human_confidence_is_the_manual_default():
    df = add_point(empty_intervals(), 1.0, 1, "a")
    assert df.loc[0, "confidence"] == HUMAN_CONFIDENCE
