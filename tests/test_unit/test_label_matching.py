"""Trial-level label matching — what "Find label inconsistencies" filters on."""

import pandas as pd
import pytest

from ethograph.utils.sequences import (
    LABEL_MATCH_MODES,
    parse_label_pattern,
    trial_matches_labels,
    trials_matching_labels,
)


def _labels(per_trial: dict[str, list[int]]) -> pd.DataFrame:
    """One row per label, onsets in the order the ids are listed."""
    rows = []
    for trial, ids in per_trial.items():
        for i, label in enumerate(ids):
            rows.append({"trial": trial, "labels": label, "onset_s": float(i), "individual": "a"})
    return pd.DataFrame(rows, columns=["trial", "labels", "onset_s", "individual"])


class TestPattern:
    def test_reads_the_sequence_spelling(self):
        assert parse_label_pattern("1-2-6-8") == [1, 2, 6, 8]
        assert parse_label_pattern(" 1 , 2 ") == [1, 2]

    def test_repeats_are_kept(self):
        """ "6-6" asks about two occurrences, not one."""
        assert parse_label_pattern("6-6") == [6, 6]

    def test_nonsense_is_no_pattern(self):
        assert parse_label_pattern("1-x") == []
        assert parse_label_pattern("") == []


class TestModes:
    TRIAL = [1, 2, 6, 6, 8]

    def test_present_ignores_order(self):
        assert trial_matches_labels(self.TRIAL, [8, 1], "present")
        assert not trial_matches_labels(self.TRIAL, [1, 9], "present")

    def test_partial_is_the_uncoupled_case(self):
        """Some but not all — one label without its partner."""
        assert trial_matches_labels([1], [1, 2], "partial")
        assert not trial_matches_labels([1, 2], [1, 2], "partial")  # both there
        assert not trial_matches_labels([9], [1, 2], "partial")  # neither there

    def test_repeated_flags_a_doubled_label(self):
        """A class that should happen once per trial happens twice."""
        assert trial_matches_labels(self.TRIAL, [6], "repeated")
        assert trial_matches_labels(self.TRIAL, [1, 6], "repeated")  # any of them
        assert not trial_matches_labels(self.TRIAL, [1, 2, 8], "repeated")
        assert not trial_matches_labels(self.TRIAL, [9], "repeated")

    def test_order_allows_labels_in_between(self):
        assert trial_matches_labels(self.TRIAL, [1, 2, 6, 8], "order")
        assert trial_matches_labels(self.TRIAL, [1, 8], "order")
        assert not trial_matches_labels(self.TRIAL, [8, 1], "order")

    def test_order_strict_wants_them_consecutive(self):
        """The checkbox that separates 1-2-6-8 from 1-2-6-6-8."""
        assert not trial_matches_labels(self.TRIAL, [1, 2, 6, 8], "order_strict")
        assert trial_matches_labels(self.TRIAL, [1, 2, 6, 6, 8], "order_strict")
        assert trial_matches_labels(self.TRIAL, [6, 6, 8], "order_strict")

    def test_an_empty_pattern_matches_nothing(self):
        assert not trial_matches_labels(self.TRIAL, [], "present")

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="Unknown label match mode"):
            trial_matches_labels(self.TRIAL, [1], "sideways")

    def test_every_mode_is_offered_and_implemented(self):
        for mode in LABEL_MATCH_MODES:
            trial_matches_labels(self.TRIAL, [1, 2], mode)


class TestOverTrials:
    DF = _labels({"1": [1, 2], "2": [1], "3": [2, 1], "4": []})

    def test_finds_the_trials_that_match(self):
        assert trials_matching_labels(self.DF, [1, 2], mode="present") == {"1", "3"}
        assert trials_matching_labels(self.DF, [1, 2], mode="order") == {"1"}

    def test_uncoupled_finds_the_lonely_label(self):
        assert trials_matching_labels(self.DF, [1, 2], mode="partial") == {"2"}

    def test_repeated_finds_the_doubled_label(self):
        df = _labels({"1": [1, 2], "2": [1, 2, 1], "3": [2, 2]})
        assert trials_matching_labels(df, [1], mode="repeated") == {"2"}
        assert trials_matching_labels(df, [1, 2], mode="repeated") == {"2", "3"}

    def test_invert_needs_the_whole_population(self):
        """A trial with no labels at all only exists if it is named.

        "Find the trials missing this" has to include the trials that carry
        nothing, and those are invisible in the labels table — so the trial
        list has to be passed in.
        """
        trials = ["1", "2", "3", "4"]
        assert trials_matching_labels(self.DF, [1, 2], mode="present", invert=True, trials=trials) == {"2", "4"}
        # Without it, trial 4 is not even judged.
        assert trials_matching_labels(self.DF, [1, 2], mode="present", invert=True) == {"2"}

    def test_individual_scopes_the_order(self):
        """Two animals interleave, and an order across both means nothing."""
        df = pd.DataFrame(
            {
                "trial": ["1", "1", "1", "1"],
                "labels": [1, 2, 2, 1],
                "onset_s": [0.0, 1.0, 2.0, 3.0],
                "individual": ["a", "b", "a", "b"],
            }
        )
        assert trials_matching_labels(df, [1, 2], mode="order", individual="a") == {"1"}
        assert trials_matching_labels(df, [1, 2], mode="order", individual="b") == set()

    def test_no_labels_at_all_matches_nothing(self):
        assert trials_matching_labels(pd.DataFrame(), [1, 2], mode="present") == set()
