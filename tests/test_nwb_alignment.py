"""Tests for ethograph.io.nwb_alignment — trials_ep generation."""

import numpy as np
import pandas as pd
import pytest

from ethograph.io.nwb_alignment import (
    EmpytAlignment,
    NWBAlignment,
    TableAlignment,
)


# ---------------------------------------------------------------------------
# trials_ep must exist when alignment has start/stop times
# ---------------------------------------------------------------------------


class TestTrialsEp:
    """If the alignment table has start_time and stop_time, trials_ep must be non-None."""

    def _make_table_alignment(self, n_trials: int = 5, with_trial_col: bool = True) -> TableAlignment:
        rng = np.random.default_rng(42)
        gaps = rng.uniform(2.0, 5.0, n_trials)
        starts = np.cumsum(gaps)
        durations = rng.uniform(0.5, 1.5, n_trials)
        stops = starts + durations
        df = pd.DataFrame({"start_time": starts, "stop_time": stops})
        if with_trial_col:
            df["trial"] = list(range(1, n_trials + 1))
        return TableAlignment(df)

    def test_table_alignment_trials_ep(self):
        alignment = self._make_table_alignment()
        ep = alignment.trials_ep
        assert ep is not None, "trials_ep must not be None when start/stop times are present"
        assert len(ep) == 5

    def test_empty_alignment_trials_ep_none(self):
        alignment = EmpytAlignment()
        assert alignment.trials_ep is None

    def test_trial_metadata_ids(self):
        alignment = self._make_table_alignment()
        ep = alignment.trials_ep
        assert ep is not None
        trial_ids = ep.metadata["trial"]
        assert list(trial_ids) == [1, 2, 3, 4, 5]

    def test_no_trial_column(self):
        """NWB trials tables often lack a 'trial' column — trials_ep should still work."""
        alignment = self._make_table_alignment(n_trials=3, with_trial_col=False)
        ep = alignment.trials_ep
        assert ep is not None, "trials_ep must work even without a 'trial' column"
        assert len(ep) == 3
        # Auto-generated trial IDs should be 1-indexed
        assert list(ep.metadata["trial"]) == [1, 2, 3]

    def test_consistency_start_stop_and_ep(self):
        """If alignment has start/stop for a trial, trials_ep must cover that trial."""
        alignment = self._make_table_alignment(n_trials=10)
        ep = alignment.trials_ep
        assert ep is not None
        for i, trial_id in enumerate(range(1, 11)):
            start = alignment.start_time(trial_id)
            stop = alignment.stop_time(trial_id)
            if stop is not None and stop > start:
                ep_start = float(ep.start[i])
                ep_end = float(ep.end[i])
                assert abs(ep_start - start) < 1e-6
                assert abs(ep_end - stop) < 1e-6
