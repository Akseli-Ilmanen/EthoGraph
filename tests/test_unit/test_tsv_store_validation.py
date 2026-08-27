"""``validate_labels_tsv``: required columns must exist, and never be blank.

Pure pandas — no Qt involved.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ethograph.labels.tsv_store import load_labels_tsv, validate_labels_tsv


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": [1, 2],
            "individual": ["crow_A", "crow_B"],
            "labels": [1, 2],
            "onset_s": [0.0, 1.0],
            "offset_s": [0.5, float("nan")],  # a point event: blank offset is fine
        }
    )


def test_valid_frame_passes():
    validate_labels_tsv(_valid_df())


def test_missing_column_raises():
    df = _valid_df().drop(columns=["individual"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_labels_tsv(df)


@pytest.mark.parametrize("col,blank", [("trial", None), ("individual", ""), ("labels", None), ("onset_s", None)])
def test_blank_required_value_raises(col, blank):
    df = _valid_df()
    df.loc[0, col] = blank
    with pytest.raises(ValueError, match=f"missing '{col}' value"):
        validate_labels_tsv(df)


def test_whitespace_only_individual_raises():
    df = _valid_df()
    df.loc[0, "individual"] = "   "
    with pytest.raises(ValueError, match="missing 'individual' value"):
        validate_labels_tsv(df)


def test_blank_offset_never_flagged():
    # offset_s is deliberately excluded from the nonnull check: point events
    # store it as NaN by design (see labels/intervals.py: EVENT_TYPE_POINT).
    df = _valid_df()
    df["offset_s"] = float("nan")
    validate_labels_tsv(df)


def test_load_labels_tsv_raises_on_blank_trial(tmp_path):
    # Written by hand (not via to_csv) so the blank cell round-trips as an
    # empty string rather than raising on assignment into an int64 column.
    path = tmp_path / "bad_labels.tsv"
    path.write_text(
        "trial\tindividual\tlabels\tonset_s\toffset_s\n1\tcrow_A\t1\t0.0\t0.5\n\tcrow_B\t2\t1.0\t1.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing 'trial' value"):
        load_labels_tsv(path)
