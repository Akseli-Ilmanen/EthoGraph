"""Sequence matching and label instance navigation for interval-based browsing."""

from __future__ import annotations

import pandas as pd


def get_label_instances(
    labels_df: pd.DataFrame,
    label_id: int,
    individual: str | None = None,
) -> list[dict]:
    """Find all instances of a label class across all trials.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Full labels DataFrame with ``onset_s``, ``offset_s``, ``labels``,
        ``individual``, ``trial`` columns.
    label_id : int
        Label class ID to filter by.
    individual : str, optional
        If given, restrict to this individual only.

    Returns
    -------
    list[dict]
        Each dict has keys ``trial``, ``onset_s``, ``offset_s``, ``row_idx``,
        ``individual``.  Sorted by (trial, onset_s).
    """
    if labels_df is None or labels_df.empty:
        return []

    mask = labels_df["labels"] == label_id
    if individual is not None:
        mask &= labels_df["individual"] == individual

    filtered = labels_df[mask].sort_values(["trial", "onset_s"])

    return [
        {
            "trial": row["trial"],
            "onset_s": float(row["onset_s"]),
            "offset_s": float(row["offset_s"]),
            "row_idx": idx,
            "individual": row.get("individual"),
        }
        for idx, row in filtered.iterrows()
    ]


def _trial_sequence(group: pd.DataFrame) -> str:
    """Compute the label sequence string for a trial group."""
    sorted_group = group.sort_values("onset_s")
    return "-".join(str(int(lbl)) for lbl in sorted_group["labels"])


def get_unique_sequences(labels_df: pd.DataFrame) -> list[str]:
    """Return all unique label sequences across trials.

    Each sequence is a hyphen-separated string of label IDs in time order,
    e.g. ``"1-2-3-5"``.
    """
    if labels_df is None or labels_df.empty:
        return []

    non_bg = labels_df[labels_df["labels"] > 0]
    if non_bg.empty:
        return []

    sequences = set()
    for _, group in non_bg.groupby("trial", sort=False):
        sequences.add(_trial_sequence(group))

    return sorted(sequences)


def match_sequences(
    labels_df: pd.DataFrame,
    pattern: str,
) -> list[dict]:
    """Find trials whose label sequence matches *pattern*.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Full labels DataFrame.
    pattern : str
        Hyphen-separated label IDs, e.g. ``"1-2-3-5"`` for exact match,
        or a sub-sequence that must appear contiguously within the trial.

    Returns
    -------
    list[dict]
        Each dict has keys ``trial``, ``onset_s``, ``offset_s``,
        ``match_rows``, ``pattern``.  ``onset_s`` / ``offset_s`` span the
        matched sub-sequence.  Sorted by trial.
    """
    if labels_df is None or labels_df.empty or not pattern:
        return []

    target = [int(x.strip()) for x in pattern.split("-") if x.strip()]
    if not target:
        return []

    non_bg = labels_df[labels_df["labels"] > 0]
    if non_bg.empty:
        return []

    matches = []
    for trial_id, group in non_bg.groupby("trial", sort=True):
        sorted_group = group.sort_values("onset_s")
        labels_list = sorted_group["labels"].tolist()
        indices = sorted_group.index.tolist()

        for start in range(len(labels_list) - len(target) + 1):
            window = labels_list[start : start + len(target)]
            if [int(x) for x in window] == target:
                match_indices = indices[start : start + len(target)]
                match_rows = sorted_group.loc[match_indices]
                matches.append(
                    {
                        "trial": trial_id,
                        "onset_s": float(match_rows["onset_s"].iloc[0]),
                        "offset_s": float(match_rows["offset_s"].iloc[-1]),
                        "match_rows": match_indices,
                        "pattern": pattern,
                    }
                )

    return matches
