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


# ---------------------------------------------------------------------------
# Trial-level label matching — "which trials do (not) look like this?"
# ---------------------------------------------------------------------------
#
# Navigation above answers "where is the next match *inside* a trial". This
# answers "which trials match at all", which is what a filter needs.

#: Match modes, key -> what the key means to a user. The order is the order
#: the Find label inconsistencies dialog offers them in.
LABEL_MATCH_MODES = {
    "present": "All of them occur (any order)",
    "partial": "Some but not all occur",
    "order": "In this order (other labels may come between)",
    "order_strict": "In this order, one straight after another",
}


def _trial_labels(labels_df: pd.DataFrame, individual: str | None = None) -> dict:
    """``{trial: [label ids in time order]}`` over the non-background labels.

    *individual* restricts to one actor: with two animals labelled in one
    trial their events interleave, and an order across both means nothing.
    """
    if labels_df is None or labels_df.empty:
        return {}
    df = labels_df[labels_df["labels"] > 0]
    if individual is not None and "individual" in df.columns:
        df = df[df["individual"].astype(str) == str(individual)]
    if df.empty:
        return {}
    return {
        trial: [int(v) for v in group.sort_values("onset_s")["labels"]]
        for trial, group in df.groupby("trial", sort=False)
    }


def _is_subsequence(target: list[int], present: list[int]) -> bool:
    """Whether *target* appears in *present* in order, gaps allowed."""
    it = iter(present)
    return all(label in it for label in target)


def _has_run(target: list[int], present: list[int]) -> bool:
    """Whether *target* appears in *present* as a contiguous run."""
    n = len(target)
    return any(present[i : i + n] == target for i in range(len(present) - n + 1))


def trial_matches_labels(present: list[int], target: list[int], mode: str) -> bool:
    """Whether one trial's label sequence matches *target* under *mode*.

    *present* is the trial's labels in time order (repeats kept — that is what
    makes ``order_strict`` differ from ``order``), *target* the ids the user
    asked about, in the order they typed them.
    """
    if not target:
        return False
    if mode == "present":
        return set(target).issubset(present)
    if mode == "partial":
        found = {label for label in target if label in present}
        return 0 < len(found) < len(set(target))
    if mode == "order":
        return _is_subsequence(target, present)
    if mode == "order_strict":
        return _has_run(target, present)
    raise ValueError(f"Unknown label match mode {mode!r} (expected one of {', '.join(LABEL_MATCH_MODES)}).")


def trials_matching_labels(
    labels_df: pd.DataFrame,
    target: list[int],
    *,
    mode: str = "present",
    invert: bool = False,
    trials=None,
    individual: str | None = None,
) -> set[str]:
    """Trial ids (as strings) whose labels match *target* under *mode*.

    *trials* is the population to judge — pass the session's full trial list
    so a trial carrying **no** labels is still considered (it matches nothing,
    and therefore matches everything once *invert* is on, which is exactly how
    "find the trials missing this" has to behave). Without it only trials that
    carry at least one label are judged.
    """
    by_trial = _trial_labels(labels_df, individual)
    population = [str(t) for t in trials] if trials is not None else [str(t) for t in by_trial]
    lookup = {str(t): labels for t, labels in by_trial.items()}
    hits = {t for t in population if trial_matches_labels(lookup.get(t, []), target, mode)}
    return set(population) - hits if invert else hits


def parse_label_pattern(text: str) -> list[int]:
    """``"1-2-6-8"`` -> ``[1, 2, 6, 8]``; anything unparseable -> ``[]``.

    The same spelling the Sequence navigate mode takes, so one habit serves
    both. Repeats are kept: ``"6-6"`` asks about two occurrences.
    """
    parts = [p.strip() for p in (text or "").replace(",", "-").split("-")]
    out: list[int] = []
    for part in parts:
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            return []
    return out
