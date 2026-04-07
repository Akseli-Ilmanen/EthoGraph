"""Interval-based label representation and core primitives for EthoGraph.

Labels are stored as a pandas DataFrame with columns:
    onset_s    (float64) - start time in seconds
    offset_s   (float64) - end time in seconds
    labels     (int32)   - label class ID (nonzero)
    individual (str)     - individual identifier
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────

INTERVAL_COLUMNS = ["onset_s", "offset_s", "labels", "individual"]

INTERVAL_DTYPES = {
    "onset_s": np.float64,
    "offset_s": np.float64,
    "labels": np.int32,
    "individual": object,
}


# ── Empty DataFrame ──────────────────────────────────────────────────────

def empty_intervals() -> pd.DataFrame:
    """Create an empty intervals DataFrame with the correct columns and dtypes.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with columns ``onset_s``, ``offset_s``, ``labels``,
        ``individual``.

    Examples
    --------
    >>> from ethograph.labels.intervals import empty_intervals
    >>> df = empty_intervals()
    >>> df.columns.tolist()
    ['onset_s', 'offset_s', 'labels', 'individual']
    >>> len(df)
    0
    """
    return pd.DataFrame(
        {col: pd.Series(dtype=INTERVAL_DTYPES[col]) for col in INTERVAL_COLUMNS}
    )


# ── Mapping loaders ─────────────────────────────────────────────────────

def load_mapping(mapping_file: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load a class-name ↔ index mapping file.

    The file is whitespace-delimited with lines ``<index> <name>``.

    Parameters
    ----------
    mapping_file : str or Path
        Path to the mapping file.

    Returns
    -------
    class_to_idx : dict[str, int]
    idx_to_class : dict[int, str]

    Examples
    --------
    >>> class_to_idx, idx_to_class = load_mapping("mapping.txt")
    >>> class_to_idx["walk"]
    1
    >>> idx_to_class[1]
    'walk'
    """
    class_to_idx: dict[str, int] = {}
    idx_to_class: dict[int, str] = {}
    with open(mapping_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                idx = int(parts[0])
                class_name = parts[1]
                class_to_idx[class_name] = idx
                idx_to_class[idx] = class_name
    return class_to_idx, idx_to_class


_LABEL_COLORS = [
    [1, 1, 1], [255, 102, 178], [102, 158, 255], [153, 51, 255],
    [255, 51, 51], [102, 255, 102], [255, 153, 102], [0, 153, 0],
    [0, 0, 128], [255, 255, 0], [0, 204, 204], [128, 128, 0],
    [255, 0, 255], [255, 165, 0], [0, 128, 255], [7, 7, 215],
    [128, 0, 255], [255, 215, 0], [73, 113, 233], [255, 128, 0],
    [138, 34, 34], [188, 82, 223], [103, 176, 29], [220, 20, 60],
    [3, 243, 3], [147, 24, 147], [178, 111, 44], [16, 166, 166],
    [71, 197, 238], [255, 149, 114], [16, 89, 162], [26, 195, 68],
    [254, 216, 103], [0, 237, 118], [177, 177, 36], [73, 243, 200],
]

_GAP_COLOR = [128 / 255.0, 128 / 255.0, 128 / 255.0]


def load_label_mapping(mapping_file: str | Path = "mapping.txt") -> Dict[int, Dict]:
    """Load a label mapping with colors for visualization.

    Parameters
    ----------
    mapping_file : str or Path
        Path to the mapping file.  Each line is ``<id> <name> [<branch>]``
        where *branch* is an optional integer (default 0) grouping labels
        into branches for independent labeling.

    Returns
    -------
    dict[int, dict]
        ``{label_id: {"name": str, "color": ndarray(3,), "order": int, "branch": int}}``.

    Raises
    ------
    FileNotFoundError
        If *mapping_file* does not exist.

    Examples
    --------
    >>> mappings = load_label_mapping("mapping.txt")
    >>> mappings[1]["name"]
    'walk'
    >>> mappings[1]["color"].shape
    (3,)

    Use the RGB colors to draw labelled rectangles on a plot::

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        mappings = load_label_mapping("mapping.txt")
        fig, ax = plt.subplots()
        ax.plot(time, signal)

        for _, row in intervals_df.iterrows():
            color = mappings[int(row["labels"])]["color"]  # (3,) RGB in [0, 1]
            ax.axvspan(row["onset_s"], row["offset_s"], alpha=0.5, color=color)

        # Build a legend from the mapping
        handles = [
            mpatches.Patch(color=m["color"], label=m["name"])
            for m in mappings.values()
        ]
        ax.legend(handles=handles)
        plt.show()
    """
    mapping_file = Path(mapping_file)
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    label_mappings: dict = {}
    with open(mapping_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if parts[0].startswith("("):
                nums = parts[0].strip("()").split(",")
                label_id = (int(nums[0]), int(nums[1]))
                order = int(parts[-1])
                label_mappings[label_id] = {
                    "name": parts[1],
                    "color": _GAP_COLOR,
                    "order": order,
                    "branch": 0,
                }
            else:
                label_id = int(parts[0])
                if len(parts) >= 3:
                    branch = int(parts[2])
                else:
                    branch = 0
                label_mappings[label_id] = {
                    "name": parts[1],
                    "color": np.array(_LABEL_COLORS[label_id % len(_LABEL_COLORS)]) / 255.0,
                    "order": label_id,
                    "branch": branch,
                }

    return label_mappings


def save_label_mapping(mapping_file: str | Path, mappings: Dict[int, Dict]) -> None:
    """Write a label mapping back to disk, preserving branch assignments.

    Parameters
    ----------
    mapping_file : str or Path
        Destination path.
    mappings : dict[int, dict]
        The mapping dict as returned by :func:`load_label_mapping`.
    """
    mapping_file = Path(mapping_file)
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for label_id, data in sorted(mappings.items(), key=lambda kv: kv[0] if isinstance(kv[0], int) else kv[0][0]):
        if isinstance(label_id, tuple):
            lines.append(f"({label_id[0]},{label_id[1]}) {data['name']} {data['order']}")
        else:
            lines.append(f"{label_id} {data['name']} {data.get('branch', 0)}")
    mapping_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Interval operations ─────────────────────────────────────────────────

def add_interval(
    df: pd.DataFrame,
    onset_s: float,
    offset_s: float,
    labels: int,
    individual: str,
    protected_label_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Add an interval, resolving overlaps for the same individual.

    If the new interval overlaps existing intervals for the same individual,
    the existing intervals are trimmed or split — unless their label ID is in
    *protected_label_ids*, in which case they are kept untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Current intervals DataFrame.
    onset_s, offset_s : float
        Start and end times in seconds.
    labels : int
        Label class ID.
    individual : str
        Individual identifier.
    protected_label_ids : set[int] | None
        Label IDs that must not be trimmed or split (e.g. labels belonging
        to inactive branches).  ``None`` means no protection.

    Returns
    -------
    pd.DataFrame
        Updated intervals DataFrame sorted by ``onset_s``.

    Examples
    --------
    >>> df = empty_intervals()
    >>> df = add_interval(df, 0.0, 1.0, 1, "crow_A")
    >>> df = add_interval(df, 0.5, 1.5, 2, "crow_A")
    >>> len(df)
    2
    >>> float(df.iloc[0]["offset_s"])  # first interval trimmed
    0.499
    """
    if onset_s > offset_s:
        onset_s, offset_s = offset_s, onset_s

    mask_same_ind = df["individual"] == individual
    other = df[~mask_same_ind]
    same = df[mask_same_ind].copy()

    kept: list[dict] = []
    for _, row in same.iterrows():
        ro, rf = row["onset_s"], row["offset_s"]
        rid = row["labels"]

        # Never trim/split intervals from protected (inactive) branches
        if protected_label_ids is not None and int(rid) in protected_label_ids:
            kept.append(row.to_dict())
            continue

        if rf <= onset_s or ro >= offset_s:
            kept.append(row.to_dict())
            continue

        eps = 1e-3
        if ro < onset_s:
            kept.append(
                {"onset_s": ro, "offset_s": onset_s - eps, "labels": rid, "individual": individual}
            )
        if rf > offset_s:
            kept.append(
                {"onset_s": offset_s + eps, "offset_s": rf, "labels": rid, "individual": individual}
            )

    kept.append(
        {"onset_s": onset_s, "offset_s": offset_s, "labels": labels, "individual": individual}
    )

    new_same = _rows_to_df(kept)
    result = pd.concat([other, new_same], ignore_index=True)
    result.sort_values("onset_s", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def delete_interval(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """Drop interval by DataFrame index."""
    return df.drop(index=idx).reset_index(drop=True)


def find_interval_at(df: pd.DataFrame, time_s: float, individual: str) -> int | None:
    """Return DataFrame index of interval containing *time_s* for *individual*.

    Returns ``None`` if no non-background interval contains the time.
    """
    mask = (
        (df["individual"] == individual)
        & (df["onset_s"] <= time_s)
        & (df["offset_s"] >= time_s)
        & (df["labels"] != 0)
    )
    matches = df.index[mask]
    if len(matches) == 0:
        return None
    return int(matches[0])


def get_interval_bounds(df: pd.DataFrame, idx: int) -> tuple[float, float, int]:
    """Return ``(onset_s, offset_s, labels)`` for interval at *idx*."""
    row = df.loc[idx]
    return float(row["onset_s"]), float(row["offset_s"]), int(row["labels"])


def purge_short_intervals(
    df: pd.DataFrame,
    min_duration_s: float,
    label_thresholds_s: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Drop intervals shorter than a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Intervals DataFrame.
    min_duration_s : float
        Default minimum duration in seconds.
    label_thresholds_s : dict[int, float], optional
        Per-label minimum durations (overrides *min_duration_s*).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.

    Examples
    --------
    >>> df = add_interval(empty_intervals(), 0.0, 0.01, 1, "A")
    >>> df = add_interval(df, 1.0, 2.0, 2, "A")
    >>> purged = purge_short_intervals(df, min_duration_s=0.1)
    >>> len(purged)
    1
    """
    if label_thresholds_s is None:
        label_thresholds_s = {}

    durations = df["offset_s"] - df["onset_s"]
    thresholds = df["labels"].map(
        lambda lid: label_thresholds_s.get(lid, min_duration_s)
    )
    keep = durations >= thresholds
    return df[keep].reset_index(drop=True)


def stitch_intervals(
    df: pd.DataFrame,
    max_gap_s: float,
    individual: str | None = None,
) -> pd.DataFrame:
    """Merge adjacent same-label intervals where gap <= *max_gap_s*.

    Parameters
    ----------
    df : pd.DataFrame
        Intervals DataFrame.
    max_gap_s : float
        Maximum gap (seconds) between intervals to merge.
    individual : str, optional
        If given, only stitch intervals for this individual.

    Returns
    -------
    pd.DataFrame
        Stitched intervals DataFrame.

    Examples
    --------
    >>> df = add_interval(empty_intervals(), 0.0, 1.0, 1, "A")
    >>> df = add_interval(df, 1.05, 2.0, 1, "A")
    >>> stitched = stitch_intervals(df, max_gap_s=0.1)
    >>> len(stitched)
    1
    >>> float(stitched.iloc[0]["offset_s"])
    2.0
    """
    if df.empty:
        return df.copy()

    if individual is not None:
        mask = df["individual"] == individual
        other = df[~mask]
        target = df[mask].copy()
    else:
        other = empty_intervals()
        target = df.copy()

    target.sort_values(["individual", "onset_s"], inplace=True)
    target.reset_index(drop=True, inplace=True)

    merged: list[dict] = []
    i = 0
    while i < len(target):
        row = target.iloc[i]
        current = row.to_dict()
        j = i + 1
        while j < len(target):
            nxt = target.iloc[j]
            if (
                nxt["individual"] == current["individual"]
                and nxt["labels"] == current["labels"]
                and (nxt["onset_s"] - current["offset_s"]) < max_gap_s
            ):
                current["offset_s"] = nxt["offset_s"]
                j += 1
            else:
                break
        merged.append(current)
        i = j

    result = pd.concat([other, _rows_to_df(merged)], ignore_index=True)
    result.sort_values("onset_s", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def snap_boundaries(
    df: pd.DataFrame,
    cp_times: np.ndarray,
    max_expansion_s: float,
    max_shrink_s: float,
) -> pd.DataFrame:
    """Snap interval onset/offset to nearest changepoint times.

    Parameters
    ----------
    df : pd.DataFrame
        Intervals DataFrame.
    cp_times : np.ndarray
        Candidate changepoint times.
    max_expansion_s : float
        Maximum allowed expansion (seconds).
    max_shrink_s : float
        Maximum allowed shrinkage (seconds).

    Returns
    -------
    pd.DataFrame
        Snapped intervals with overlaps resolved.
    """
    if df.empty or len(cp_times) == 0:
        return df.copy()

    cp_times = np.sort(cp_times)
    rows = []

    for _, row in df.iterrows():
        onset = row["onset_s"]
        offset = row["offset_s"]

        snap_onset = _snap_onset(onset, cp_times, max_expansion_s, max_shrink_s)
        snap_offset = _snap_offset(offset, cp_times, max_expansion_s, max_shrink_s)

        if snap_onset >= snap_offset:
            snap_onset = onset
            snap_offset = offset

        rows.append({
            "onset_s": snap_onset,
            "offset_s": snap_offset,
            "labels": row["labels"],
            "individual": row["individual"],
        })

    result = _rows_to_df(rows)
    result.sort_values(["individual", "onset_s"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    result = _resolve_overlaps(result)
    return result


# ── Private helpers ──────────────────────────────────────────────────────

def _snap_onset(boundary, cp_times, max_expansion_s, max_shrink_s):
    nearest_idx = np.argmin(np.abs(cp_times - boundary))
    cp_val = float(cp_times[nearest_idx])
    expansion = boundary - cp_val
    shrink = cp_val - boundary
    if expansion > max_expansion_s or shrink > max_shrink_s:
        return boundary
    return cp_val


def _snap_offset(boundary, cp_times, max_expansion_s, max_shrink_s):
    nearest_idx = np.argmin(np.abs(cp_times - boundary))
    cp_val = float(cp_times[nearest_idx])
    expansion = cp_val - boundary
    shrink = boundary - cp_val
    if expansion > max_expansion_s or shrink > max_shrink_s:
        return boundary
    return cp_val


def _resolve_overlaps(df: pd.DataFrame, eps: float = 1e-3) -> pd.DataFrame:
    if df.empty:
        return df

    groups = []
    for ind, group in df.groupby("individual", sort=False):
        group = group.sort_values("onset_s").reset_index(drop=True)
        for i in range(len(group) - 1):
            if group.at[i, "offset_s"] > group.at[i + 1, "onset_s"]:
                if group.at[i, "labels"] != group.at[i + 1, "labels"]:
                    group.at[i, "offset_s"] = group.at[i + 1, "onset_s"] - eps
            elif group.at[i, "offset_s"] == group.at[i + 1, "onset_s"]:
                group.at[i, "offset_s"] = group.at[i + 1, "onset_s"] - eps
        groups.append(group)

    result = pd.concat(groups, ignore_index=True)
    result.sort_values("onset_s", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_intervals()
    df = pd.DataFrame(rows, columns=INTERVAL_COLUMNS)
    for col, dtype in INTERVAL_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df
