"""Dense (array-based) label operations for ML pipelines.

This module provides tools for converting between the interval-based label
format (used by the GUI and TSV storage) and dense integer arrays (used by
ML models).  It also contains post-processing operations commonly applied
to model predictions before evaluation or storage.

Typical ML workflow
-------------------
1. **Load labels from TSV** → ``pd.DataFrame`` with ``onset_s``, ``offset_s``,
   ``labels``, ``individual`` (plus ``n_samples`` per-trial metadata).
2. **Convert to dense** → ``intervals_to_dense(df, sample_rate, individuals, n_samples)``
   gives an ``(n_samples, n_individuals)`` int8 array ready for training.
3. **Run model** → get a dense prediction array of shape ``(T,)`` or ``(T, n_classes)``.
4. **Post-process** → ``purge_small_blocks`` → ``stitch_gaps`` → ``fix_endings``.
5. **Convert back** → ``dense_to_intervals(pred, individuals, sample_rate=sr)``
   gives an intervals DataFrame for storage or evaluation.

The ``n_samples`` value stored in the TSV file (per-trial metadata) tells you
exactly how long the dense array should be — you only need to additionally
know the ``sample_rate`` to drive the conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ethograph.labels.intervals import _rows_to_df


# ── Primitives ───────────────────────────────────────────────────────────

def find_blocks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find contiguous True blocks in a boolean array.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array.

    Returns
    -------
    starts : np.ndarray
        Start indices of True blocks.
    ends : np.ndarray
        End indices (inclusive) of True blocks.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([False, True, True, False, True])
    >>> starts, ends = find_blocks(mask)
    >>> starts
    array([1, 4])
    >>> ends
    array([2, 4])
    """
    padded = np.concatenate(([0], mask.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return starts, ends


def _get_segments(col, bg_class=0):
    """Find contiguous labeled segments in a 1-D array.

    Example: ``[0,1,1,1,0,2,2]`` → ``[(1,1,4), (2,5,7)]``
    Each tuple is ``(label, start_index, end_index_exclusive)``.
    """
    padded = np.concatenate([[-1], col, [-1]])
    change_indices = np.nonzero(padded[:-1] != padded[1:])[0]

    segments = []
    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end = change_indices[i + 1]
        label = int(col[start])
        if label != bg_class:
            segments.append((label, start, end))
    return segments


def _get_labels_start_end_times(col, time_coord, individual, bg_class=0):
    """Convert segments to time intervals (inclusive end)."""
    segments = _get_segments(col, bg_class)
    return [{
        "onset_s": float(time_coord[start]),
        "offset_s": float(time_coord[end - 1]),
        "labels": label,
        "individual": individual,
    } for label, start, end in segments]


# ── Interval ↔ Dense conversion ─────────────────────────────────────────

def dense_to_intervals(
    dense_array: np.ndarray,
    individuals: list[str],
    *,
    sample_rate: float | None = None,
    time_coord: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert a dense label array to an intervals DataFrame.

    Provide either *sample_rate* (uniform spacing starting at t = 0) or an
    explicit *time_coord* array.

    Parameters
    ----------
    dense_array : np.ndarray
        Shape ``(n_samples,)`` for a single individual, or
        ``(n_samples, n_individuals)`` for multiple.
    individuals : list[str]
        Individual identifiers — length must match the second axis.
    sample_rate : float, optional
        Sampling rate in Hz.  Timestamps are computed as
        ``np.arange(n_samples) / sample_rate``.
    time_coord : np.ndarray, optional
        Explicit time array of length ``n_samples``.  Use this when timestamps
        are non-uniform or do not start at zero.

    Returns
    -------
    pd.DataFrame
        Intervals with columns ``onset_s``, ``offset_s``, ``labels``,
        ``individual``.  ``offset_s`` is **inclusive** (last sample of the
        segment).

    Raises
    ------
    ValueError
        If neither *sample_rate* nor *time_coord* is given, or if the number
        of individuals does not match the array width.

    Examples
    --------
    Convert a 1-D dense array at 10 Hz:

    >>> import numpy as np
    >>> from ethograph.labels.ml import dense_to_intervals
    >>> labels = np.array([0, 1, 1, 1, 0, 2, 2])
    >>> df = dense_to_intervals(labels, ["crow_A"], sample_rate=10.0)
    >>> df[["onset_s", "offset_s", "labels"]].values.tolist()
    [[0.1, 0.3, 1], [0.5, 0.6, 2]]

    With explicit timestamps:

    >>> times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> df = dense_to_intervals(labels, ["crow_A"], time_coord=times)
    >>> df["onset_s"].tolist()
    [0.1, 0.5]
    """
    dense_array = np.asarray(dense_array)

    if sample_rate is None and time_coord is None:
        raise ValueError("Provide either sample_rate or time_coord")

    if time_coord is None:
        time_coord = np.arange(dense_array.shape[0]) / sample_rate
    else:
        time_coord = np.asarray(time_coord)

    if dense_array.ndim == 1:
        dense_array = dense_array[:, np.newaxis]

    if dense_array.shape[1] != len(individuals):
        raise ValueError(
            f"dense_array has {dense_array.shape[1]} columns but "
            f"{len(individuals)} individuals given"
        )

    rows: list[dict] = []
    for ind_idx, ind_name in enumerate(individuals):
        col = dense_array[:, ind_idx]
        rows.extend(_get_labels_start_end_times(col, time_coord, str(ind_name)))

    return _rows_to_df(rows)


def intervals_to_dense(
    df: pd.DataFrame,
    sample_rate: float,
    individuals: list[str],
    n_samples: int,
) -> np.ndarray:
    """Convert an intervals DataFrame to a dense label array.

    Each interval is mapped onto the nearest sample indices using
    ``round(time * sample_rate)``.  Overlapping intervals for the same
    individual are resolved by last-write-wins.

    Parameters
    ----------
    df : pd.DataFrame
        Intervals DataFrame with columns ``onset_s``, ``offset_s``, ``labels``,
        ``individual``.
    sample_rate : float
        Sampling rate in Hz (e.g. 30.0 for 30 fps video features).
    individuals : list[str]
        Individual identifiers.  The output column order matches this list.
    n_samples : int
        Number of output time steps.  Typically available as per-trial
        ``n_samples`` metadata in the TSV file.

    Returns
    -------
    np.ndarray
        Dense label array of shape ``(n_samples, len(individuals))``, dtype
        ``int8``.  Background (unlabeled) time steps are 0.

    Examples
    --------
    >>> import pandas as pd
    >>> from ethograph.labels.ml import intervals_to_dense
    >>> df = pd.DataFrame({
    ...     "onset_s": [0.1, 0.5], "offset_s": [0.3, 0.6],
    ...     "labels": [1, 2], "individual": ["A", "A"],
    ... })
    >>> dense = intervals_to_dense(df, sample_rate=10.0, individuals=["A"], n_samples=7)
    >>> dense[:, 0].tolist()
    [0, 1, 1, 1, 0, 2, 2]
    """
    dense = np.zeros((n_samples, len(individuals)), dtype=np.int8)
    ind_to_idx = {name: i for i, name in enumerate(individuals)}

    for _, row in df.iterrows():
        ind_idx = ind_to_idx.get(row["individual"])
        if ind_idx is None:
            continue
        start_idx = int(round(row["onset_s"] * sample_rate))
        end_idx = int(round(row["offset_s"] * sample_rate))
        start_idx = max(0, start_idx)
        end_idx = min(n_samples - 1, end_idx)
        dense[start_idx : end_idx + 1, ind_idx] = int(row["labels"])

    return dense


# ── Segment index extraction ────────────────────────────────────────────

def get_labels_start_end_indices(col, bg_class=0):
    """Return segment boundaries as sample indices (exclusive end).

    Useful for slicing dense arrays or computing segment-level metrics.

    Parameters
    ----------
    col : array-like
        1-D dense label array.
    bg_class : int
        Background class to ignore (default 0).

    Returns
    -------
    labels : list[int]
        Label class for each segment.
    starts : list[int]
        Start index (inclusive) of each segment.
    ends : list[int]
        End index (**exclusive**) — use ``array[start:end]`` to slice.

    Examples
    --------
    >>> from ethograph.labels.ml import get_labels_start_end_indices
    >>> labels, starts, ends = get_labels_start_end_indices([0,1,1,1,0,2,2])
    >>> labels
    [1, 2]
    >>> starts
    [1, 5]
    >>> ends
    [4, 7]
    >>> # To extract the first segment from a feature array:
    >>> # segment_features = features[starts[0]:ends[0], :]
    """
    segments = _get_segments(col, bg_class)
    labels = [s[0] for s in segments]
    starts = [s[1] for s in segments]
    ends = [s[2] for s in segments]
    return labels, starts, ends


# ── Dense post-processing ────────────────────────────────────────────────

def stitch_gaps(
    labels: np.ndarray,
    max_gap_len: int,
    skip_labels: set[int] | None = None,
) -> np.ndarray:
    """Fill small background gaps between same-label segments.

    Scans the dense label array for short runs of zeros (background) flanked
    by the same non-zero label on both sides.  When the gap is at most
    *max_gap_len* samples, it is filled with that label.

    This is typically used **after** model prediction to clean up fragmented
    outputs where a behaviour is briefly interrupted by a few background frames.

    Parameters
    ----------
    labels : np.ndarray
        1-D dense label array (int), where 0 = background.
    max_gap_len : int
        Maximum gap length **in samples** to fill.  Gaps longer than this are
        left untouched.  Convert from seconds: ``int(gap_s * sample_rate)``.
    skip_labels : set[int], optional
        Labels whose trailing gaps should **never** be filled.  For example,
        ``skip_labels={3}`` means that a gap preceded by label 3 is always kept
        even if the same label follows.

    Returns
    -------
    np.ndarray
        Copy of *labels* with qualifying gaps filled.

    Examples
    --------
    Basic gap stitching at 30 Hz (fill gaps up to 5 frames):

    >>> import numpy as np
    >>> from ethograph.labels.ml import stitch_gaps
    >>> pred = np.array([1, 1, 0, 1, 1, 0, 0, 0, 2, 2])
    >>> stitch_gaps(pred, max_gap_len=2).tolist()
    [1, 1, 1, 1, 1, 0, 0, 0, 2, 2]

    The single-frame gap (index 2) is filled because label 1 appears on both
    sides.  The 3-frame gap (indices 5–7) is left alone because it exceeds
    ``max_gap_len=2``.

    Using ``skip_labels`` to protect specific transitions:

    >>> pred = np.array([3, 3, 0, 3, 3])
    >>> stitch_gaps(pred, max_gap_len=2, skip_labels={3}).tolist()
    [3, 3, 0, 3, 3]
    """
    if skip_labels is None:
        skip_labels = set()

    stitched = labels.copy()
    zero_starts, zero_ends = find_blocks(labels == 0)

    for start, end in zip(zero_starts, zero_ends):
        gap_len = end - start

        if gap_len > max_gap_len:
            continue

        left_label = labels[start - 1] if start > 0 else 0
        right_label = labels[end + 1] if end < len(labels) - 1 else 0

        if left_label in skip_labels:
            continue

        if left_label != 0 and left_label == right_label:
            stitched[start:end + 1] = left_label

    return stitched


def purge_small_blocks(
    labels: np.ndarray,
    min_length: int,
    label_thresholds: dict[int, int] | None = None,
) -> np.ndarray:
    """Remove label blocks shorter than a threshold (set to background).

    Scans the dense label array for contiguous runs of the same non-zero
    label.  If a run is shorter than its threshold, every sample in it is
    set to 0 (background).

    This is the dense-array counterpart of
    :func:`~ethograph.labels.intervals.purge_short_intervals` (which works
    in seconds on interval DataFrames).

    Parameters
    ----------
    labels : np.ndarray
        1-D dense label array (int), where 0 = background.
    min_length : int
        Default minimum block length **in samples**.  Blocks shorter than
        this are zeroed out.  Convert from seconds:
        ``int(min_duration_s * sample_rate)``.
    label_thresholds : dict[int, int], optional
        Per-label minimum lengths that override *min_length*.  For example,
        ``{1: 10, 3: 30}`` means label 1 needs ≥10 samples and label 3
        needs ≥30 samples; all other labels use *min_length*.

    Returns
    -------
    np.ndarray
        Copy of *labels* with short blocks zeroed out.

    Examples
    --------
    Remove any block shorter than 3 samples:

    >>> import numpy as np
    >>> from ethograph.labels.ml import purge_small_blocks
    >>> pred = np.array([0, 1, 0, 2, 2, 2, 2, 0])
    >>> purge_small_blocks(pred, min_length=3).tolist()
    [0, 0, 0, 2, 2, 2, 2, 0]

    With per-label thresholds (label 2 needs ≥5 samples):

    >>> purge_small_blocks(pred, min_length=1, label_thresholds={2: 5}).tolist()
    [0, 1, 0, 0, 0, 0, 0, 0]

    Typical pipeline — purge then stitch:

    >>> pred = np.array([1,1,1, 0, 1, 0, 1,1,1])
    >>> cleaned = purge_small_blocks(pred, min_length=2)  # remove isolated 1-sample
    >>> cleaned.tolist()
    [1, 1, 1, 0, 0, 0, 1, 1, 1]
    >>> stitch_gaps(cleaned, max_gap_len=4).tolist()
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
    """
    labels = np.asarray(labels)
    if len(labels) == 0:
        return labels.copy()

    if label_thresholds is None:
        label_thresholds = {}
    else:
        label_thresholds = {int(k): v for k, v in label_thresholds.items()}

    output = labels.copy()

    padded = np.concatenate([[-1], labels, [-1]])
    change_mask = padded[:-1] != padded[1:]
    change_indices = np.nonzero(change_mask)[0]

    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i + 1]

        if start_idx >= len(labels):
            continue

        label_val = int(labels[start_idx])
        if label_val == 0:
            continue

        threshold = label_thresholds.get(label_val, min_length)
        run_length = end_idx - start_idx

        if run_length < threshold:
            output[start_idx:end_idx] = 0

    return output


def fix_endings(labels: np.ndarray, changepoints) -> np.ndarray:
    """Extend label endings by one sample at changepoint boundaries.

    When a labelled segment ends and the very next sample is a changepoint,
    the segment is extended by one sample.  This accounts for the common
    off-by-one between predicted segment boundaries and detected changepoints.

    Parameters
    ----------
    labels : np.ndarray
        1-D dense label array (int).
    changepoints : array-like
        Either a boolean mask of the same length (True = changepoint), or an
        array of integer changepoint indices.

    Returns
    -------
    np.ndarray
        Copy of *labels* with qualifying endings extended by one sample.

    Examples
    --------
    >>> import numpy as np
    >>> from ethograph.labels.ml import fix_endings
    >>> labels = np.array([0, 1, 1, 0, 0, 2, 2, 0])
    >>> cps     = np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    >>> fix_endings(labels, cps).tolist()
    [0, 1, 1, 1, 0, 2, 2, 2]

    The segment of label 1 ended at index 2, and index 3 is a changepoint,
    so label 1 is extended to index 3.  Same for label 2 at index 7.
    """
    labels_out = np.array(labels).reshape(-1)

    changepoints_arr = np.array(changepoints)
    if changepoints_arr.dtype == bool or (
        changepoints_arr.dtype == int and set(np.unique(changepoints_arr)).issubset({0, 1})
    ):
        changepoints_idxs = set(np.where(changepoints_arr)[0])
    else:
        changepoints_idxs = set(changepoints)

    is_nonzero = labels_out != 0
    is_zero_next = np.concatenate([labels_out[1:] == 0, [False]])
    segment_ends = np.where(is_nonzero & is_zero_next)[0]

    for seg_end in segment_ends:
        if (seg_end + 1) in changepoints_idxs:
            if labels_out[seg_end] != 0 and labels_out[seg_end + 1] == 0:
                labels_out[seg_end + 1] = labels_out[seg_end]

    return labels_out
