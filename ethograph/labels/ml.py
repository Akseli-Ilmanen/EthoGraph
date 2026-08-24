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
4. **Convert back** → ``dense_to_intervals(pred, individuals, sample_rate=sr)``
   gives an intervals DataFrame for storage or evaluation.
5. **Post-process** on the intervals — purge, stitch, snap — through
   :func:`ethograph.features.changepoints.correct_changepoints`; there is no
   dense post-processing.

The ``n_samples`` value stored in the TSV file (per-trial metadata) tells you
exactly how long the dense array should be — you only need to additionally
know the ``sample_rate`` to drive the conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ethograph.labels.intervals import _rows_to_df, states_only

# ── Primitives ───────────────────────────────────────────────────────────


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
    return [
        {
            "onset_s": float(time_coord[start]),
            "offset_s": float(time_coord[end - 1]),
            "labels": label,
            "individual": individual,
        }
        for label, start, end in segments
    ]


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
        raise ValueError(f"dense_array has {dense_array.shape[1]} columns but {len(individuals)} individuals given")

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
    >>> df = pd.DataFrame(
    ...     {
    ...         "onset_s": [0.1, 0.5],
    ...         "offset_s": [0.3, 0.6],
    ...         "labels": [1, 2],
    ...         "individual": ["A", "A"],
    ...     }
    ... )
    >>> dense = intervals_to_dense(df, sample_rate=10.0, individuals=["A"], n_samples=7)
    >>> dense[:, 0].tolist()
    [0, 1, 1, 1, 0, 2, 2]
    """
    dense = np.zeros((n_samples, len(individuals)), dtype=np.int8)
    ind_to_idx = {name: i for i, name in enumerate(individuals)}
    df = states_only(df)

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
    >>> labels, starts, ends = get_labels_start_end_indices([0, 1, 1, 1, 0, 2, 2])
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
