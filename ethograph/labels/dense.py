"""Dense (array-based) label operations for ML pipelines."""

from __future__ import annotations

from typing import Dict, Union

import numpy as np

from ethograph.labels.core import find_blocks, get_segments


def get_labels_start_end_indices(col, bg_class=0):
    """Returns indices for array slicing (exclusive end).

    Example: [0,1,1,1,0,2,2] → labels=[1,2], starts=[1,5], ends=[4,7]
    """
    segments = get_segments(col, bg_class)
    labels = [s[0] for s in segments]
    starts = [s[1] for s in segments]
    ends = [s[2] for s in segments]
    return labels, starts, ends


def stitch_gaps(labels: np.ndarray, max_gap_len: int) -> np.ndarray:
    stitched = labels.copy()
    zero_starts, zero_ends = find_blocks(labels == 0)

    for start, end in zip(zero_starts, zero_ends):
        gap_len = end - start

        if gap_len > max_gap_len:
            continue

        left_label = labels[start - 1] if start > 0 else 0
        right_label = labels[end + 1] if end < len(labels) - 1 else 0

        # Toss exception - HARD CODED
        if left_label == 3:
            continue

        if left_label != 0 and left_label == right_label:
            stitched[start:end + 1] = left_label

    return stitched


def purge_small_blocks(
    labels: np.ndarray,
    min_length: int,
    label_thresholds: Dict[Union[int, str], int] = None,
) -> np.ndarray:
    """Remove label blocks shorter than their threshold (set to 0)."""
    if isinstance(labels, (str, bytes)):
        labels = np.array([int(c) for c in str(labels)])
    else:
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


def fix_endings(labels, changepoints):
    """Extend label endings by one sample when a changepoint falls at the boundary."""
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
