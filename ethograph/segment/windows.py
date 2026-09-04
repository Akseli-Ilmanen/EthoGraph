"""Windows tiled over an epoch, as the trials of a second alignment.

Inference runs per trial, and a trial comes from the alignment NWB — so a
stretch of a recording with no trials of its own (a sleep epoch, in which
replay of the behaviour is the question) is predicted by tiling it into
contiguous windows and writing those as the trials table of an alignment
listed under ``sessions[].alignment``. Nothing about the source file changes:
the same ``units.npz`` is one session with its behaviour trials and another
with these windows.

Windows never overlap — the alignment loader clips any trial reaching the
next one's start, and a pynapple ``IntervalSet`` merges overlaps outright.
Overlap, where wanted, is a second tiling shifted by a phase in a second
alignment. The model matters too: a window longer than a trial is only safe
for an architecture whose receptive field the training trials filled
(``mstcn`` with its layers cut to the trial length); C2F-TCN pools over the
whole input and has only ever seen trial-sized ones.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ethograph.io.nwb_alignment import _EPOCH_GAP

logger = logging.getLogger(__name__)

#: Metadata column the tiled trials carry, so ``trials.where`` can tell them apart.
STATE_COLUMN = "state"


def tile_windows(start_s: float, end_s: float, window_s: float, *, first_id: int = 1, state: str = "sleep") -> Any:
    """Contiguous windows of *window_s* seconds covering ``[start_s, end_s)`` as a trials ``IntervalSet``.

    The last window is whatever is left, so the epoch is covered without a
    gap; it is dropped only when it would be shorter than a second, too
    short to carry a prediction. Trial ids count up from *first_id*, and
    every row carries ``state`` so the windows can be filtered, or told
    apart from behaviour trials, by ``trials.where``.
    """
    import pynapple as nap

    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be after start_s ({start_s})")
    starts = np.arange(start_s, end_s, window_s, dtype=np.float64)
    # Each window stops a hair before the next begins: the alignment loader
    # clips a trial that reaches the next one's start by this same gap, and
    # pynapple takes a start equal to the previous end as an overlap.
    ends = np.minimum(starts + window_s, end_s) - _EPOCH_GAP
    keep = (ends - starts) >= 1.0
    if not keep.all():
        logger.info("Dropping a final window of %.3g s (shorter than a second)", float((ends - starts)[~keep][0]))
    starts, ends = starts[keep], ends[keep]
    if starts.size == 0:
        raise ValueError(f"[{start_s}, {end_s}) holds no window of at least a second")
    ep = nap.IntervalSet(start=starts, end=ends)
    ep.set_info(trial=np.arange(first_id, first_id + starts.size), **{STATE_COLUMN: [state] * starts.size})
    return ep


def write_windows_alignment(
    path: str | Path, start_s: float, end_s: float, window_s: float, *, first_id: int = 1, state: str = "sleep"
) -> Path:
    """Write the alignment NWB whose trials are :func:`tile_windows` of the epoch; returns its path."""
    from ethograph.io.nwb_alignment import alignment_from_trials_ep

    ep = tile_windows(start_s, end_s, window_s, first_id=first_id, state=state)
    written = alignment_from_trials_ep(ep, path)
    logger.info(
        "Wrote %d windows of %g s over [%g, %g) s → %s",
        len(ep),
        window_s,
        float(ep.start[0]),
        float(ep.end[-1]),
        written,
    )
    return written
