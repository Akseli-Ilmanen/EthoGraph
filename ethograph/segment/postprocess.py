"""Post-processing of predicted labels: purge → stitch → snap → purge.

These steps run through the same functions the GUI's changepoint correction
uses, so a prediction set written by a script is what the GUI would have
produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ethograph.features.changepoints import correct_changepoints
from ethograph.labels.ml import dense_to_intervals
from ethograph.segment.config import PostprocessConfig
from ethograph.segment.samples import ClassTable, frame_span


def postprocess_intervals(
    intervals: pd.DataFrame, cfg: PostprocessConfig, cp_times: np.ndarray | None = None
) -> pd.DataFrame:
    """Apply the configured steps; snapping only when enabled *and* changepoints exist."""
    if intervals.empty:
        return intervals
    snap = cfg.changepoint_correction and cp_times is not None and len(cp_times) > 0
    return correct_changepoints(
        intervals,
        cp_times if cp_times is not None else np.array([]),
        min_duration_s=cfg.min_duration_s,
        stitch_gap_s=cfg.stitch_gap_s,
        max_expansion_s=cfg.max_expansion_s,
        max_shrink_s=cfg.max_shrink_s,
        label_thresholds_s=cfg.label_thresholds or None,
        do_purge=cfg.min_duration_s > 0 or bool(cfg.label_thresholds),
        do_stitch=cfg.stitch_gap_s > 0,
        do_snap=snap,
        do_purge_after=cfg.min_duration_s > 0 or bool(cfg.label_thresholds),
    )


def postprocess_dense(
    indices: np.ndarray,
    fs: float,
    classes: ClassTable,
    cfg: PostprocessConfig,
    time: np.ndarray | None = None,
    cp_times: np.ndarray | None = None,
) -> np.ndarray:
    """Dense class indices → intervals → post-process → dense class indices."""
    n = len(indices)
    if time is None:
        time = np.arange(n) / fs
    ids = classes.ids(indices)
    intervals = dense_to_intervals(ids, ["_"], time_coord=time)
    intervals = postprocess_intervals(intervals, cfg, cp_times)
    out = np.zeros(n, dtype=np.int64)
    for _, row in intervals.iterrows():
        i0, i1 = frame_span(time, float(row["onset_s"]), float(row["offset_s"]))
        out[i0:i1] = classes.id_to_index.get(int(row["labels"]), 0)
    return out
