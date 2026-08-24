"""Segmentation metrics: frame accuracy, edit score, segmental F1@k, frame F1,
class-wise F1, IoUs and boundary deltas.

All inputs are dense class-index arrays (0 = background) per sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ethograph.labels.ml import get_labels_start_end_indices

TEST_METRICS_FILE = "test_metrics.yaml"
EVAL_ARRAYS_FILE = "test_eval.npz"
"""The matched-segment IoUs and onset/offset deltas :func:`evaluate` returns —
too bulky for the YAML, so they live beside it as one ``.npz`` per run. See
:func:`save_eval_arrays` / :func:`load_eval_arrays`, and
:mod:`ethograph.segment.plotting` for the cross-run comparison figure that
reads them back."""


def levenshtein(p: list, y: list, norm: bool = False) -> float:
    m, n = len(p), len(y)
    d = np.zeros((m + 1, n + 1), dtype=np.float64)
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if y[j - 1] == p[i - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1)
    if norm:
        return (1 - d[m, n] / max(m, n)) * 100 if max(m, n) > 0 else 100.0
    return float(d[m, n])


def edit_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p_labels, _, _ = get_labels_start_end_indices(pred, 0)
    y_labels, _, _ = get_labels_start_end_indices(gt, 0)
    return levenshtein(list(p_labels), list(y_labels), norm=True)


def segment_matches(
    pred: np.ndarray, gt: np.ndarray, overlap: float
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """TP/FP/FN at IoU ≥ *overlap*, plus per-predicted-segment IoU and the
    start/end deltas (frames) of matched segments."""
    p_label, p_start, p_end = get_labels_start_end_indices(pred, 0)
    y_label, y_start, y_end = get_labels_start_end_indices(gt, 0)
    y_start, y_end = np.asarray(y_start), np.asarray(y_end)
    tp = fp = 0
    hits = np.zeros(len(y_label), dtype=bool)
    ious = np.zeros(len(p_label))
    starts, ends = [], []
    for j in range(len(p_label)):
        if len(y_label) == 0:
            fp += 1
            continue
        inter = np.maximum(0, np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start))
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        same = np.array([p_label[j] == yl for yl in y_label])
        iou = (inter / (union + 1e-10)) * same
        idx = int(np.argmax(iou))
        ious[j] = iou[idx]
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = True
            starts.append(abs(y_start[idx] - p_start[j]))
            ends.append(abs(y_end[idx] - p_end[j]))
        else:
            fp += 1
    fn = len(y_label) - int(hits.sum())
    return float(tp), float(fp), float(fn), ious, np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def _f1(tp: float, fp: float, fn: float) -> float:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 100.0 * 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(f1)


def metric_key(threshold: float) -> str:
    return f"f1@{int(round(threshold * 100))}"


def evaluate(
    gt: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    thresholds: list[float],
    fs: float,
) -> dict[str, Any]:
    """Overall + class-wise metrics over matching keys of *gt* and *pred*."""
    keys = [k for k in gt if k in pred]
    if not keys:
        raise ValueError("No samples to evaluate.")
    correct = total = 0
    frame_tp = frame_fp = frame_fn = 0
    edit = 0.0
    tp = np.zeros(len(thresholds))
    fp = np.zeros(len(thresholds))
    fn = np.zeros(len(thresholds))
    class_ids = sorted({int(c) for k in keys for c in np.unique(gt[k]) if c != 0})
    per_class = {
        c: {"tp": np.zeros(len(thresholds)), "fp": np.zeros(len(thresholds)), "fn": np.zeros(len(thresholds))}
        for c in class_ids
    }
    ious: list[np.ndarray] = []
    start_deltas: list[np.ndarray] = []
    end_deltas: list[np.ndarray] = []

    for key in keys:
        g = np.asarray(gt[key]).astype(int)
        p = np.asarray(pred[key]).astype(int)
        n = min(len(g), len(p))
        g, p = g[:n], p[:n]
        correct += int((g == p).sum())
        total += n
        frame_tp += int(((p == g) & (g != 0)).sum())
        frame_fp += int(((p != 0) & (g == 0)).sum())
        frame_fn += int(((p == 0) & (g != 0)).sum())
        edit += edit_score(p, g)
        for s, thr in enumerate(thresholds):
            tp1, fp1, fn1, iou, sd, ed = segment_matches(p, g, thr)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
            if s == 0:
                ious.append(iou)
                start_deltas.append(sd)
                end_deltas.append(ed)
        for c in class_ids:
            g_c = np.where(g == c, c, 0)
            p_c = np.where(p == c, c, 0)
            for s, thr in enumerate(thresholds):
                tp1, fp1, fn1, _, _, _ = segment_matches(p_c, g_c, thr)
                per_class[c]["tp"][s] += tp1
                per_class[c]["fp"][s] += fp1
                per_class[c]["fn"][s] += fn1

    out: dict[str, Any] = {
        "acc": 100.0 * correct / max(total, 1),
        "edit": edit / len(keys),
        "frame_f1": 100.0 * 2 * frame_tp / max(2 * frame_tp + frame_fn + frame_fp, 1),
        # TP/FP/FN at the first threshold only — same convention as `ious`/
        # `start_deltas_s`/`end_deltas_s` below, and what the comparison
        # figure's inset bar chart reads.
        "tp": float(tp[0]),
        "fp": float(fp[0]),
        "fn": float(fn[0]),
    }
    for s, thr in enumerate(thresholds):
        out[metric_key(thr)] = _f1(tp[s], fp[s], fn[s])
    out["classwise"] = {
        int(c): {metric_key(thr): _f1(v["tp"][s], v["fp"][s], v["fn"][s]) for s, thr in enumerate(thresholds)}
        for c, v in per_class.items()
    }
    out["ious"] = np.concatenate(ious) if ious else np.array([])
    out["start_deltas_s"] = (np.concatenate(start_deltas) if start_deltas else np.array([])) / fs
    out["end_deltas_s"] = (np.concatenate(end_deltas) if end_deltas else np.array([])) / fs
    out["n_samples"] = len(keys)
    return out


def scalar_metrics(m: dict[str, Any]) -> dict[str, float]:
    """The YAML/TSV-able part of an :func:`evaluate` result."""
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating, np.integer))}


def save_eval_arrays(path: Path, raw: dict[str, Any], processed: dict[str, Any]) -> None:
    """Save the arrays an :func:`evaluate` result carries that don't fit in YAML.

    *raw* and *processed* are two :func:`evaluate` results (same samples,
    before/after post-processing). Every trained run with a test split writes
    one of these beside its ``test_metrics.yaml``.
    """
    np.savez(
        path,
        raw_ious=raw["ious"],
        raw_start_deltas_s=raw["start_deltas_s"],
        raw_end_deltas_s=raw["end_deltas_s"],
        post_ious=processed["ious"],
        post_start_deltas_s=processed["start_deltas_s"],
        post_end_deltas_s=processed["end_deltas_s"],
    )


def load_eval_arrays(path: Path) -> dict[str, np.ndarray]:
    """The arrays :func:`save_eval_arrays` wrote, keyed the same way."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
