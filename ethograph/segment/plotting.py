"""The figures written beside a run's test metrics.

``eval.pdf`` is the summary — overall and class-wise F1 plus the boundary-delta
histograms. ``boundary.pdf`` is written only by a run whose architecture has a
boundary head, and it is the qualitative counterpart: the predicted boundary
probability against the true transitions, sample by sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

from ethograph.segment.metrics import EVAL_ARRAYS_FILE, TEST_METRICS_FILE, load_eval_arrays, metric_key
from ethograph.segment.samples import ClassTable

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_THRESHOLD_COLOURS = ["#9467bd", "#2ecc71", "#e377c2", "#ff7f0e", "#17becf"]
_STAGE_COLOURS = {"raw": "#1f77b4", "processed": "#d62728"}
_DOT_COLOUR = "black"
_DOT_SIZE = 6


def write_eval_pdf(
    path: Path,
    raw: dict[str, Any],
    processed: dict[str, Any],
    classes: ClassTable,
    thresholds: list[float],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _overall(axes[0, 0], raw, processed, thresholds)
    _classwise(axes[0, 1], raw, processed, classes, thresholds[0])
    _deltas(axes[1, 0], raw["start_deltas_s"], processed["start_deltas_s"], "Onset |Δ| (s)")
    _deltas(axes[1, 1], raw["end_deltas_s"], processed["end_deltas_s"], "Offset |Δ| (s)")
    fig.suptitle(f"Test set — {raw['n_samples']} samples")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _overall(ax, raw, processed, thresholds) -> None:
    keys = ["acc", "edit", "frame_f1", *(metric_key(t) for t in thresholds)]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [raw[k] for k in keys], 0.4, label="raw")
    ax.bar(x + 0.2, [processed[k] for k in keys], 0.4, label="post-processed")
    ax.set_xticks(x, keys, rotation=30)
    ax.set_ylim(0, 100)
    ax.set_title("Overall")
    ax.legend()


def _classwise(ax, raw, processed, classes: ClassTable, threshold: float) -> None:
    key = metric_key(threshold)
    ids = sorted(set(raw["classwise"]) | set(processed["classwise"]))
    names = [classes.names[i] if i < classes.n_classes else str(i) for i in ids]
    x = np.arange(len(ids))
    ax.bar(x - 0.2, [raw["classwise"].get(i, {}).get(key, 0.0) for i in ids], 0.4, label="raw")
    ax.bar(x + 0.2, [processed["classwise"].get(i, {}).get(key, 0.0) for i in ids], 0.4, label="post-processed")
    ax.set_xticks(x, names, rotation=30)
    ax.set_ylim(0, 100)
    ax.set_title(f"Class-wise {key}")
    ax.legend()


def _deltas(ax, raw: np.ndarray, processed: np.ndarray, label: str) -> None:
    data = [d for d in (raw, processed) if len(d)]
    if not data:
        ax.set_title(f"{label}: no matched segments")
        return
    upper = float(np.percentile(np.concatenate(data), 95)) or 1.0
    bins = np.linspace(0, upper, 30)
    ax.hist(raw, bins=bins, alpha=0.6, label=f"raw (median {np.median(raw):.3f})" if len(raw) else "raw")
    ax.hist(
        processed, bins=bins, alpha=0.6, label=f"post (median {np.median(processed):.3f})" if len(processed) else "post"
    )
    ax.set_xlabel(label)
    ax.set_ylabel("matched segments")
    ax.legend()


@dataclass
class BoundaryPanel:
    """One sample's boundary diagnostic — what the head predicted, against the truth.

    *reference* is the kinematic trace the boundary is supposed to coincide
    with (the beak-tip speed, typically) and *changepoints* the times a
    detector found in it. Both are optional here and neither is available
    inside a training run, which sees only the normalised feature matrix —
    they are what an analysis script passes in once it has the session open.
    """

    key: str
    time: np.ndarray
    probability: np.ndarray
    gt: np.ndarray
    pred: np.ndarray
    threshold: float = 0.5
    reference: np.ndarray | None = None
    reference_label: str = "speed"
    changepoints: np.ndarray | None = None


def write_boundary_pdf(path: Path, panels: list[BoundaryPanel]) -> Path:
    """One row per sample: the boundary curve, its peaks, and the true transitions.

    This is the figure that says whether the head learnt *where* rather than
    *what*: a peak that sits on a true transition is the claim, and a peak
    that sits on a speed minimum with no transition is the failure mode the
    hybrid refinement exists to survive.
    """
    if not panels:
        raise ValueError("write_boundary_pdf needs at least one panel.")
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.6 * len(panels)), squeeze=False)
    for ax, panel in zip(axes[:, 0], panels):
        _boundary_panel(ax, panel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _boundary_panel(ax, panel: BoundaryPanel) -> None:
    from ethograph.segment.boundary import boundary_peaks

    time = panel.time
    ax.plot(time, panel.probability, lw=1.0, color="tab:blue", label="predicted boundary p")
    ax.axhline(panel.threshold, lw=0.6, color="tab:blue", ls=":")
    for i, index in enumerate(np.flatnonzero(np.diff(panel.gt) != 0) + 1):
        ax.axvline(time[index], color="k", lw=1.0, alpha=0.7, label="true boundary" if i == 0 else None)
    peaks = boundary_peaks(panel.probability, panel.threshold)
    for i, index in enumerate(peaks):
        ax.axvline(time[index], color="tab:red", lw=0.9, ls="--", label="predicted peak" if i == 0 else None)
    if panel.changepoints is not None:
        for i, t in enumerate(np.asarray(panel.changepoints)):
            ax.axvline(t, color="tab:green", lw=0.6, alpha=0.5, label="detected changepoint" if i == 0 else None)
    if panel.reference is not None:
        twin = ax.twinx()
        twin.plot(time, panel.reference, lw=0.8, color="tab:grey", alpha=0.8)
        twin.set_ylabel(panel.reference_label)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("boundary p")
    ax.set_title(f"{panel.key} — {len(peaks)} peaks, {int((np.diff(panel.gt) != 0).sum())} true boundaries")
    ax.legend(loc="upper right", fontsize="x-small", ncol=2)


# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------


@dataclass
class RunEval:
    """One run's held-out test evaluation — everything :func:`write_comparison_pdf` reads.

    A fold, a search trial, a benchmarked architecture: whatever is being
    compared, each contributes one of these. Built by :func:`load_run_eval`
    from a run directory's ``test_metrics.yaml`` (the scalars and class-wise
    F1) plus ``test_eval.npz`` (the matched-segment IoUs and onset/offset
    deltas — too bulky for YAML, see :func:`ethograph.segment.metrics.save_eval_arrays`).
    """

    name: str
    thresholds: list[float]
    raw: dict[str, Any]
    processed: dict[str, Any]
    raw_ious: np.ndarray
    processed_ious: np.ndarray
    raw_deltas_s: np.ndarray
    processed_deltas_s: np.ndarray
    run_dir: Path | None = None


def load_run_eval(run_dir: Path, name: str | None = None) -> RunEval:
    """A run's :class:`RunEval`, from its ``test_metrics.yaml`` + ``test_eval.npz``."""
    data = yaml.safe_load((run_dir / TEST_METRICS_FILE).read_text(encoding="utf-8"))
    arrays = load_eval_arrays(run_dir / EVAL_ARRAYS_FILE)
    return RunEval(
        name=name or run_dir.name,
        thresholds=list(data["thresholds"]),
        raw=data["raw"],
        processed=data["postprocessed"],
        raw_ious=arrays["raw_ious"],
        processed_ious=arrays["post_ious"],
        raw_deltas_s=np.concatenate([arrays["raw_start_deltas_s"], arrays["raw_end_deltas_s"]]),
        processed_deltas_s=np.concatenate([arrays["post_start_deltas_s"], arrays["post_end_deltas_s"]]),
        run_dir=run_dir,
    )


def write_comparison_pdf(path: Path, evals: list[RunEval], classes: ClassTable, title: str = "") -> Path:
    """The cross-run comparison figure — folds, search trials, or benchmarked runs.

    Every bar is a mean over *evals*; every bar also carries one small black
    dot per run, sitting at exactly the value that run landed on — zoom in to
    read them individually, the bars are what matters at a glance.
    """
    if len(evals) < 2:
        raise ValueError(f"write_comparison_pdf needs at least two runs to compare, got {len(evals)}.")
    fig = _eval_figure()
    _eval_panels(fig, evals, classes)
    fig.suptitle(title or f"{len(evals)} runs compared", fontsize=15, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)
    return path


def _eval_figure():
    """A constrained-layout figure sized for :func:`_eval_panels`' six panels."""
    return plt.figure(figsize=(18, 17), layout="constrained")


def _eval_panels(fig, evals: list[RunEval], classes: ClassTable) -> None:
    """The six evaluation panels — over one run, or over a whole set of them."""
    thresholds = evals[0].thresholds
    mosaic = [
        ["A", "A", "B", "B", "B"],
        ["A", "A", "B", "B", "B"],
        ["C", "C", "C", "D", "D"],
        ["C", "C", "C", "D", "D"],
        ["E", "E", "E", "E", "E"],
        ["F", "F", "F", "F", "F"],
    ]
    axes = fig.subplot_mosaic(mosaic)
    _iou_illustration(axes["A"], thresholds)
    _overall_comparison(axes["B"], evals, thresholds)
    _iou_distribution(axes["C"], evals, thresholds)
    _deltas_comparison(axes["D"], evals)
    _classwise_comparison(axes["E"], evals, classes, thresholds, "raw", "Class-wise F1 (raw)")
    _classwise_comparison(axes["F"], evals, classes, thresholds, "processed", "Class-wise F1 (post-processed)")


def _iou_illustration(ax, thresholds: list[float]) -> None:
    """What an IoU threshold means: how far a prediction can drift and still count."""
    gt_len = pred_len = 30.0
    pad = 20.0
    total_len = int(pad * 2 + gt_len)
    gt_start = pad
    heights = np.linspace(0.4, 2.5, max(len(thresholds), 1))
    gt_height = heights[-1] + 0.7

    ax.set_xlim(-1, total_len + 1)
    ax.set_ylim(0, gt_height + 0.7)
    ax.set_yticks([])
    ax.barh(gt_height, gt_len, left=gt_start, height=0.6, color="gray")
    ax.hlines(gt_height, 0, total_len, linewidth=6, color="lightgray")

    for i, thr in enumerate(thresholds):
        idx = (gt_len + pred_len) * thr / (1.0 + thr)
        pred_start = gt_start + (gt_len - idx)
        colour = _THRESHOLD_COLOURS[i % len(_THRESHOLD_COLOURS)]
        ax.barh(heights[i], pred_len, left=pred_start, height=0.6, color=colour)
        ax.text(total_len / 2, heights[i] + 0.15, f"IoU = {thr}", ha="center", va="center", fontweight="bold")
        ax.hlines(heights[i], 0, total_len, linewidth=6, color="lightgray")

    ax.set_xlabel("time (samples)")
    ax.set_title("What IoU means")
    legend = [Patch(facecolor="gray", label="ground truth")]
    legend += [
        Patch(facecolor=_THRESHOLD_COLOURS[i % len(_THRESHOLD_COLOURS)], label=f"f1@{int(t * 100)}")
        for i, t in enumerate(thresholds)
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True, fancybox=True)


def _overall_comparison(ax, evals: list[RunEval], thresholds: list[float]) -> None:
    """Mean ± per-run dots for accuracy, frame F1 and every segmental F1@k."""
    keys = ["acc", "frame_f1", *(metric_key(t) for t in thresholds)]
    labels = ["acc", "frame_f1", *(f"f1@{int(t * 100)}" for t in thresholds)]
    x = np.arange(len(keys))
    width = 0.35
    for offset, stage in ((-width / 2, "raw"), (width / 2, "processed")):
        values = [[getattr(e, stage)[k] for e in evals] for k in keys]
        ax.bar(x + offset, [np.mean(v) for v in values], width, label=stage, alpha=0.85, color=_STAGE_COLOURS[stage])
        if len(evals) > 1:
            for i, v in enumerate(values):
                ax.scatter([x[i] + offset] * len(v), v, color=_DOT_COLOUR, s=_DOT_SIZE, alpha=0.6, zorder=3)
    ax.set_xticks(x, labels, rotation=30)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Overall" if len(evals) == 1 else f"Overall ({len(evals)} runs)")
    ax.legend()


def _classwise_comparison(
    ax, evals: list[RunEval], classes: ClassTable, thresholds: list[float], stage: str, title: str
) -> None:
    """Mean ± per-run dots for each class's F1@k, one group of bars per threshold."""
    ids = sorted({i for e in evals for i in getattr(e, stage)["classwise"]})
    if not ids:
        ax.set_title(f"{title}: no classes")
        return
    names = [classes.names[i] if i < classes.n_classes else str(i) for i in ids]
    x = np.arange(len(ids))
    n = len(thresholds)
    width = 0.8 / n
    for t_idx, thr in enumerate(thresholds):
        key = metric_key(thr)
        colour = _THRESHOLD_COLOURS[t_idx % len(_THRESHOLD_COLOURS)]
        offset = (t_idx - (n - 1) / 2) * width
        points = [[getattr(e, stage)["classwise"].get(cid, {}).get(key, 0.0) for e in evals] for cid in ids]
        ax.bar(x + offset, [np.mean(p) for p in points], width, label=f"f1@{int(thr * 100)}", alpha=0.85, color=colour)
        if len(evals) > 1:
            for i, p in enumerate(points):
                ax.scatter([x[i] + offset] * len(p), p, color=_DOT_COLOUR, s=_DOT_SIZE, alpha=0.6, zorder=3)
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend()


def _iou_distribution(ax, evals: list[RunEval], thresholds: list[float]) -> None:
    """Matched-segment IoU histogram, raw vs post-processed, plus a TP/FP/FN inset."""
    raw = np.concatenate([e.raw_ious for e in evals])
    processed = np.concatenate([e.processed_ious for e in evals])
    ax.hist(raw, bins=50, alpha=0.4, label="raw", color=_STAGE_COLOURS["raw"])
    ax.hist(processed, bins=50, alpha=0.4, label="post-processed", color=_STAGE_COLOURS["processed"])
    for i, thr in enumerate(thresholds):
        ax.axvline(thr, color=_THRESHOLD_COLOURS[i % len(_THRESHOLD_COLOURS)], linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("IoU")
    ax.set_ylabel("matched-segment count (log)")
    ax.set_title("IoU distribution")
    ax.legend(loc="upper right")

    axins = ax.inset_axes([0.34, 0.58, 0.30, 0.36])
    metrics = ["tp", "fp", "fn"]
    x = np.arange(len(metrics))
    width = 0.35
    axins.bar(
        x - width / 2,
        [sum(e.raw[m] for e in evals) for m in metrics],
        width,
        label="raw",
        color=_STAGE_COLOURS["raw"],
    )
    axins.bar(
        x + width / 2,
        [sum(e.processed[m] for e in evals) for m in metrics],
        width,
        label="post",
        color=_STAGE_COLOURS["processed"],
    )
    axins.set_xticks(x, [m.upper() for m in metrics], fontsize=8)
    axins.tick_params(axis="y", labelsize=8)
    axins.set_ylabel("count", fontsize=8)
    axins.legend(fontsize=7, loc="upper right")


def _deltas_comparison(ax, evals: list[RunEval]) -> None:
    """Combined onset/offset |Δ| histogram, raw vs post-processed."""
    raw = np.concatenate([e.raw_deltas_s for e in evals])
    processed = np.concatenate([e.processed_deltas_s for e in evals])
    data = [d for d in (raw, processed) if len(d)]
    if not data:
        ax.set_title("Onset/offset |Δ|: no matched segments")
        return
    upper = float(np.percentile(np.concatenate(data), 95)) or 1.0
    bins = np.linspace(0, upper, 40)
    ax.hist(
        raw,
        bins=bins,
        alpha=0.5,
        label=f"raw (median {np.median(raw):.3f}s)" if len(raw) else "raw",
        color=_STAGE_COLOURS["raw"],
    )
    ax.hist(
        processed,
        bins=bins,
        alpha=0.5,
        label=f"post (median {np.median(processed):.3f}s)" if len(processed) else "post",
        color=_STAGE_COLOURS["processed"],
    )
    ax.set_xlabel("onset/offset |Δ| (s)")
    ax.set_ylabel("matched segments")
    ax.set_title("Boundary deltas")
    ax.legend()


def write_model_report_pdf(
    path: Path,
    evals: list[RunEval],
    classes: ClassTable,
    title: str = "",
    stamp: str | None = None,
) -> Path:
    """One PDF to review a whole comparison in: an overview page, then a page per run.

    Page 1 ranks the runs against each other — segmental F1 at every IoU
    threshold, plus the scalars that curve leaves out. Every page after it is
    one run's own evaluation, drawn by the same :func:`_eval_panels` the
    cross-run figure uses, titled with the run it belongs to: the IoU
    distribution, the boundary deltas and the class-wise F1 of *that model*,
    rather than one bar among five. Every page carries *stamp* (default: now),
    so a printed page says which comparison it came from.
    """
    if not evals:
        raise ValueError("write_model_report_pdf needs at least one run.")
    stamp = stamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    title = title or f"{len(evals)} runs compared"
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(18, 7), layout="constrained")
        axes = fig.subplot_mosaic([["lines", "lines", "scalars"]])
        _threshold_lines(axes["lines"], evals)
        _scalar_comparison(axes["scalars"], evals)
        _save_page(pdf, fig, title, stamp)

        for e in evals:
            fig = _eval_figure()
            _eval_panels(fig, [e], classes)
            _save_page(pdf, fig, e.name, stamp, note=str(e.run_dir) if e.run_dir else "")
        pdf.infodict()["Title"] = f"{title} — {stamp}"
    return path


def _save_page(pdf: PdfPages, fig, title: str, stamp: str, note: str = "") -> None:
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.995, 0.004, stamp, ha="right", va="bottom", fontsize=8, color="gray")
    if note:
        fig.text(0.005, 0.004, note, ha="left", va="bottom", fontsize=8, color="gray")
    pdf.savefig(fig)
    plt.close(fig)


def _run_colour(i: int) -> tuple[float, float, float]:
    """One colour per run, the same in every panel that draws runs side by side."""
    colours = plt.get_cmap("tab10").colors
    return colours[i % len(colours)]


def _threshold_lines(ax, evals: list[RunEval], stage: str = "processed") -> None:
    """Every run's segmental F1 across every IoU threshold — one line each.

    The comparison figure's ``Overall`` panel averages runs into one bar per
    metric; this keeps them apart, which is how a model that only wins at the
    loose threshold gives itself away. *stage* solid, the other stage dashed
    in the same colour.
    """
    other = "raw" if stage == "processed" else "processed"
    keys = [metric_key(t) for t in evals[0].thresholds]
    x = np.arange(len(keys))
    for i, e in enumerate(evals):
        colour = _run_colour(i)
        values = [getattr(e, stage)[k] for k in keys]
        # The numbers ride in the legend, not on the points: runs that tie at the
        # loose threshold sit on top of one another and annotations would overlap.
        label = f"{e.name} — " + " / ".join(f"{v:.1f}" for v in values)
        ax.plot(x, values, marker="o", color=colour, label=label)
        ax.plot(x, [getattr(e, other)[k] for k in keys], marker=".", linestyle="--", color=colour, alpha=0.4)
    ax.set_xticks(x, keys)
    ax.set_xlabel("segmental IoU threshold")
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Segmental F1 across thresholds")
    ax.legend(title=f"{' / '.join(keys)}   (solid = {stage}, dashed = {other})", loc="lower left", fontsize=9)


def _scalar_comparison(ax, evals: list[RunEval], stage: str = "processed") -> None:
    """What the threshold curve leaves out: frame accuracy, edit score, frame F1."""
    keys = ["acc", "edit", "frame_f1"]
    x = np.arange(len(keys))
    width = 0.8 / len(evals)
    for i, e in enumerate(evals):
        offset = (i - (len(evals) - 1) / 2) * width
        ax.bar(x + offset, [getattr(e, stage)[k] for k in keys], width, color=_run_colour(i), label=e.name)
    ax.set_xticks(x, keys)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Frame-level scores ({stage})")
    ax.legend(fontsize=9)
