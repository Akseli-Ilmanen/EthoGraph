"""Experiment 3 — a boundary regression branch (ASRF-style).

Boundary-weighted cross-entropy did not help, and there is a principled reason:
reweighting a frame-wise objective adds no *localisation* gradient. The question
asked of the model is still "is frame t class c?", never "where is the
transition?". A separate class-agnostic boundary head changes the target, and in
the ASRF paper the gain grows as the IoU threshold gets stricter — which is
exactly the F1@90 regime.

Conceptually it is the *learned* version of this project's changepoint
correction: instead of a hand rule snapping to the nearest changepoint, the
network learns which of the overspecified changepoints are real syllable
boundaries.

Two sweeps, written into ``{project}/experiment3/``:

**The head** — the boundary loss weight ``w_b`` crossed with the target's
dilation, spelled in **seconds** because the literature's "+/-4 frames" was
tuned at 15-30 fps and means something entirely different at 200 Hz. The
architecture is ``asrf`` wrapping the project's own encoder, so the class
branch is bit-identical to the encoder-only baseline and only the head is new.
``w_b = 0`` is that baseline, built with the head and not training it.

**The refinement modes** — for free, on the models the first sweep already
trained. The four ways a dense prediction becomes intervals are all
post-processing, so each trained run is re-scored under every one of them
rather than retrained:

* ``raw`` — no post-processing at all;
* ``existing`` — the current pipeline: purge, stitch, snap to *detected*
  changepoints;
* ``predicted`` — cut at the peaks of the model's own boundary probability;
* ``hybrid`` — predicted peaks restricted to the detected changepoint
  candidates and snapped onto them.

Hybrid is the one to expect to win: it keeps the physical prior that a boundary
coincides with a speed minimum, and learns only the selection.

    python scripts/experiment3_boundary.py

``results.tsv`` is append-only and keyed by (cell, fold), so the sweep is
resumable. Costs scale like Experiment 2's: the head adds one 1x1 convolution
over the trunk, which is negligible, and ``brb_stages > 1`` adds a full
single-stage TCN per extra stage, which is not.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ethograph as eto  # noqa: E402
from _sweep import Cell, dataset_rate, fold_dots, load_results, summarise, sweep  # noqa: E402
from ethograph.segment.boundary import boundary_probabilities  # noqa: E402
from ethograph.segment.config import PostprocessConfig, SessionSpec  # noqa: E402
from ethograph.segment.dataset import MaterialisedStore  # noqa: E402
from ethograph.segment.infer import load_run  # noqa: E402
from ethograph.segment.metrics import evaluate, metric_key, scalar_metrics  # noqa: E402
from ethograph.segment.postprocess import postprocess_dense  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "data" / "model" / "project.yaml"

#: The architecture: the project's own encoder, plus the boundary branch. Keep
#: ``backbone_params`` identical to what the baseline trains with, or the class
#: branch is no longer the thing being compared.
ARCHITECTURE: dict[str, Any] = {
    "model.architecture": "asrf",
    "model.params": {
        "backbone": "asformer",
        "backbone_params": {"num_decoders": 0},
        "brb_stages": 1,
    },
}

#: ``w_b`` — the boundary term's weight. ``0`` is the encoder-only baseline
#: with the head built but untrained, which is the row every other cell is read
#: against.
WEIGHTS = [0.0, 0.1, 0.5, 1.0]

#: The target's dilation, in **seconds**. At 200 Hz these are +/-2, +/-5 and
#: +/-10 frames — the brief's three settings, written the way the config takes
#: them so they still mean the same thing at any other rate.
TOLERANCES_S = [0.01, 0.025, 0.05]

#: How many sessions each sweep is run over; ``None`` = every session. Screening
#: on one fold answers roughly where the optimum is at a quarter of the cost.
SCREEN_FOLDS: int | None = 1

#: Cells carried from the screening pass into the full four-fold pass, best
#: first. ``w_b = 0`` is always carried: it is the baseline the write-up quotes.
CONFIRM_TOP: int = 2

#: Overrides every run shares. Turn the epoch budget down here, for every cell
#: at once, never per cell.
COMMON: dict[str, Any] = {}

#: Substring identifying the *binary* changepoint mask columns of the
#: materialised layout — the candidate set the hybrid mode selects from. These
#: are the columns ``features.changepoint_features`` generates with the
#: ``binary`` transform.
CHANGEPOINT_COLUMN = "_cp_binary"

#: How far a predicted peak may be moved onto a candidate, in seconds.
SNAP_S = 0.05

#: Boundary probability below which a local maximum is not a peak.
THRESHOLD = 0.5

PRIMARY = metric_key(0.9)

MODES = ("raw", "existing", "predicted", "hybrid")

logger = logging.getLogger("experiment3")


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


def head_cells() -> list[Cell]:
    """``w_b`` x dilation, with ``w_b = 0`` collapsed: untrained, the dilation does nothing."""
    cells = []
    for weight in WEIGHTS:
        for tolerance in TOLERANCES_S[:1] if weight == 0 else TOLERANCES_S:
            cells.append(
                Cell(
                    tag=f"wb{weight:g}_tol{tolerance:g}",
                    overrides={
                        **ARCHITECTURE,
                        "train.boundary.weight": weight,
                        "train.boundary.tolerance_s": tolerance,
                    },
                    describe={"w_b": weight, "tolerance_s": tolerance},
                )
            )
    return cells


# ---------------------------------------------------------------------------
# Re-scoring one trained run under every refinement mode
# ---------------------------------------------------------------------------


def changepoint_frames(store: MaterialisedStore, x: np.ndarray) -> np.ndarray:
    """Frame indices where any binary changepoint column of this sample fires.

    This is the *existing* candidate set — the same overspecified detections
    the hand rule snaps to — read straight off the materialised features, so
    the hybrid mode selects from exactly what the current pipeline uses.
    """
    columns = [i for i, name in enumerate(store.layout.names) if CHANGEPOINT_COLUMN in name]
    if not columns:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.abs(x[columns]).max(axis=0) > 0)


def postprocess_for(mode: str, base: PostprocessConfig) -> PostprocessConfig | None:
    """The post-processing settings each mode stands for; ``None`` means "do nothing"."""
    if mode == "raw":
        return None
    import dataclasses

    changes: dict[str, Any] = {
        "boundary_threshold": THRESHOLD,
        "boundary_snap_s": SNAP_S,
        "boundary_refinement": {"existing": "none", "predicted": "predicted", "hybrid": "hybrid"}[mode],
    }
    if mode == "hybrid":
        changes["changepoint_correction"] = True
    return dataclasses.replace(base, **changes)


def score_modes(run_dir: Path) -> dict[str, float]:
    """Re-score one trained run's test samples under every refinement mode.

    The model is run once; only the post-processing differs, which is what
    makes the four modes a fair comparison rather than four training runs that
    also happen to differ by their random seed.
    """
    import torch

    run = load_run(run_dir)
    store = MaterialisedStore.open(run.config.data_dir, run.config.train.subsample)
    keys = (run_dir / "splits" / "test.bundle").read_text(encoding="utf-8").split()
    fs = store.layout.fs
    thresholds = run.config.train.f1_thresholds

    gt: dict[str, np.ndarray] = {}
    dense: dict[str, np.ndarray] = {}
    boundary: dict[str, np.ndarray] = {}
    candidates: dict[str, np.ndarray] = {}
    for key in keys:
        x, y = store.load(key)
        xn = run.stats.apply(x if run.keep is None else x[run.keep])
        tensor = torch.from_numpy(np.ascontiguousarray(xn)).unsqueeze(0).to(run.device)
        mask = torch.ones(1, 1, tensor.shape[-1], device=run.device)
        with torch.no_grad():
            from ethograph.segment.models import as_output

            output = as_output(run.model(tensor, mask))
            dense[key] = output.logits[-1, 0].argmax(dim=0).cpu().numpy()
            if output.boundary is not None:
                boundary[key] = boundary_probabilities(output.boundary)[0].cpu().numpy()
        gt[key] = y
        candidates[key] = changepoint_frames(store, x)

    out: dict[str, float] = {}
    for mode in MODES:
        cfg = postprocess_for(mode, run.config.infer.postprocess)
        if mode in ("predicted", "hybrid") and not boundary:
            continue  # this run has no head; the mode does not exist for it
        predicted = {}
        for key, indices in dense.items():
            if cfg is None:
                predicted[key] = indices
                continue
            time = np.arange(len(indices)) / fs
            predicted[key] = postprocess_dense(
                indices,
                fs,
                store.classes,
                cfg,
                time=time,
                cp_times=candidates[key] / fs,
                boundary=boundary.get(key),
            )
        metrics = scalar_metrics(evaluate(gt, predicted, thresholds, fs))
        out.update({f"mode.{mode}.{k}": v for k, v in metrics.items()})
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def write_head_figure(results: pd.DataFrame, path: Path) -> Path:
    """The heatmap over (w_b, dilation), and the folds behind each cell."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, mode in zip(axes[:2], ("existing", "hybrid")):
        _heatmap(ax, results, f"mode.{mode}.{PRIMARY}", f"{mode} {PRIMARY}")
    fold_dots(axes[2], results, f"mode.hybrid.{PRIMARY}")
    axes[2].set_title("Per fold (hybrid)")
    fig.suptitle(f"Experiment 3 — the boundary head at {dataset_rate(results):.0f} Hz")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _heatmap(ax, results: pd.DataFrame, column: str, title: str) -> None:
    weights = sorted(results["w_b"].unique())
    tolerances = sorted(results["tolerance_s"].unique())
    mean = np.full((len(tolerances), len(weights)), np.nan)
    sd = np.full_like(mean, np.nan)
    for i, tolerance in enumerate(tolerances):
        for j, weight in enumerate(weights):
            picked = results[
                np.isclose(results["w_b"], weight)
                & (np.isclose(results["tolerance_s"], tolerance) | (weight == 0))
            ]
            if picked.empty or column not in picked:
                continue
            mean[i, j] = picked[column].mean()
            sd[i, j] = picked[column].std()
    image = ax.imshow(mean, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(weights)), [f"{v:g}" for v in weights])
    ax.set_yticks(range(len(tolerances)), [f"{v * 1000:g} ms" for v in tolerances])
    ax.set_xlabel("w_b (weight of the boundary term)")
    ax.set_ylabel("target dilation")
    ax.set_title(title)
    for i in range(len(tolerances)):
        for j in range(len(weights)):
            if np.isnan(mean[i, j]):
                continue
            label = f"{mean[i, j]:.1f}" if np.isnan(sd[i, j]) else f"{mean[i, j]:.1f}\n±{sd[i, j]:.1f}"
            ax.text(j, i, label, ha="center", va="center", color="w", fontsize=8)
    plt.colorbar(image, ax=ax)


def write_mode_figure(results: pd.DataFrame, path: Path) -> Path:
    """The four refinement modes at three IoU thresholds, on the same trained models."""
    trained = results[results["w_b"] > 0] if "w_b" in results else results
    if trained.empty:
        return path
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, threshold in zip(axes, (0.5, 0.75, 0.9)):
        key = metric_key(threshold)
        columns = [f"mode.{m}.{key}" for m in MODES if f"mode.{m}.{key}" in trained]
        for x, column in enumerate(columns):
            values = trained[column].dropna().to_numpy()
            ax.bar(x, values.mean(), 0.6, color="tab:grey")
            ax.scatter(np.full(len(values), x), values, s=16, color="tab:blue", zorder=3)
        ax.set_xticks(range(len(columns)), [c.split(".")[1] for c in columns], rotation=20)
        ax.set_ylabel(key)
        ax.set_title(key)
    fig.suptitle("Refinement modes — one trained model, four ways of reading it")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    project = eto.segment.Project(CONFIG)
    out_dir = project.root / "experiment3"
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions: list[SessionSpec] = list(project.config.sessions)

    # Every cell shares the dataset, so it is materialised once and never varied.
    project.materialise()

    def after(cell: Cell, held_out: SessionSpec, result) -> dict[str, float]:
        return score_modes(result.run_dir)

    cells = head_cells()
    results = sweep(CONFIG, cells, sessions[:SCREEN_FOLDS], out_dir, "exp3", COMMON, after)
    ranking = summarise(results, f"mode.hybrid.{PRIMARY}")
    print(ranking.to_string(index=False))

    if CONFIRM_TOP and len(sessions) > 1:
        best = [c for c in cells if c.tag in set(ranking.head(CONFIRM_TOP)["cell"])]
        baseline = [c for c in cells if c.describe["w_b"] == 0]
        results = sweep(CONFIG, best + [b for b in baseline if b not in best], sessions, out_dir, "exp3", COMMON, after)

    write_head_figure(results, out_dir / "head_f1@90.pdf")
    write_mode_figure(results, out_dir / "refinement_modes.pdf")
    print(summarise(load_results(out_dir / "results.tsv"), f"mode.hybrid.{PRIMARY}").to_string(index=False))
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
