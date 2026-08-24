"""Experiment 2 — retune the MS-TCN smoothing loss for 200 Hz.

The consistency (truncated-MSE) term of ``MS_TCN_Loss`` penalises how fast
the log-probabilities may change between adjacent frames. It is, by
construction, a boundary-blurring regulariser, and both of its numbers were
tuned in the temporal-action-segmentation literature at 15–30 fps: ``alpha``
(how much the term counts, MS-TCN's λ = 0.15, DLC2Action's 0.001) and ``tau``
(how large a log-probability jump it still penalises, MS-TCN's τ = 4, which
upstream writes into the arithmetic as ``clamp(..., max=16)``). At 200 Hz the
same ``alpha`` smooths over roughly 13× more real time, so this is the
cheapest intervention available.

Two sweeps, written into ``{project}/experiment2/``:

**The grid** — ``alpha × tau``, everything else fixed, one training run per
cell per fold. ``alpha = 0`` turns the term off entirely and makes ``tau``
meaningless, so it is run once rather than once per ``tau``.

**The rate ablation** — the same model trained at 200 / 100 / 50 Hz
(``train.subsample`` 1 / 2 / 4). Its runs report metrics in their *own*
frames, which is not comparable across rates, so every rate is re-scored here
on the 200 Hz grid against the same ground truth: the prediction is held
constant across the ``k`` frames it covers, exactly as it would be when
written back into the GUI's labels. F1@90 is an IoU ratio and the boundary
deltas are already in seconds, so both then speak real time.

Folds are leave-one-session-out, as in
:meth:`~ethograph.segment.project.Project.cross_validate`, but the cells are
driven here rather than through it so a cell can be screened on one fold
before the other three are paid for, and so the rate ablation can be
re-scored. Each fold trains on the sessions it does not hold out with no
validation slice (``val_fraction = 0``): the epoch budget is fixed and equal
for every cell, so the comparison is between losses rather than between
early-stopping points.

``results.tsv`` is append-only and keyed by (alpha, tau, subsample, fold), so
the sweep is resumable — rerun after an interruption and it picks up where it
stopped. Nothing is recomputed and nothing is overwritten.

    python scripts/experiment2_smoothing.py

**What it costs.** Measured on the crow project (RTX 3080, 633 samples of
~1840 frames, 96 columns): encoder-only ASFormer at ``batch_size = 1`` runs
~130 s/epoch over 368 training samples, and a leave-one-session-out fold
trains on ~480, so ~165 s/epoch — about 4.5 h for the configured 100 epochs.
That architecture runs one sample at a time whatever ``batch_size`` says (its
sliding attention cross-wires a batch, see ``models/vendored.py``), so a
bigger batch buys nothing here and pads the short samples to the longest.

The grid is 16 cells. Screening one fold is therefore ~72 h, and confirming
four cells over the other three folds another ~55 h. Three ways to fit that,
in the order worth trying:

1. Run ``STAGES = ("rate",)`` first (~8 h): if 50 Hz holds F1@90, the whole
   grid drops to ~18 h and so does everything after it.
2. Cut the epoch budget for every cell — ``COMMON = {"train.epochs": 40}``.
   The published runs were still improving at epoch 15, so this is a real
   compromise; make it once, for the whole sweep, and say so in the write-up.
3. Screen the grid on a cheap architecture (``COMMON = {"model.architecture":
   "mstcn"}``, which does batch) and confirm the winners on ASFormer. The
   optimum need not transfer, so the confirming pass is not optional.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

import ethograph as eto
from ethograph.labels.onset_model import session_id
from ethograph.segment.config import SegmentConfig, SessionSpec
from ethograph.segment.dataset import MaterialisedStore
from ethograph.segment.infer import load_run, predict_probabilities
from ethograph.segment.losses import DEFAULT_TAU, upstream_defaults
from ethograph.segment.materialise import COLUMNS_FILE
from ethograph.segment.metrics import evaluate, metric_key, scalar_metrics
from ethograph.segment.postprocess import postprocess_dense
from ethograph.segment.train import RunResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "data" / "model" / "project.yaml"

#: λ — the weight of the consistency term. ``0.001`` is DLC2Action's default
#: (the incumbent), ``0.15`` MS-TCN's published value; ``0`` turns it off.
LAMBDAS = [0.0, 0.001, 0.015, 0.05, 0.15, 0.3]

#: τ — the truncation threshold. ``4`` is upstream's, written into its
#: ``clamp(..., max=16)``; larger values truncate less, so a real class change
#: is penalised harder.
TAUS = [4.0, 16.0, 48.0]

#: The rate ablation: frame stride over the materialised 200 Hz dataset.
RATES = [1, 2, 4]

#: How many sessions each sweep is run over. Screening on one fold costs a
#: quarter of the grid and answers where the optimum roughly is; the
#: confirming pass is what produces mean ± sd. ``None`` = every session.
GRID_FOLDS: int | None = 1
RATE_FOLDS: int | None = 1

#: Cells carried from the screening pass into the full four-fold pass, best
#: first, plus the incumbent and ``alpha = 0`` (which are the two the write-up
#: has to quote whatever wins). Set to ``0`` to skip the confirming pass.
CONFIRM_TOP: int = 2

#: Overrides every run in this experiment shares.
#:
#: ``num_decoders: 0`` is the brief's model — ASFormer's encoder only. It is
#: pinned here rather than left to ``project.yaml``, whose empty ``params:``
#: means upstream's default of three refinement decoders, so the sweep would
#: otherwise measure a different architecture than the one the result is
#: written up for. Empty this dict to sweep whatever the project config says.
#:
#: The epoch budget is the one knob worth turning down when the sweep has to
#: fit a night rather than a week — but turn it down for *every* cell, never
#: per cell.
COMMON: dict[str, Any] = {"model.params": {"num_decoders": 0}}

#: Which sweeps this invocation runs, in order. Both are resumable and share
#: one ``results.tsv``, so running them in separate sittings costs nothing —
#: and ``("rate",)`` first is the cheaper order when the grid does not fit the
#: machine yet (see the note on cost above).
STAGES: tuple[str, ...] = ("grid", "rate")

RESULTS_FILE = "results.tsv"
PRIMARY = metric_key(0.9)

logger = logging.getLogger("experiment2")


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One point of the sweep: a loss setting at a temporal resolution."""

    alpha: float
    tau: float
    subsample: int = 1

    @property
    def tag(self) -> str:
        return f"a{self.alpha:g}_t{self.tau:g}_k{self.subsample}"

    @property
    def overrides(self) -> dict[str, Any]:
        return {
            "train.loss.alpha": self.alpha,
            "train.loss.tau": self.tau,
            "train.subsample": self.subsample,
        }


def grid_cells(lambdas: Iterable[float] | None = None, taus: Iterable[float] | None = None) -> list[Cell]:
    """The (λ, τ) grid, with λ = 0 collapsed: with the term off, τ does nothing."""
    lambdas = LAMBDAS if lambdas is None else lambdas
    taus = list(TAUS if taus is None else taus)
    cells = []
    for alpha in lambdas:
        for tau in taus[:1] if alpha == 0 else taus:
            cells.append(Cell(float(alpha), float(tau), 1))
    return cells


def rate_cells(cell: Cell, rates: Iterable[int] | None = None) -> list[Cell]:
    """*cell*'s loss setting at each temporal resolution."""
    return [Cell(cell.alpha, cell.tau, int(k)) for k in (RATES if rates is None else rates)]


# ---------------------------------------------------------------------------
# Running one (cell, fold)
# ---------------------------------------------------------------------------


def fold_overrides(held_out: SessionSpec, run_name: str) -> dict[str, Any]:
    """Train on every session but *held_out*, and test on it — one fold."""
    return {
        "train.split.holdout_sessions": [str(held_out.source)],
        "train.split.train_fraction": 1.0,
        "train.split.val_fraction": 0.0,
        "train.split.test_fraction": 0.0,
        "train.run_name": run_name,
    }


def run_cell(config_path: Path, cell: Cell, held_out: SessionSpec, name: str) -> tuple[RunResult, float]:
    """One training run; returns it with its wall-clock seconds."""
    fold = session_id(held_out.source)  # the stem alone collides: every session file is Trial_data3.nc
    overrides = eto.segment.as_overrides(
        {**COMMON, **cell.overrides, **fold_overrides(held_out, f"exp2/{name}/{cell.tag}/fold-{fold}")}
    )
    project = eto.segment.Project(config_path, *overrides)
    started = time.perf_counter()
    result = project.train()
    return result, time.perf_counter() - started


def row_for(cell: Cell, held_out: SessionSpec, result: RunResult, seconds: float, extra: dict[str, Any]) -> dict:
    """One results row: what was run, how long it took, and what it scored."""
    row: dict[str, Any] = {
        "alpha": cell.alpha,
        "tau": cell.tau,
        "subsample": cell.subsample,
        "fold": session_id(held_out.source),
        "run_dir": str(result.run_dir),
        "epochs": result.best_epoch,
        "seconds": round(seconds, 1),
    }
    for stage in ("raw", "postprocessed"):
        for key, value in (result.test_metrics or {}).get(stage, {}).items():
            if key != "classwise":
                row[f"{stage}.{key}"] = value
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Scoring a lower-rate run back on the full-rate grid
# ---------------------------------------------------------------------------


def score_at_reference(run_dir: Path) -> dict[str, dict[str, float]]:
    """Re-score a run's test samples on the materialised dataset's own rate.

    A run trained with ``train.subsample = k`` predicts one label per *k*
    frames; here that prediction is held across those *k* frames and compared
    against the full-rate ground truth, so 200, 100 and 50 Hz runs are judged
    on the same timeline in the same units. At ``k = 1`` this reproduces the
    run's own test metrics.
    """
    run = load_run(run_dir)
    config = run.config  # the run's own, saved with absolute paths
    store = MaterialisedStore.open(config.data_dir)  # the full rate, whatever the run trained at
    step = int(config.train.subsample)
    keys = (run_dir / "splits" / "test.bundle").read_text(encoding="utf-8").split()

    gt: dict[str, np.ndarray] = {}
    pred: dict[str, np.ndarray] = {}
    for key in keys:
        x, y = store.load(key)
        probs, _boundary = predict_probabilities(run, np.ascontiguousarray(x[:, ::step]))
        indices = probs.argmax(axis=1)
        if step > 1:
            indices = np.repeat(indices, step)[: len(y)]
            if len(indices) < len(y):  # the tail the stride did not reach keeps the last decision
                indices = np.concatenate([indices, np.full(len(y) - len(indices), indices[-1])])
        gt[key], pred[key] = y, indices

    fs = store.layout.fs
    thresholds = config.train.f1_thresholds
    raw = evaluate(gt, pred, thresholds, fs)
    processed = evaluate(
        gt,
        {k: postprocess_dense(v, fs, store.classes, config.infer.postprocess) for k, v in pred.items()},
        thresholds,
        fs,
    )
    return {"raw": scalar_metrics(raw), "postprocessed": scalar_metrics(processed)}


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def already_done(results: pd.DataFrame, cell: Cell, fold: str) -> bool:
    if results.empty:
        return False
    match = (
        np.isclose(results["alpha"], cell.alpha)
        & np.isclose(results["tau"], cell.tau)
        & (results["subsample"] == cell.subsample)
        & (results["fold"] == fold)
    )
    return bool(match.any())


def backfill_reference(results: pd.DataFrame, cells: list[Cell], folds: list[SessionSpec], path: Path) -> pd.DataFrame:
    """Add the reference-grid metrics to rows that were trained without them."""
    column = f"ref.postprocessed.{PRIMARY}"
    wanted = {(c.alpha, c.tau, c.subsample) for c in cells}
    keys = {session_id(f.source) for f in folds}
    for i, row in results.iterrows():
        if (row["alpha"], row["tau"], row["subsample"]) not in wanted or row["fold"] not in keys:
            continue
        if column in results and pd.notna(row.get(column)):
            continue
        logger.info("scoring %s on the reference grid", Path(row["run_dir"]).name)
        for stage, metrics in score_at_reference(Path(row["run_dir"])).items():
            for key, value in metrics.items():
                results.loc[i, f"ref.{stage}.{key}"] = value
    results.to_csv(path, sep="	", index=False)
    return results


def sweep(
    config_path: Path,
    cells: list[Cell],
    folds: list[SessionSpec],
    out_dir: Path,
    name: str,
    reference_scoring: bool = False,
) -> pd.DataFrame:
    """Run every (cell, fold) not already in ``results.tsv``, appending as it goes."""
    path = out_dir / RESULTS_FILE
    results = load_results(path)
    todo = [(c, f) for c in cells for f in folds if not already_done(results, c, session_id(f.source))]
    logger.info("%s: %d cells × %d folds, %d runs to do", name, len(cells), len(folds), len(todo))

    if reference_scoring:
        # A cell this sweep needs may already have been trained by another one
        # (k = 1 is both a grid cell and a rate condition). Score that run on
        # the reference grid rather than training it a second time.
        results = backfill_reference(results, cells, folds, path)

    for n, (cell, held_out) in enumerate(todo, start=1):
        logger.info("[%d/%d] %s, holding out %s", n, len(todo), cell.tag, session_id(held_out.source))
        result, seconds = run_cell(config_path, cell, held_out, name)
        extra: dict[str, Any] = {"sweep": name}
        if reference_scoring:
            for stage, metrics in score_at_reference(result.run_dir).items():
                extra.update({f"ref.{stage}.{k}": v for k, v in metrics.items()})
        row = row_for(cell, held_out, result, seconds, extra)
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(path, sep="\t", index=False)
        logger.info(
            "    %s raw %.2f, post-processed %.2f (%.0f s)",
            PRIMARY,
            row.get(f"raw.{PRIMARY}", float("nan")),
            row.get(f"postprocessed.{PRIMARY}", float("nan")),
            seconds,
        )
    return results


def summarise(results: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean ± sd of *column* over folds, one row per cell."""
    if results.empty or column not in results:
        return pd.DataFrame()
    grouped = results.groupby(["alpha", "tau", "subsample"])[column]
    table = grouped.agg(["mean", "std", "count"]).reset_index()
    return table.sort_values("mean", ascending=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def write_grid_figure(results: pd.DataFrame, path: Path) -> Path:
    """The heatmap the experiment is for, plus the folds behind each cell."""
    grid = results[results["subsample"] == 1]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, stage in zip(axes[:2], ("raw", "postprocessed")):
        _heatmap(ax, grid, f"{stage}.{PRIMARY}", f"{stage} {PRIMARY}")
    _folds(axes[2], grid, f"postprocessed.{PRIMARY}")
    fig.suptitle(f"Experiment 2 — the smoothing loss at {_rate(results):.0f} Hz")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _heatmap(ax, grid: pd.DataFrame, column: str, title: str) -> None:
    lambdas = sorted(grid["alpha"].unique())
    taus = sorted(grid["tau"].unique())
    mean = np.full((len(taus), len(lambdas)), np.nan)
    sd = np.full_like(mean, np.nan)
    for i, tau in enumerate(taus):
        for j, alpha in enumerate(lambdas):
            # alpha = 0 is one cell, drawn across the tau row it stands for
            picked = grid[np.isclose(grid["alpha"], alpha) & (np.isclose(grid["tau"], tau) | (alpha == 0))]
            if picked.empty or column not in picked:
                continue
            mean[i, j] = picked[column].mean()
            sd[i, j] = picked[column].std()
    image = ax.imshow(mean, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(lambdas)), [f"{v:g}" for v in lambdas])
    ax.set_yticks(range(len(taus)), [f"{v:g}" for v in taus])
    ax.set_xlabel("alpha (λ, weight of the smoothing term)")
    ax.set_ylabel("tau (τ, truncation)")
    ax.set_title(title)
    for i in range(len(taus)):
        for j in range(len(lambdas)):
            if np.isnan(mean[i, j]):
                continue
            label = f"{mean[i, j]:.1f}" if np.isnan(sd[i, j]) else f"{mean[i, j]:.1f}\n±{sd[i, j]:.1f}"
            ax.text(j, i, label, ha="center", va="center", color="w", fontsize=8)
    plt.colorbar(image, ax=ax)


def _folds(ax, grid: pd.DataFrame, column: str) -> None:
    """Every fold as a dot over its cell's mean — n is small, so show it."""
    cells = sorted({(a, t) for a, t in zip(grid["alpha"], grid["tau"])})
    for x, (alpha, tau) in enumerate(cells):
        picked = grid[np.isclose(grid["alpha"], alpha) & np.isclose(grid["tau"], tau)]
        values = picked[column].to_numpy()
        ax.scatter(np.full(len(values), x), values, s=18, color="tab:blue", zorder=3)
        ax.hlines(values.mean(), x - 0.3, x + 0.3, color="k", zorder=4)
    ax.set_xticks(range(len(cells)), [f"λ{a:g}\nτ{t:g}" for a, t in cells], fontsize=7)
    ax.set_ylabel(column)
    ax.set_title("Per fold")


def write_rate_figure(results: pd.DataFrame, path: Path) -> Path:
    """The rate ablation, scored on the full-rate grid so the units are seconds."""
    rates = results.dropna(subset=[f"ref.postprocessed.{PRIMARY}"]) if f"ref.postprocessed.{PRIMARY}" in results else []
    if len(rates) == 0:
        return path
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fs = _rate(results)
    for ax, stage in zip(axes, ("raw", "postprocessed")):
        column = f"ref.{stage}.{PRIMARY}"
        steps = sorted(rates["subsample"].unique())
        for x, step in enumerate(steps):
            values = rates[rates["subsample"] == step][column].to_numpy()
            ax.bar(x, values.mean(), 0.6, color="tab:grey")
            ax.scatter(np.full(len(values), x), values, s=18, color="tab:blue", zorder=3)
        ax.set_xticks(range(len(steps)), [f"{fs / s:.0f} Hz" for s in steps])
        ax.set_ylabel(f"{PRIMARY}, scored at {fs:.0f} Hz")
        ax.set_title(stage)
    fig.suptitle("Rate ablation — every condition scored on the same timeline")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _rate(results: pd.DataFrame) -> float:
    """The dataset's own rate, read off a run rather than assumed."""
    run_dir = Path(results["run_dir"].iloc[0])
    import yaml

    return float(yaml.safe_load((run_dir / "columns.yaml").read_text(encoding="utf-8"))["fs"])


# ---------------------------------------------------------------------------


def run_grid(config: SegmentConfig, sessions: list[SessionSpec], out_dir: Path) -> pd.DataFrame:
    """Screen the grid on ``GRID_FOLDS`` sessions, then confirm the best cells on all of them."""
    results = sweep(CONFIG, grid_cells(), sessions[:GRID_FOLDS], out_dir, "grid")
    ranking = summarise(results[results["subsample"] == 1], f"postprocessed.{PRIMARY}")
    print(ranking.to_string(index=False))

    if CONFIRM_TOP and len(sessions) > 1:
        # The screened winners plus the two the write-up has to quote either way.
        best = [Cell(r.alpha, r.tau) for r in ranking.head(CONFIRM_TOP).itertuples()]
        for cell in (Cell(0.0, TAUS[0]), Cell(incumbent_alpha(config), 4.0)):
            if cell not in best:
                best.append(cell)
        results = sweep(CONFIG, best, sessions, out_dir, "grid")

    write_grid_figure(results[results["subsample"] == 1], out_dir / "grid_f1@90.pdf")
    return results


def incumbent_alpha(config: SegmentConfig) -> float:
    """The λ this project trains with today — the baseline every cell is compared to."""
    return float(config.train.loss.get("alpha", upstream_defaults()["alpha"]))


def run_rates(config: SegmentConfig, sessions: list[SessionSpec], out_dir: Path) -> pd.DataFrame:
    """The rate ablation, at whichever loss setting the grid chose.

    It asks "does 200 Hz still pay once the loss is right?" rather than
    re-testing the loss — so when the grid has not run yet it falls back to
    the project's own λ rather than inventing one.
    """
    done = load_results(out_dir / RESULTS_FILE)
    grid = done[done["subsample"] == 1] if not done.empty else done
    ranking = summarise(grid, f"postprocessed.{PRIMARY}")
    reference = (
        Cell(float(ranking.iloc[0].alpha), float(ranking.iloc[0].tau))
        if not ranking.empty
        else Cell(incumbent_alpha(config), DEFAULT_TAU)
    )
    logger.info("Rate ablation at the loss setting %s", reference.tag)

    cells = rate_cells(reference)
    results = sweep(CONFIG, cells, sessions[:RATE_FOLDS], out_dir, "rate", reference_scoring=True)
    rates = results[
        np.isclose(results["alpha"], reference.alpha)
        & np.isclose(results["tau"], reference.tau)
        & results["subsample"].isin([c.subsample for c in cells])
    ]
    write_rate_figure(rates, out_dir / "rate_ablation.pdf")
    print(summarise(rates, f"ref.postprocessed.{PRIMARY}").to_string(index=False))
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    project = eto.segment.Project(CONFIG)
    out_dir = project.root / "experiment2"
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = list(project.config.sessions)

    # The dataset is what every cell shares, so it is materialised once,
    # before anything is trained, and never varied by the sweep — and not
    # again on a resumed run, which would re-open every session for nothing.
    if not (project.config.data_dir / COLUMNS_FILE).is_file():
        project.materialise()

    for stage in STAGES:
        if stage == "grid":
            run_grid(project.config, sessions, out_dir)
        elif stage == "rate":
            run_rates(project.config, sessions, out_dir)
        else:
            raise ValueError(f"STAGES names {stage!r}; this script has 'grid' and 'rate'.")
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
