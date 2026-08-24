"""The bookkeeping every experiment sweep needs, and nothing about any one of them.

A *cell* is one point of a sweep: a tag and a dict of dotted config overrides.
A *fold* is one held-out session. :func:`sweep` runs every (cell, fold) that is
not already in ``results.tsv`` and appends as it goes, so an interrupted sweep
resumes where it stopped and nothing is ever recomputed or overwritten.

Folds are leave-one-session-out, as in
:meth:`~ethograph.segment.project.Project.cross_validate`, but driven here so a
cell can be screened on one fold before the rest are paid for. Each fold trains
with no validation slice: the epoch budget is fixed and equal for every cell, so
the comparison is between the settings rather than between early-stopping
points.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ethograph as eto
from ethograph.labels.onset_model import session_id
from ethograph.segment.config import SessionSpec
from ethograph.segment.train import RunResult

RESULTS_FILE = "results.tsv"

logger = logging.getLogger("sweep")


@dataclass(frozen=True)
class Cell:
    """One point of a sweep: what to call it, and what to override."""

    tag: str
    overrides: dict[str, Any] = field(default_factory=dict)
    #: Columns describing this cell, written into every one of its result rows.
    describe: dict[str, Any] = field(default_factory=dict)


def fold_overrides(held_out: SessionSpec, run_name: str) -> dict[str, Any]:
    """Train on every session but *held_out*, and test on it — one fold."""
    return {
        "train.split.holdout_sessions": [str(held_out.source)],
        "train.split.train_fraction": 1.0,
        "train.split.val_fraction": 0.0,
        "train.split.test_fraction": 0.0,
        "train.run_name": run_name,
    }


def fold_id(held_out: SessionSpec) -> str:
    """The session *id*, not its stem: one project's sessions are routinely files of one name."""
    return session_id(held_out.source)


def run_cell(
    config_path: Path, cell: Cell, held_out: SessionSpec, prefix: str, common: dict[str, Any] | None = None
) -> tuple[RunResult, float]:
    """One training run; returns it with its wall-clock seconds."""
    overrides = eto.segment.as_overrides(
        {
            **(common or {}),
            **cell.overrides,
            **fold_overrides(held_out, f"{prefix}/{cell.tag}/fold-{fold_id(held_out)}"),
        }
    )
    started = time.perf_counter()
    result = eto.segment.Project(config_path, *overrides).train()
    return result, time.perf_counter() - started


def row_for(cell: Cell, held_out: SessionSpec, result: RunResult, seconds: float) -> dict[str, Any]:
    """One results row: what was run, how long it took, and what it scored."""
    row: dict[str, Any] = {
        "cell": cell.tag,
        **cell.describe,
        "fold": fold_id(held_out),
        "run_dir": str(result.run_dir),
        "epochs": result.best_epoch,
        "seconds": round(seconds, 1),
    }
    for stage in ("raw", "postprocessed"):
        for key, value in (result.test_metrics or {}).get(stage, {}).items():
            if key != "classwise":
                row[f"{stage}.{key}"] = value
    return row


def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t") if path.is_file() else pd.DataFrame()


def already_done(results: pd.DataFrame, cell: Cell, fold: str) -> bool:
    if results.empty or "cell" not in results:
        return False
    return bool(((results["cell"] == cell.tag) & (results["fold"] == fold)).any())


def sweep(
    config_path: Path,
    cells: list[Cell],
    folds: list[SessionSpec],
    out_dir: Path,
    prefix: str,
    common: dict[str, Any] | None = None,
    after: Any = None,
) -> pd.DataFrame:
    """Run every (cell, fold) not already recorded, appending to ``results.tsv``.

    *after* is called with ``(cell, held_out, result)`` once a run finishes and
    returns extra columns for its row — that is where an experiment puts work
    it wants done on the trained model rather than during training, such as
    re-scoring one run under several post-processing settings.
    """
    path = out_dir / RESULTS_FILE
    results = load_results(path)
    todo = [(c, f) for c in cells for f in folds if not already_done(results, c, fold_id(f))]
    logger.info("%s: %d cells x %d folds, %d runs to do", prefix, len(cells), len(folds), len(todo))
    for n, (cell, held_out) in enumerate(todo, start=1):
        logger.info("[%d/%d] %s, holding out %s", n, len(todo), cell.tag, fold_id(held_out))
        result, seconds = run_cell(config_path, cell, held_out, prefix, common)
        row = row_for(cell, held_out, result, seconds)
        if after is not None:
            row.update(after(cell, held_out, result))
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(path, sep="\t", index=False)
        logger.info("    %s (%.0f s)", cell.tag, seconds)
    return results


def summarise(results: pd.DataFrame, column: str, by: str = "cell") -> pd.DataFrame:
    """Mean +/- sd of *column* over folds, one row per cell, best first."""
    if results.empty or column not in results:
        return pd.DataFrame()
    table = results.groupby(by)[column].agg(["mean", "std", "count"]).reset_index()
    return table.sort_values("mean", ascending=False)


def fold_dots(ax, results: pd.DataFrame, column: str, order: list[str] | None = None) -> None:
    """Every fold as a dot over its cell's mean — n is four, so show it."""
    cells = order or sorted(results["cell"].unique())
    for x, tag in enumerate(cells):
        values = results.loc[results["cell"] == tag, column].dropna().to_numpy()
        if not len(values):
            continue
        ax.scatter(np.full(len(values), x), values, s=18, color="tab:blue", zorder=3)
        ax.hlines(values.mean(), x - 0.3, x + 0.3, color="k", zorder=4)
    ax.set_xticks(range(len(cells)), cells, rotation=30, fontsize=7, ha="right")
    ax.set_ylabel(column)


def dataset_rate(results: pd.DataFrame) -> float:
    """The dataset's own sampling rate, read off a run rather than assumed."""
    import yaml

    run_dir = Path(results["run_dir"].iloc[0])
    return float(yaml.safe_load((run_dir / "columns.yaml").read_text(encoding="utf-8"))["fs"])
