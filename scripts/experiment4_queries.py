"""Experiment 4 — a segment-level objective (BaFormer's query head).

Frame-wise losses optimise a different thing from the segmental metric being
reported. A query-based head predicts segments as *instances* with one-to-one
matching, so the objective is IoU-shaped from the start. BaFormer adds a global
class-agnostic boundary query that cuts the timeline into continuous proposals
and classifies each boundary-delimited span by query voting — structurally the
same as this project's pipeline (cut at changepoints, then classify), learned
end to end. It runs on an ASFormer encoder backbone, so the validated encoder is
retained and only the head changes.

Written into ``{project}/experiment4/``:

**The head** — number of instance queries, crossed with whether the frame loss
is kept as an auxiliary. Upstream trains on the set objective alone
(``train.frame_weight = 0``); keeping the frame loss is cheap and is the kind of
thing that helps when the training set is four sessions rather than fifty, so it
is a condition rather than an assumption. The query count starts near twice the
worst trial's segment count, which :func:`worst_segment_count` reads off the
materialised dataset rather than guessing.

**The stress test the paper's caveat asks for.** Query-based methods assume a
reasonable number of segments per sequence, and a trial with fifteen consecutive
``toss`` syllables is exactly the case that breaks one-to-many matching. So the
per-class F1@90 of every run is written out, and ``CAVEAT_CLASSES`` names the
classes to watch — the repetitive ones and the rare ones. The aggregate hides
them; this table does not.

    python scripts/experiment4_queries.py

Compare against Experiment 3's models on F1@90: they share the encoder, the
features and the folds, so the only difference is what sits on top.

``results.tsv`` is append-only and keyed by (cell, fold), so the sweep is
resumable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ethograph as eto  # noqa: E402
from _sweep import Cell, fold_dots, summarise, sweep  # noqa: E402
from ethograph.segment.config import SessionSpec  # noqa: E402
from ethograph.segment.dataset import MaterialisedStore  # noqa: E402
from ethograph.segment.metrics import metric_key  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "data" / "model" / "project.yaml"

#: The encoder, kept identical to what Experiment 3 wraps so the comparison is
#: about the head. ``num_decode`` defaults to one level per encoder layer.
ENCODER: dict[str, Any] = {"num_f_maps": 64, "nheads": 4, "dropout": 0.1}

#: Multiples of the worst trial's segment count to try as the query budget.
#: The paper's rule of thumb is 2x; 1.5x asks whether that is generous and 3x
#: whether it is tight.
QUERY_FACTORS = [1.5, 2.0, 3.0]

#: Whether the frame-wise loss is kept as an auxiliary beside the set loss.
#: ``0`` is upstream's setting.
FRAME_WEIGHTS = [0.0, 1.0]

#: Classes the paper's caveat is about: long repetitive runs stress one-to-many
#: matching, and rare classes have too few instances for a query to specialise.
#: Named by their label id in ``mapping.txt``.
CAVEAT_CLASSES: list[str] = ["toss", "nodding", "snapPellet"]

SCREEN_FOLDS: int | None = 1
CONFIRM_TOP: int = 2
COMMON: dict[str, Any] = {}

PRIMARY = metric_key(0.9)

logger = logging.getLogger("experiment4")


def worst_segment_count(config) -> int:
    """The most segments any materialised sample contains.

    The query budget has to cover this or the set criterion refuses the batch,
    so it is read off the dataset rather than picked. Counts maximal runs of a
    class, background included — the same definition
    :func:`~ethograph.segment.queries.segment_targets` uses.
    """
    store = MaterialisedStore.open(config.data_dir)
    worst = 0
    for key in store.keys:
        _x, y = store.load(key)
        worst = max(worst, int((np.diff(y) != 0).sum()) + 1)
    return worst


def head_cells(worst: int) -> list[Cell]:
    """Query budget x frame-loss weight."""
    cells = []
    for factor in QUERY_FACTORS:
        queries = int(np.ceil(factor * worst))
        for frame_weight in FRAME_WEIGHTS:
            cells.append(
                Cell(
                    tag=f"q{queries}_fw{frame_weight:g}",
                    overrides={
                        "model.architecture": "baformer",
                        "model.params": {**ENCODER, "num_queries": queries},
                        "train.frame_weight": frame_weight,
                    },
                    describe={"num_queries": queries, "factor": factor, "frame_weight": frame_weight},
                )
            )
    return cells


def classwise(run_dir: Path) -> dict[str, float]:
    """Per-class F1@90 of a finished run, named rather than indexed.

    The aggregate hides exactly the classes this experiment's caveat is about,
    so every class gets its own column and the write-up does not have to open
    each run's ``test_metrics.yaml``.
    """
    metrics = yaml.safe_load((run_dir / "test_metrics.yaml").read_text(encoding="utf-8"))
    classes = yaml.safe_load((run_dir / "classes.yaml").read_text(encoding="utf-8"))
    names = classes["names"]
    out: dict[str, float] = {}
    for stage in ("raw", "postprocessed"):
        for index, scores in (metrics.get(stage, {}).get("classwise") or {}).items():
            name = names[int(index)] if int(index) < len(names) else str(index)
            out[f"class.{stage}.{name}"] = scores.get(PRIMARY, float("nan"))
    return out


def write_figure(results: pd.DataFrame, path: Path) -> Path:
    """Overall F1@90 per cell, and the classes the caveat is about."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fold_dots(axes[0], results, f"postprocessed.{PRIMARY}")
    axes[0].set_title(f"Overall {PRIMARY}")

    columns = [c for c in results.columns if c.startswith("class.postprocessed.")]
    watched = [c for c in columns if c.rsplit(".", 1)[-1] in CAVEAT_CLASSES] or columns
    for x, column in enumerate(watched):
        values = results[column].dropna().to_numpy()
        if not len(values):
            continue
        axes[1].bar(x, values.mean(), 0.6, color="tab:grey")
        axes[1].scatter(np.full(len(values), x), values, s=16, color="tab:blue", zorder=3)
    axes[1].set_xticks(range(len(watched)), [c.rsplit(".", 1)[-1] for c in watched], rotation=30, ha="right")
    axes[1].set_ylabel(PRIMARY)
    axes[1].set_title("Per class — the matching caveat")

    fig.suptitle("Experiment 4 — the query head")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    project = eto.segment.Project(CONFIG)
    out_dir = project.root / "experiment4"
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions: list[SessionSpec] = list(project.config.sessions)

    project.materialise()
    worst = worst_segment_count(project.config)
    logger.info("The worst materialised sample holds %d segments; query budgets follow from that", worst)

    def after(cell: Cell, held_out: SessionSpec, result) -> dict[str, float]:
        return classwise(result.run_dir)

    cells = head_cells(worst)
    results = sweep(CONFIG, cells, sessions[:SCREEN_FOLDS], out_dir, "exp4", COMMON, after)
    ranking = summarise(results, f"postprocessed.{PRIMARY}")
    print(ranking.to_string(index=False))

    if CONFIRM_TOP and len(sessions) > 1:
        best = [c for c in cells if c.tag in set(ranking.head(CONFIRM_TOP)["cell"])]
        results = sweep(CONFIG, best, sessions, out_dir, "exp4", COMMON, after)

    write_figure(results, out_dir / "queries_f1@90.pdf")
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
