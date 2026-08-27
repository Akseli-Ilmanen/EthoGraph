"""Which loss terms earn their place — per individual, per architecture, cross-validated.

The objective is a sum of up to three terms (``docs/add_to_docs_later/segment/config.md``,
*Losses*): the frame cross-entropy, the consistency (smoothing) term it
carries at ``train.loss.alpha``, and the circle metric-learning term at
``train.circle.weight``. This bench trains every architecture in
:data:`ARCHITECTURES` under three arms and asks what each term buys:

=============  ==================================================
``all``        cross-entropy + smoothing + circle
``no_smooth``  ``train.loss.alpha = 0`` — no consistency term
``no_circle``  ``train.circle.weight = 0`` — no circle term
=============  ==================================================

Both weights are pinned in every arm rather than read from the project
config: the "with" values are :data:`SMOOTHING_ALPHA` and
:data:`CIRCLE_WEIGHT`, so what an arm trained with is in this file and in
the run's ``config.yaml``, nowhere else.

**One model per individual.** Each entry of :data:`INDIVIDUALS` is a config
beside the project's — ``data/crow1.yaml`` inherits ``project.yaml`` through
``base:`` and lists only that individual's sessions under its own
``features.name`` — and every cell is ``Project.cross_validate()`` on it:
leave-one-session-out, so a cell's score is the mean over sessions the
model never saw, with one dot per session in every figure. The stem of the
config is how the individual is named in every output (``crow 1``); nothing
here knows anything else about it.

**Resumable, fold by fold.** A cross-validation's folds are ordinary runs
under ``runs/cv_{run_name}/fold-{session}_{timestamp}/``, and each writes
its ``test_metrics.yaml`` + ``test_eval.npz`` as it finishes. A cell whose
sessions all have such a fold is read back, never retrained; a cell with some
missing holds out only those, through ``cross_validate(folds=...)``. Every
cell is sequential — they all want the GPU.

The output is ``data/bench_loss.pdf`` (:func:`ethograph.segment.plotting.write_factorial_pdf`):
the summary grids first — segmental F1 at every threshold and the frame-level
scores, one row per individual plus all of them pooled, architectures along
x and a bar per arm — then one page per individual × architecture with the
three arms' IoU distributions, boundary deltas and class-wise F1 side by side.
``data/bench_loss.tsv`` holds every fold's numbers.

    python scripts/bench.py                 # train what is missing, then draw
    python scripts/bench.py --report-only   # draw from what has finished
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import ethograph as eto
from ethograph.labels.onset_model import session_id
from ethograph.segment.crossval import cross_validation_name_for
from ethograph.segment.metrics import EVAL_ARRAYS_FILE, TEST_METRICS_FILE
from ethograph.segment.plotting import FactorCell, load_run_eval, write_factorial_pdf
from ethograph.segment.samples import ClassTable

#: Where the configs live. Set BENCH_CONFIG_DIR to point at another machine's copy.
CONFIG_DIR = Path(os.environ.get("BENCH_CONFIG_DIR") or Path(__file__).resolve().parents[1] / "data")

#: One config per individual, ``{stem}.yaml`` in :data:`CONFIG_DIR`; the stem
#: names the individual everywhere (``crow1`` → ``crow 1``).
INDIVIDUALS = ["crow1", "crow2", "crow3"]

#: Compared at their upstream defaults (``model.params: {}``): the bench asks
#: about the loss, not the hyperparameters — those are ``bench_search.py``'s.
ARCHITECTURES = ["mlp", "c2f_tcn", "c2f_transformer", "mstcn"]

#: ``train.loss.alpha`` of the arms that keep the smoothing term. MS-TCN's
#: published λ, and what the archived CETNet script trained with
#: (``segment/archive/cetnet_encoder.py``, ``0.15 * mse_loss``). DLC2Action's
#: YAML default is ``0.001``, at which the term barely registers — an ablation
#: against *that* would compare two nearly identical objectives.
SMOOTHING_ALPHA = 0.15

#: ``train.circle.weight`` of the arms that keep the circle term — the same
#: script's ``0.001 * CircleLoss(m=0.25, gamma=128)``; ``m`` and ``gamma`` stay
#: at those defaults.
CIRCLE_WEIGHT = 0.001

#: The arms. Every weight is spelled in every arm, so no arm inherits one.
LOSSES: dict[str, dict[str, Any]] = {
    "all": {"train.loss.alpha": SMOOTHING_ALPHA, "train.circle.weight": CIRCLE_WEIGHT},
    "no_smooth": {"train.loss.alpha": 0.0, "train.circle.weight": CIRCLE_WEIGHT},
    "no_circle": {"train.loss.alpha": SMOOTHING_ALPHA, "train.circle.weight": 0.0},
}

#: Appended to every run name, so one bench's folds stay together under ``runs/``.
SUFFIX = "loss"

OUTPUT = CONFIG_DIR / "bench_loss.pdf"
TABLE = CONFIG_DIR / "bench_loss.tsv"

logger = logging.getLogger("bench")


def display_name(individual: str) -> str:
    """``crow1`` → ``crow 1`` — the config stem, as the figures spell it."""
    return re.sub(r"(?<=\D)(\d+)$", r" \1", individual)


def run_name(individual: str, architecture: str, loss: str) -> str:
    """The cell's base run name; its cross-validation is ``cv_`` + this."""
    return f"{individual}_{architecture}_{loss}_{SUFFIX}"


def project_for(individual: str, architecture: str, loss: str) -> eto.segment.Project:
    """The individual's config with the cell's architecture, loss weights and run name pinned."""
    overrides = eto.segment.as_overrides(
        {
            "model.architecture": architecture,
            "train.run_name": run_name(individual, architecture, loss),
            **LOSSES[loss],
        }
    )
    return eto.segment.Project(CONFIG_DIR / f"{individual}.yaml", *overrides)


def finished_folds(project: eto.segment.Project) -> dict[str, Path]:
    """Session source → its newest fold run that finished its test evaluation.

    A fold interrupted before ``test_eval.npz`` does not count, so rerunning
    the bench trains it again rather than reading half a result.
    """
    config = project.config
    folds_dir = config.runs_dir / cross_validation_name_for(config)
    done: dict[str, Path] = {}
    for spec in config.sessions:
        candidates = sorted(folds_dir.glob(f"fold-{session_id(spec.source)}_*"))
        finished = [d for d in candidates if (d / TEST_METRICS_FILE).is_file() and (d / EVAL_ARRAYS_FILE).is_file()]
        if finished:
            done[str(spec.source)] = finished[-1]
    return done


def cross_validate_cell(individual: str, architecture: str, loss: str) -> dict[str, Path]:
    """Train the cell's missing folds, if any, and return every session's finished fold."""
    project = project_for(individual, architecture, loss)
    sessions = [str(s.source) for s in project.config.sessions]
    done = finished_folds(project)
    missing = [s for s in sessions if s not in done]
    label = f"{display_name(individual)} / {architecture} / {loss}"
    if not missing:
        logger.info("[%s] every fold finished — read back", label)
        return done
    logger.info("[%s] %d of %d folds to train: %s", label, len(missing), len(sessions), LOSSES[loss])
    project.cross_validate(folds=missing)
    done = finished_folds(project)
    still_missing = [s for s in sessions if s not in done]
    if still_missing:
        raise RuntimeError(
            f"[{label}] cross_validate returned, but these folds wrote no test evaluation: {still_missing}"
        )
    return done


def collect() -> tuple[list[FactorCell], pd.DataFrame, ClassTable | None]:
    """Every cell with at least one finished fold, its folds loaded, plus one row per fold."""
    cells: list[FactorCell] = []
    rows: list[dict[str, Any]] = []
    classes: ClassTable | None = None
    for individual in INDIVIDUALS:
        for architecture in ARCHITECTURES:
            for loss in LOSSES:
                project = project_for(individual, architecture, loss)
                done = finished_folds(project)
                if not done:
                    logger.warning("%s / %s / %s: no finished fold", display_name(individual), architecture, loss)
                    continue
                folds = []
                for spec in project.config.sessions:
                    run_dir = done.get(str(spec.source))
                    if run_dir is None:
                        continue
                    e = load_run_eval(run_dir, name=session_id(spec.source))
                    folds.append(e)
                    row: dict[str, Any] = {
                        "individual": display_name(individual),
                        "architecture": architecture,
                        "loss": loss,
                        "session": e.name,
                        "run_dir": str(run_dir),
                        "train_seconds": e.train_seconds,
                    }
                    for stage, metrics in (("raw", e.raw), ("postprocessed", e.processed)):
                        row.update({f"{stage}.{k}": v for k, v in metrics.items() if k != "classwise"})
                    rows.append(row)
                    if classes is None:
                        classes = ClassTable.from_dict(
                            yaml.safe_load((run_dir / "classes.yaml").read_text(encoding="utf-8"))
                        )
                cells.append(FactorCell(display_name(individual), architecture, loss, folds))
    return cells, pd.DataFrame(rows), classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--report-only", action="store_true", help="draw from the folds that finished; train nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.report_only:
        for individual in INDIVIDUALS:
            for architecture in ARCHITECTURES:
                for loss in LOSSES:
                    cross_validate_cell(individual, architecture, loss)

    cells, table, classes = collect()
    if not cells or classes is None:
        raise SystemExit("No finished folds under any individual's runs/ — nothing to draw.")
    table.to_csv(TABLE, sep="\t", index=False)
    select_on = eto.segment.Project(CONFIG_DIR / f"{INDIVIDUALS[0]}.yaml").config.train.select_on
    column = f"postprocessed.{select_on}"
    summary = (
        table.groupby(["individual", "architecture", "loss"])[column]
        .mean()
        .unstack("loss")
        .reindex(columns=list(LOSSES))
    )
    print(f"\nMean post-processed {select_on} over held-out sessions:\n{summary.round(1).to_string()}\n")

    title = f"Loss ablation — {len(INDIVIDUALS)} individuals × {len(ARCHITECTURES)} architectures, cross-validated"
    path = write_factorial_pdf(
        OUTPUT,
        cells,
        classes,
        title=title,
        classwise_key=select_on if select_on.startswith("f1@") else None,
    )
    logger.info("Wrote %s and %s", path, TABLE)


if __name__ == "__main__":
    main()
