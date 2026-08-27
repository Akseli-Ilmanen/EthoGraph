"""Cross-validation: one fold per session, each predicting the session it never saw.

Stage 2 of the workflow, once a {mod}`search <ethograph.segment.search>` has
settled the hyperparameters. Leave-one-session-out: fold *i* trains on every
session but the *i*-th and then runs inference over that held-out session, so
each session ends up with a prediction set produced by a model that never saw
one of its frames.

That last part is the point. Predictions land in the GUI's own labels format
beside the session, under its own ``labels/`` folder
(``labels/predictions_{run}_{timestamp}/{stem}_predictions.tsv``), so you load them
next to the curated labels and look at *where* the model is still wrong —
which classes, which trials, which boundaries — rather than at one aggregate
number. A random trial split cannot give you that: its test trials share a
recording day, lighting and animal with the trials the model trained on, and
they are scattered across sessions rather than making up one you can open.

Everything is written under ``{root}/cross_validation/{name}/``::

    folds.tsv  # one row per fold: session, run, test metrics, prediction path
    eval_comparison.pdf  # class-wise F1 / IoU / boundary deltas across folds, one dot per fold
    crossval.log

The folds themselves are ordinary runs, nested under ``runs/{name}/`` so they
do not bury the runs you trained by hand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from ethograph.labels.onset_model import session_id
from ethograph.segment.config import (
    SegmentConfig,
    SessionSpec,
    apply_overrides,
    as_overrides,
    config_from_dict,
    config_to_dict,
)
from ethograph.segment.inference import inference
from ethograph.segment.materialise import COLUMNS_FILE, materialise
from ethograph.segment.metrics import EVAL_ARRAYS_FILE
from ethograph.segment.samples import ClassTable
from ethograph.segment.train import RunResult, run_name_for, train
from ethograph.utils.logging import log_to_file

logger = logging.getLogger(__name__)

FOLDS_FILE = "folds.tsv"


@dataclass
class Fold:
    """One held-out session and what training on the rest produced."""

    session: str
    source: Path
    run_dir: Path
    best_epoch: int
    #: ``test_metrics.yaml`` of the fold: the held-out session, raw and post-processed.
    metrics: dict[str, Any] | None
    #: The prediction set written beside the held-out session (``None`` if ``predict=False``).
    predictions: Path | None


def cross_validation_name_for(config: SegmentConfig) -> str:
    """The directory a cross-validation writes into — ``{root}/cross_validation/{name}``."""
    return f"cv_{run_name_for(config)}"


def _fold_config(config: SegmentConfig, held_out: SessionSpec, name: str, val_fraction: float) -> SegmentConfig:
    """*config* with *held_out* pinned as the test session and the fold's run name."""
    overrides = as_overrides(
        {
            "train.split.holdout_sessions": [str(held_out.source)],
            "train.split.train_fraction": 1.0 - val_fraction,
            "train.split.val_fraction": val_fraction,
            "train.split.test_fraction": 0.0,
            # The session *id*, not its stem: sessions of one project are
            # routinely files of the same name in different session folders,
            # and a fold has to be tellable from the other three.
            "train.run_name": f"{name}/fold-{session_id(held_out.source)}",
        }
    )
    base_dir = config.config_path.parent if config.config_path else config.root
    return config_from_dict(
        apply_overrides(config_to_dict(config), overrides), base_dir, config_path=config.config_path
    )


def cross_validate(
    config: SegmentConfig,
    folds: Iterable[str | Path] | None = None,
    val_fraction: float = 0.0,
    predict: bool = True,
) -> pd.DataFrame:
    """Leave-one-session-out over *folds*, returning one row per fold.

    *folds* names the sessions to hold out — by path or by source stem —
    defaulting to every session, which is the full cross-validation. Naming
    two of six is how you compare parameter sets without paying for all six.

    *val_fraction* carves a validation slice out of the *training* sessions
    of each fold, to select ``best.pt``. It defaults to ``0``: after a
    search the hyperparameters (``epochs`` included) are settled, every
    remaining trial is worth training on, and ``best.pt`` is the last epoch.

    With *predict*, each fold also writes a prediction set beside its
    held-out session — the reason to run this rather than a random split.
    """
    if config.train.split.holdout_sessions:
        raise ValueError(
            "train.split.holdout_sessions is already set — that is one fold, pinned by hand. "
            "Cross-validation writes it per fold; leave it out of the config."
        )
    if len(config.sessions) < 2:
        raise ValueError(
            f"Leave-one-session-out needs at least two sessions, this config has {len(config.sessions)}. "
            "With one session, the ratio split in train.split is the honest option."
        )

    held_out = config.select_sessions(folds)
    name = cross_validation_name_for(config)
    out_dir = config.cross_validation_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (config.data_dir / COLUMNS_FILE).is_file():
        logger.info("No materialised dataset at %s — materialising once for every fold", config.data_dir)
        materialise(config)

    with log_to_file(out_dir / "crossval.log"):
        logger.info(
            "Cross-validation %r: %d of %d sessions held out, val_fraction=%g",
            name,
            len(held_out),
            len(config.sessions),
            val_fraction,
        )
        results: list[Fold] = []
        for n, spec in enumerate(held_out, start=1):
            logger.info("[fold %d/%d] holding out %s", n, len(held_out), session_id(spec.source))
            fold_config = _fold_config(config, spec, name, val_fraction)
            result: RunResult = train(fold_config)
            predictions = None
            if predict:
                written = inference(fold_config, run=result.run_dir, sessions=[str(spec.source)])
                predictions = written[0]
            results.append(
                Fold(
                    session=session_id(spec.source),
                    source=spec.source,
                    run_dir=result.run_dir,
                    best_epoch=result.best_epoch,
                    metrics=result.test_metrics,
                    predictions=predictions,
                )
            )

        table = _fold_table(results)
        table.to_csv(out_dir / FOLDS_FILE, sep="\t", index=False)
        logger.info("Wrote %s", out_dir / FOLDS_FILE)
        _write_fold_comparison(out_dir, results)
        return table


def _write_fold_comparison(out_dir: Path, folds: list[Fold]) -> None:
    """The cross-fold comparison figure — one dot per fold's held-out test."""
    from ethograph.segment.plotting import load_run_eval, write_comparison_pdf

    evals = [load_run_eval(f.run_dir, name=f.session) for f in folds if (f.run_dir / EVAL_ARRAYS_FILE).is_file()]
    if len(evals) < 2:
        logger.info(
            "Fewer than 2 folds wrote %s (test_fraction may be 0, or a fold never reached a test evaluation) — "
            "skipping the comparison figure",
            EVAL_ARRAYS_FILE,
        )
        return
    classes = ClassTable.from_dict(yaml.safe_load((folds[0].run_dir / "classes.yaml").read_text(encoding="utf-8")))
    path = write_comparison_pdf(
        out_dir / "eval_comparison.pdf", evals, classes, title=f"Cross-validation — {len(evals)} folds"
    )
    logger.info("Wrote %s", path)


def _fold_table(folds: list[Fold]) -> pd.DataFrame:
    """One row per fold: its session, its run, and the metrics on the session it never saw."""
    rows = []
    for fold in folds:
        row: dict[str, Any] = {
            "session": fold.session,
            "run": fold.run_dir.name,
            "run_dir": str(fold.run_dir),
            "best_epoch": fold.best_epoch,
            "predictions": str(fold.predictions) if fold.predictions else None,
        }
        for stage in ("raw", "postprocessed"):
            for key, value in (fold.metrics or {}).get(stage, {}).items():
                if key != "classwise":
                    row[f"{stage}.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
