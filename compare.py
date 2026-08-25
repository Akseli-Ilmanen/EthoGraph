"""Draw the comparison figure over runs that already finished.

``bench.py`` compares architectures by training them; this compares them by
*reading what they wrote*. Every run writes ``test_metrics.yaml`` (the scalars
and class-wise F1) and ``test_eval.npz`` (the matched-segment IoUs and
onset/offset deltas) the moment it finishes, so a bench that crashed — or one
still running in another terminal — has already left everything this needs on
disk. Nothing is retrained.

The figure is ``segment/plotting.py``'s ``write_comparison_pdf``: the IoU
illustration, overall metrics, IoU distribution with a TP/FP/FN inset, the
boundary-delta histogram, and class-wise F1 raw vs post-processed. It is the
successor to the archived ``segment/archive/eval_plotting.py``'s
``plot_metrics_best_model``, reading a run directory instead of that script's
``test_results_epoch{N}.npy``.

``search()`` and ``cross_validate()`` draw the same figure over their own
trials and folds (``searches/{name}/eval_comparison.pdf``,
``cross_validation/{name}/eval_comparison.pdf``). This one is for everything
else: a search's winner against another search's winner, a hand-trained run
against a swept cell, whatever finished before the crash.

    python compare.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

import ethograph as eto
from ethograph.segment.metrics import EVAL_ARRAYS_FILE
from ethograph.segment.plotting import load_run_eval, write_comparison_pdf
from ethograph.segment.samples import ClassTable

CONFIG = Path(r"C:\Users\aksel\Documents\Code\ethograph\data\model\project.yaml")

#: Which runs to compare, as paths relative to ``runs/`` — a run trained by
#: hand is one level deep (``asformer_kin_v1_20260824_1200``), a search trial
#: or a swept cell two (``asformer_enc_alpha_kin_v1/trial003_…``). Empty
#: compares **every** run that wrote ``test_eval.npz``, which for a finished
#: bench is every trial of every variant — informative but crowded, so name
#: the winners here once you know them (``searches/{name}/trials.tsv`` has
#: each trial's score and run dir; ``bench_cells.tsv`` has each swept cell's).
RUNS: list[str] = []

OUTPUT = CONFIG.with_name("compare.pdf")

logger = logging.getLogger("compare")


def run_dirs(runs_dir: Path) -> list[Path]:
    """The run directories to compare — :data:`RUNS` if it names any, else all of them."""
    if RUNS:
        chosen = [runs_dir / name for name in RUNS]
        missing = [d for d in chosen if not (d / EVAL_ARRAYS_FILE).is_file()]
        if missing:
            raise FileNotFoundError(
                f"No {EVAL_ARRAYS_FILE} in {[str(d) for d in missing]} — that run never finished its "
                "test evaluation (it may have been interrupted, or trained with train.split.test_fraction=0)."
            )
        return chosen
    return sorted(p.parent for p in runs_dir.rglob(EVAL_ARRAYS_FILE))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = eto.segment.Project(CONFIG).config
    dirs = run_dirs(config.runs_dir)
    if len(dirs) < 2:
        raise SystemExit(f"Need at least two finished runs to compare, found {len(dirs)} under {config.runs_dir}.")

    classes = ClassTable.from_dict(yaml.safe_load((dirs[0] / "classes.yaml").read_text(encoding="utf-8")))
    evals = [load_run_eval(d, name=str(d.relative_to(config.runs_dir))) for d in dirs]
    for e in evals:
        logger.info("%-50s %s", e.name, {k: round(v, 2) for k, v in e.processed.items() if k != "classwise"})

    path = write_comparison_pdf(OUTPUT, evals, classes, title=f"{len(evals)} runs compared")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
